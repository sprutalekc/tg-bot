import asyncio
import base64
import io
import random
import re
import time
import os
from collections import defaultdict

from PIL import Image
from aiohttp import web
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from openai import AsyncOpenAI

# ───────────────────────────── env ──────────────────────────────
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not BOT_TOKEN or not OPENAI_API_KEY:
    raise ValueError("Проверьте BOT_TOKEN и OPENAI_API_KEY в настройках Render")

# ───────────────────────────── init ─────────────────────────────
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

MODEL = "gpt-4o-mini"
# Используем память, чтобы Render не терял контекст при чтении битых файлов
user_context = {} 

# ───────────────────────────── prompts ──────────────────────────

SYSTEM_PROMPT = """
Ты — учебный Telegram-ассистент по имени Трегубчик.

Характер:
Строгий преподаватель с живым характером. Точный, без воды. Иногда — неожиданно смешной,
но юмор у тебя специфический: мрачноватый, абсурдный, иногда на грани. Не пошлый, не детский.
Шутишь редко, но метко — и именно тогда, когда студент этого не ждёт.
Не оскорбляешь, не хамишь, но можешь поддеть. Мат не используешь.

Примеры твоего юмора (для вдохновения):
- "Ошибка в знаке. Классика. Из-за таких ошибок ракеты летят не туда."
- "Верно. Можешь идти праздновать. Или решить следующую задачу — на твоё усмотрение."
- "Это называется расходящийся ряд. Он уходит в бесконечность, как и некоторые студенты на пересдаче."

Правила форматирования (строго):
- НЕ используй LaTeX, символы $, \\frac, \\times и подобные.
- НЕ используй Markdown: никаких **, *, ##, `, _.
- Формулы пиши понятными Unicode-символами: √, ², ³, ±, ≠, ≈, ∞, ∫, π.
- Дроби пиши через слэш и скобки: (x+1)/(2y).
- Текст как обычное сообщение — никакой разметки.

Правила ответа:
1. Кратко распознай условие задачи.
2. Реши задачу по шагам.
3. Простая задача — короткий ответ. Сложная — пошагово.
4. Если есть поля для ввода — в конце отдельно перечисли значения.
5. Если фото плохо видно — скажи об этом в своём стиле.
"""

FOLLOWUP_SYSTEM = """
Ты — учебный ассистент Трегубчик. Строгий, точный, с живым характером.
Отвечай на вопрос пользователя по предыдущему решению.

Про стиль:
- Никогда не заканчивай фразами "Если есть вопросы — спрашивай", "Не стесняйся", "Удачи!".
- Если пользователь благодарит или говорит "понял/ок/спасиб" — отвечай 1-2 фразами.
  Преимущественно сухо, с иронией или коротким напутствием.
  Примеры: "Разобрались. На экзамене я рядом не буду.", "Принято.", "Ещё 40 таких задач — и можно считать себя человеком."
- Юмор — мрачноватый, абсурдный, неожиданный. Редко.
- Форматирование: только обычный текст и Unicode (√, ², ±), никакого LaTeX и Markdown.
"""

# ───────────────────────────── фразы ────────────────────────────

PHOTO_ACCEPT_PHRASES = [
    "Принял, смотрю...", "Фото получил. Считаю...", "Разбираю задачу...",
    "Сейчас посмотрим...", "Глянем, что тут у нас...", "Опять задачи? Ладно, жди."
]

THINKING_PHRASES = [
    "Смотрю...", "Считаю...", "Разбираю...", "Секунду...", "Думаю...",
    "Так-так...", "Где-то я это уже видел...", "Интересно."
]

# ───────────────────────────── utils ────────────────────────────

def clean_response(text: str) -> str:
    """Удаляет Markdown и остатки LaTeX"""
    text = re.sub(r'\*\*|\*|__|#', '', text)
    text = re.sub(r'\\(?:frac|sqrt|cdot|times|pm|leq|geq|neq|approx|infty)', '', text)
    return text.strip()

def compress_image(image_bytes: bytes) -> str:
    """Сжатие фото (экономия токенов и трафика)"""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=75, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

async def send_long_message(message: Message, text: str):
    text = clean_response(text)
    for i in range(0, len(text), 4000):
        await message.answer(text[i : i + 4000])

async def generate_with_retry(messages):
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=2000,
            temperature=0.4
        )
        return response
    except Exception as e:
        print(f"OpenAI Error: {e}")
        return None

async def send_typing_while(message: Message, coro):
    task = asyncio.create_task(coro)
    while not task.done():
        await bot.send_chat_action(message.chat.id, "typing")
        await asyncio.sleep(3)
    return await task

# ───────────────────────────── handlers ──────────────────────────

@dp.message(F.text == "/start")
async def start(message: Message):
    name = message.from_user.first_name or "студент"
    await message.answer(f"Здравствуй, {name}. Я Трегубчик. Присылай фото задачи, разберём.")

@dp.message(F.text == "/clear")
async def clear(message: Message):
    user_context.pop(message.from_user.id, None)
    await message.answer("Контекст сброшен. Как будто ничего и не было.")

@dp.message(F.photo)
async def solve_photo(message: Message):
    uid = message.from_user.id
    await message.answer(random.choice(PHOTO_ACCEPT_PHRASES))

    try:
        photo = message.photo[-1]
        file_info = await bot.get_file(photo.file_id)
        photo_bytes = await bot.download_file(file_info.file_path)
        
        image_b64 = compress_image(photo_bytes.read())

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Реши задачу с фото:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }
        ]

        response = await send_typing_while(message, generate_with_retry(messages))
        if response:
            answer = response.choices[0].message.content
            user_context[uid] = answer
            await send_long_message(message, answer)
        else:
            await message.answer("OpenAI молчит. Видимо, задачка слишком сложная. Попробуй позже.")

    except Exception as e:
        await message.answer("Не смог прочитать фото. Попробуй еще раз.")

@dp.message(F.text)
async def handle_text(message: Message):
    uid = message.from_user.id

    if uid in user_context:
        await message.answer(random.choice(THINKING_PHRASES))
        
        messages = [
            {"role": "system", "content": FOLLOWUP_SYSTEM + f"\nПредыдущее решение:\n{user_context[uid]}"},
            {"role": "user", "content": message.text}
        ]

        response = await generate_with_retry(messages)
        if response:
            answer = response.choices[0].message.content
            user_context[uid] += f"\nСтудент: {message.text}\nТрегубчик: {answer}"
            await send_long_message(message, answer)
        else:
            await message.answer("Я тебя не понял. Попробуй переформулировать.")
    else:
        await message.answer("Сначала пришли фото задачи, потом будем разговаривать.")

# ───────────────────────────── server ─────────────────────────────

async def main():
    app = web.Application()
    app.router.add_get("/", lambda r: web.Response(text="Running"))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", int(os.getenv("PORT", 8080)))
    asyncio.create_task(site.start())
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
