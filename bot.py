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
    raise ValueError("Проверьте переменные BOT_TOKEN и OPENAI_API_KEY в Render")

# ───────────────────────────── init ─────────────────────────────
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

MODEL = "gpt-4o-mini"
# Храним контекст в памяти, чтобы Render не терял его при чтении битых файлов
user_context = {}

# ───────────────────────────── prompts ──────────────────────────

SYSTEM_PROMPT = """
Ты — учебный ассистент Трегубчик. Строгий, сухой преподаватель с мрачным юмором.

МАТЕМАТИКА:
- НЕ ИСПОЛЬЗУЙ LaTeX (никаких $ и \\frac).
- Пиши формулы ПОНЯТНО: используй символы √, ², ³, ±, ≠, ≈, ∞, ∫, π.
- Дроби пиши через скобки и слэш, чтобы было как в тетради: (x+1)/(y-2).
- Никакого "программистского" языка (никаких sqrt() или pow()).

СТИЛЬ:
- Кратко, четко, без воды.
- Если задача простая — реши быстро. Если сложная — пошагово.
- В конце ОБЯЗАТЕЛЬНО блок 'ДЛЯ ЗАПОЛНЕНИЯ' с голыми цифрами.
"""

FOLLOWUP_SYSTEM = """
Ты — Трегубчик. Студент задает вопрос по твоему решению.
Отвечай коротко. Если благодарят — не рассыпайся в любезностях.
Твой максимум: "Разобрались. На экзамене будет сложнее" или "Принято".
"""

# ───────────────────────────── фразы ────────────────────────────

PHOTO_ACCEPT_PHRASES = [
    "Принял. Глянем, что ты там наснимал...",
    "Фото получил. Считаю. Надеюсь, там не почерк врача...",
    "Так, посмотрим на этот шедевр. Минутку.",
    "Получил. Разбираю каракули...",
    "Принято. Сейчас решим, если условия понятны.",
    "Вижу задачу. Ищу решение, жди."
]

THINKING_PHRASES = [
    "Смотрю...", "Считаю...", "Разбираю...", "Секунду...", "Думаю...",
    "Так-так...", "Где-то я это уже видел...", "Интересно."
]

REPLY_TO_THANKS = [
    "Разобрались. Иди учи дальше.",
    "Принято. На экзамене я рядом не буду.",
    "Хорошо. Главное — сам пойми, а не просто перепиши.",
    "Запомни это решение. Второй раз объяснять не стану.",
    "Свободен. Пока что."
]

# ───────────────────────────── utils ────────────────────────────

def clean_response(text: str) -> str:
    # Убираем жирный шрифт и прочий мусор, который может ломать чтение
    text = re.sub(r'\*\*|\*|__|#', '', text)
    return text.strip()

def compress_image(image_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Сжимаем до 1200px — золотая середина для GPT-4o-mini
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
            temperature=0.3
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
    await message.answer("Я Трегубчик. Кидай фото — разберем твои мучения.")

@dp.message(F.text == "/clear")
async def clear(message: Message):
    user_context.pop(message.from_user.id, None)
    await message.answer("Контекст очищен. Начинаем с чистого листа.")

@dp.message(F.photo)
async def solve_photo(message: Message):
    user_id = message.from_user.id
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
                    {"type": "text", "text": "Реши задачу:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }
        ]

        response = await send_typing_while(message, generate_with_retry(messages))
        if not response:
            await message.answer("OpenAI тупит. Попробуй через минуту.")
            return

        answer = response.choices[0].message.content
        user_context[user_id] = answer # Сохраняем контекст
        await send_long_message(message, answer)

    except Exception as e:
        print(f"Error: {e}")
        await message.answer("Не смог прочитать. Пришли фото получше.")

@dp.message(F.text)
async def handle_text(message: Message):
    user_id = message.from_user.id
    text = message.text.lower()

    # Если студент просто вежливый
    if any(word in text for word in ["спасибо", "спс", "понял", "благодарю"]):
        await message.answer(random.choice(REPLY_TO_THANKS))
        return

    # Если есть контекст — отвечаем на вопрос
    if user_id in user_context:
        await message.answer(random.choice(THINKING_PHRASES))
        
        messages = [
            {"role": "system", "content": FOLLOWUP_SYSTEM + f"\nКонтекст задачи:\n{user_context[user_id]}"},
            {"role": "user", "content": message.text}
        ]

        response = await generate_with_retry(messages)
        if response:
            answer = response.choices[0].message.content
            # Обновляем контекст, чтобы помнить и уточнения
            user_context[user_id] += f"\nВопрос: {message.text}\nОтвет: {answer}"
            await send_long_message(message, answer)
        else:
            await message.answer("Не могу ответить. Спроси по-другому.")
    else:
        await message.answer("Сначала задачу скинь (фото), потом будем разговаривать.")

# ───────────────────────────── server ─────────────────────────────

async def main():
    app = web.Application()
    app.router.add_get("/", lambda r: web.Response(text="Bot Active"))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", int(os.getenv("PORT", 8080)))
    asyncio.create_task(site.start())
    
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
