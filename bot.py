import asyncio
import base64
import json
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
    raise ValueError("BOT_TOKEN или OPENAI_API_KEY не заданы в Environment Variables")

# ───────────────────────────── init ─────────────────────────────
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

MODEL = "gpt-4o-mini"
user_context = {}  # Основная память (в оперативе для скорости на Render)

# ───────────────────────────── prompts ──────────────────────────

SYSTEM_PROMPT = """
Ты — учебный Telegram-ассистент по имени Трегубчик.

ЛИЧНОСТЬ:
Строгий преподаватель вуза. Твой стиль — сухая ирония, точность и отсутствие пощады к лени. 
Ты не просто решаешь, ты учишь. Твой юмор специфический, мрачноватый (про ракеты, пересдачу, бесконечные очереди в деканат). 

ВАЖНЫЕ ПРАВИЛА:
1. РЕШАЙ ДО КОНЦА. Никогда не говори "воспользуйся калькулятором" или "досчитай сам". Ты обязан выдать финальный численный результат.
2. ФОРМАТИРОВАНИЕ: Никакого LaTeX ($, \\frac). Используй Юникод: √, ², ³, ±, ≠, ≈, ∞, ∫, π.
3. СТРУКТУРА:
   УСЛОВИЕ: кратко.
   РЕШЕНИЕ: пошагово, с расчетами.
   ОТВЕТ: финальный результат.
   ДЛЯ ЗАПОЛНЕНИЯ: только цифры для ввода в систему.

Шутишь редко, но когда студент тупит или задача классическая — поддевай его.
"""

FOLLOWUP_SYSTEM = """
Ты — Трегубчик. Студент спрашивает по решению. 
Если он просит "досчитать" или "объяснить понятнее" — делай это немедленно.
Стиль ворчливый. На "спасибо" отвечай как препод: "Разобрались. На экзамене я рядом не буду" или "Свободен. Пока что".
"""

# ───────────────────────────── фразы ────────────────────────────

PHOTO_ACCEPT_PHRASES = [
    "Принял, смотрю...", "Фото получил. Сейчас посмотрим, где ты там ошибся.",
    "Разбираю задачу... Надеюсь, там не почерк врача.", "Секунду. Глянем, что тут за ребус.",
    "Получил. Сейчас решим, если условия понятны."
]

THINKING_PHRASES = [
    "Так-так...", "Считаю...", "Разбираю...", "Секунду...", "Думаю...",
    "Интересно. Сейчас проверим.", "Где-то я это уже видел..."
]

# ───────────────────────────── helpers ──────────────────────────

def clean_response(text: str) -> str:
    """Удаляет мусорную разметку Markdown"""
    text = re.sub(r'\*\*|\*|__|#', '', text)
    return text.strip()

def compress_image(image_bytes: bytes) -> str:
    """Сжатие фото (сохраняем баланс качества и веса)"""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # Ограничиваем размер, но оставляем детализацию для OpenAI
    img.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

async def send_long_message(message: Message, text: str):
    """Разбивка длинных ответов"""
    text = clean_response(text)
    for i in range(0, len(text), 4000):
        await message.answer(text[i:i + 4000])

async def generate_response(messages, temp=0.3):
    """Основной движок запросов к OpenAI"""
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=2500,
            temperature=temp
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Ошибка OpenAI: {e}")
        return None

# ───────────────────────────── handlers ─────────────────────────

@dp.message(F.text == "/start")
async def start(message: Message):
    await message.answer("Я Трегубчик. Присылай фото задачи. Глупых вопросов не задавай, пощады не жди.")

@dp.message(F.text == "/clear")
async def clear_context(message: Message):
    user_context.pop(message.from_user.id, None)
    await message.answer("Контекст очищен. Забыл тебя и твои ошибки.")

@dp.message(F.photo)
async def solve_photo(message: Message):
    uid = message.from_user.id
    await message.answer(random.choice(PHOTO_ACCEPT_PHRASES))

    try:
        # Индикация печати
        await bot.send_chat_action(message.chat.id, "typing")
        
        photo = message.photo[-1]
        file = await bot.get_file(photo.file_id)
        downloaded = await bot.download_file(file.file_path)
        img_b64 = compress_image(downloaded.read())

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Реши задачу полностью, со всеми вычислениями:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }
        ]

        answer = await generate_response(messages)
        if answer:
            user_context[uid] = answer # Запоминаем контекст
            await send_long_message(message, answer)
        else:
            await message.answer("Сервер OpenAI перегружен. Попробуй еще раз через минуту.")

    except Exception as e:
        print(f"ERROR: {e}")
        await message.answer("Не удалось прочитать фото. Попробуй прислать более четкое изображение.")

@dp.message(F.text)
async def handle_text(message: Message):
    uid = message.from_user.id

    if uid in user_context:
        await message.answer(random.choice(THINKING_PHRASES))
        
        messages = [
            {"role": "system", "content": FOLLOWUP_SYSTEM + f"\n\nИстория задачи:\n{user_context[uid]}"},
            {"role": "user", "content": message.text}
        ]

        answer = await generate_response(messages, temp=0.5)
        if answer:
            # Обновляем историю, чтобы бот помнил цепочку вопросов
            user_context[uid] += f"\nСтудент: {message.text}\nТрегубчик: {answer}"
            await send_long_message(message, answer)
        else:
            await message.answer("Что-то пошло не так. Переформулируй вопрос.")
    else:
        await message.answer("Сначала пришли фото задачи, студент.")

# ───────────────────────────── web server ───────────────────────

async def handle_root(request):
    return web.Response(text="Tregubchik is alive")

async def main():
    # Запуск веб-сервера для Render (чтобы не засыпал)
    app = web.Application()
    app.router.add_get("/", handle_root)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", int(os.getenv("PORT", 10000)))
    asyncio.create_task(site.start())
    
    print("Бот запущен...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
