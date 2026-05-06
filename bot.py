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
    raise ValueError("Проверьте переменные BOT_TOKEN и OPENAI_API_KEY")

# ───────────────────────────── init ─────────────────────────────
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

MODEL = "gpt-4o-mini"
user_context = {}

# ───────────────────────────── prompts ──────────────────────────

SYSTEM_PROMPT = """
Ты — учебный Telegram-ассистент по имени Трегубчик.

Характер:
Строгий преподаватель с живым характером. Точный, без воды. Твой юмор специфический: мрачноватый, абсурдный. 
Ты не хамишь, но можешь тонко поддеть за невнимательность.

Примеры настроения:
- "Ошибка в знаке. Классика. Из-за таких ошибок ракеты летят не туда."
- "Верно. Можешь идти праздновать. Или решить следующую задачу — на твоё усмотрение."
- "Это называется расходящийся ряд. Он уходит в бесконечность, как и некоторые студенты на пересдаче."

ПРАВИЛА:
1. Реши задачу пошагово. НЕ используй LaTeX (никаких $, \\frac, \\times). Пиши простым текстом: x^2, sqrt(y), (a+b)/c.
2. В конце ОБЯЗАТЕЛЬНО заполни блок 'ДЛЯ ЗАПОЛНЕНИЯ', выписав туда финальные значения.
3. Если на фото ничего не понятно, так и скажи в своем стиле.
"""

FOLLOWUP_SYSTEM = """
Ты — Трегубчик. Отвечай на уточняющие вопросы по решению. 
Если студент благодарит, ответь коротко и сухо (например: "Разобрались. На экзамене я рядом не буду").
Никаких "Рад помочь" и "Удачи". Это не твой стиль.
"""

# ───────────────────────────── phrases ──────────────────────────

PHOTO_ACCEPT_PHRASES = [
    "Принял, смотрю...", "Фото получил. Считаю...", "Разбираю задачу...",
    "Сейчас посмотрим...", "Глянем, что тут у нас...", "Опять задачи? Ладно, жди."
]

THINKING_PHRASES = [
    "Так, посмотрим...", "Считаю...", "Разбираю...", "Секунду...", "Думаю..."
]

# ───────────────────────────── utils ────────────────────────────

def clean_response(text: str) -> str:
    """Удаляет Markdown-разметку, которая может ломаться в ТГ"""
    text = re.sub(r'\*\*|\*|__|#', '', text)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    return text.strip()

def compress_image(image_bytes: bytes) -> str:
    """Сжатие фото для экономии токенов (как советовал Клод)"""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Ограничиваем размер до 1200px по большой стороне
    max_size = 1200
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
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
    await message.answer("Я Трегубчик. Присылай фото — решу. Будешь задавать глупые вопросы — отвечу.")

@dp.message(F.text == "/clear")
async def clear(message: Message):
    user_context.pop(message.from_user.id, None)
    await message.answer("Контекст очищен. Как будто ничего и не было.")

@dp.message(F.photo)
async def solve_photo(message: Message):
    user_id = message.from_user.id
    await message.answer(random.choice(PHOTO_ACCEPT_PHRASES))

    try:
        photo = message.photo[-1]
        file_info = await bot.get_file(photo.file_id)
        photo_bytes = await bot.download_file(file_info.file_path)
        
        # Сжимаем и кодируем
        image_b64 = compress_image(photo_bytes.read())

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Реши это:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }
        ]

        response = await send_typing_while(message, generate_with_retry(messages))
        if not response:
            await message.answer("OpenAI молчит. Видимо, задача слишком сложная даже для него. Попробуй позже.")
            return

        answer = response.choices[0].message.content
        user_context[user_id] = answer
        await send_long_message(message, answer)

    except Exception as e:
        print(f"Error: {e}")
        await message.answer("Что-то пошло не так. Либо фото плохое, либо я сегодня не в духе.")

@dp.message(F.text)
async def ask_question(message: Message):
    user_id = message.from_user.id
    if user_id not in user_context:
        await message.answer("Сначала задачу скинь, потом спрашивай.")
        return

    await message.answer(random.choice(THINKING_PHRASES))
    
    messages = [
        {"role": "system", "content": FOLLOWUP_SYSTEM + f"\nКонтекст:\n{user_context[user_id]}"},
        {"role": "user", "content": message.text}
    ]

    response = await generate_with_retry(messages)
    if response:
        answer = response.choices[0].message.content
        user_context[user_id] += f"\nQ: {message.text}\nA: {answer}"
        await send_long_message(message, answer)
    else:
        await message.answer("Не смог сообразить ответ. Перефразируй.")

# ───────────────────────────── server ─────────────────────────────

async def main():
    # Простейший сервер для Render
    app = web.Application()
    app.router.add_get("/", lambda r: web.Response(text="Running"))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", int(os.getenv("PORT", 8080)))
    asyncio.create_task(site.start())
    
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
