import asyncio
from google import genai
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from google.genai import types

BOT_TOKEN = "8707015405:AAHnh_3QRYtD66hW1Pjas2uooycoO3pSu1A"
GEMINI_API_KEY = "AIzaSyDnbHuABWDUBpen3O5QUs61bSFNX9K5a6w"

client = genai.Client(api_key=GEMINI_API_KEY)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


@dp.message(F.text == "/start")
async def start(message: Message):
    await message.answer("Отправь фото задачи")


@dp.message(F.photo)
async def solve(message: Message):
    await message.answer("Решаю...")

    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    downloaded = await bot.download_file(file.file_path)

    image_bytes = downloaded.read()

    response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        """
Реши задачу с изображения.

Правила ответа:
1. Сначала кратко распознай условие.
2. Затем реши пошагово.
3. Не используй LaTeX с символами $, \\times, \\frac.
4. Пиши обычным текстом, удобно для Telegram.
5. В конце обязательно дай блок:

ОТВЕТ:
Общее решение: ...
Базисное решение: ...

Если задача просит заполнить поля, выдели только значения для полей.
""",
        types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg"
        )
    ],
)

    await message.answer(response.text[:4000])


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
