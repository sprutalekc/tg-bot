import os
import asyncio
from google import genai
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from google.genai import types

BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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
Ты решаешь учебные задачи по фото.

Правила:
1. Сначала кратко напиши, что дано в задаче.
2. Потом реши задачу пошагово, но без лишней воды.
3. Пиши простым языком, как студенту.
4. Не используй LaTeX, символы $, \\frac, \\times.
5. Формулы пиши обычным текстом.
6. Если задача с вариантами ответа — выбери правильный вариант и объясни.
7. Если задача требует заполнить поля — в конце отдельно напиши значения для полей.
8. Если на фото плохо видно условие — честно напиши, что не можешь точно распознать, и попроси фото получше.

Формат ответа:

УСЛОВИЕ:
кратко распознанное условие

РЕШЕНИЕ:
пошаговое решение

ОТВЕТ:
короткий итоговый ответ

Если есть поля для ввода:
ДЛЯ ЗАПОЛНЕНИЯ:
значение 1 = ...
значение 2 = ...
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
