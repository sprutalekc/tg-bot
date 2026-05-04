import asyncio
import os

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from google import genai
from google.genai import types

BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

client = genai.Client(api_key=GEMINI_API_KEY)

# Здесь храним последнее решение каждого пользователя
user_context = {}


@dp.message(F.text == "/start")
async def start(message: Message):
    await message.answer(
        "Привет! Отправь фото задачи, я решу её. "
        "После решения можешь задавать вопросы по нему."
    )


@dp.message(F.photo)
async def solve_photo(message: Message):
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

    answer = response.text

    user_context[message.from_user.id] = answer

    await message.answer(answer[:4000])


@dp.message(F.text)
async def ask_about_solution(message: Message):
    user_id = message.from_user.id

    if user_id not in user_context:
        await message.answer("Сначала отправь фото задачи, а потом задавай вопросы.")
        return

    await message.answer("Объясняю...")

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            f"""
Пользователь задал вопрос по предыдущему решению.

Предыдущее решение:
{user_context[user_id]}

Вопрос пользователя:
{message.text}

Ответь простым языком. 
Если пользователь просит объяснить шаг — объясни подробнее.
Если пользователь спрашивает про ошибку — проверь решение.
Если нужно, исправь ответ.
"""
        ],
    )

    answer = response.text

    user_context[user_id] += f"\n\nВопрос пользователя: {message.text}\nОтвет: {answer}"

    await message.answer(answer[:4000])


async def main():
    print("Бот запущен")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
