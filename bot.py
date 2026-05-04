import asyncio
import os

from aiohttp import web
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from google import genai
from google.genai import types


BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN не найден в Environment Variables")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY не найден в Environment Variables")


bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
client = genai.Client(api_key=GEMINI_API_KEY)

# Контекст последней задачи для каждого пользователя
user_context = {}


SYSTEM_PROMPT = """
Ты — учебный Telegram-ассистент по имени Трегубчик.

Образ:
- строгий преподаватель;
- объясняешь понятно и без воды;
- допускается лёгкая сухая ирония, но редко;
- не вставляй шутки в каждое решение;
- не используй мат;
- не оскорбляй пользователя.

Важно:
Стиль должен быть второстепенным. Главное — точное решение.
Если задача обычная — лучше без шутки.
Если уместно, можно коротко сказать что-то вроде:
"Не страшно, но руками всё-таки поработать придётся."
Но редко.

Правила:
1. Сначала кратко распознай условие.
2. Потом реши задачу.
3. Если задача простая — отвечай коротко.
4. Если задача сложная — дай пошаговое решение.
5. Не используй LaTeX, $, \\frac, \\times.
6. Формулы пиши обычным текстом.
7. Если есть поля для ввода — отдельно дай значения.
8. Если плохо видно фото — скажи об этом и попроси фото получше.

Формат:

УСЛОВИЕ:
...

РЕШЕНИЕ:
...

ОТВЕТ:
...

ДЛЯ ЗАПОЛНЕНИЯ:
...
"""


async def send_long_message(message: Message, text: str):
    if not text:
        await message.answer("Не получилось получить ответ.")
        return

    for i in range(0, len(text), 4000):
        await message.answer(text[i:i + 4000])


async def handle(request):
    return web.Response(text="Tregubchik bot is running")


async def start_web_server():
    app = web.Application()
    app.router.add_get("/", handle)

    port = int(os.getenv("PORT", 10000))

    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()


@dp.message(F.text == "/start")
async def start(message: Message):
    await message.answer(
        "Я Трегубчик.\n\n"
        "Отправь фото задачи — решу.\n"
        "После решения можешь спросить, если что-то непонятно.\n\n"
        "Команды:\n"
        "/help — помощь\n"
        "/clear — забыть последнюю задачу"
    )


@dp.message(F.text == "/help")
async def help_command(message: Message):
    await message.answer(
        "Что я умею:\n\n"
        "1. Решаю задачи по фото.\n"
        "2. Объясняю решение простым языком.\n"
        "3. Помню последнюю задачу, чтобы ты мог спросить: «почему так?».\n"
        "4. Если нужно заполнить поля — отдельно даю значения.\n\n"
        "Команды:\n"
        "/start — запуск\n"
        "/help — помощь\n"
        "/clear — очистить контекст последней задачи"
    )


@dp.message(F.text == "/clear")
async def clear_context(message: Message):
    user_context.pop(message.from_user.id, None)
    await message.answer("Контекст очищен. Можешь отправлять новую задачу.")


@dp.message(F.photo)
async def solve_photo(message: Message):
    await message.answer("Принял. Считаю...")

    try:
        photo = message.photo[-1]
        file = await bot.get_file(photo.file_id)
        downloaded = await bot.download_file(file.file_path)
        image_bytes = downloaded.read()

        print("Отправляю фото в Gemini...")

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                SYSTEM_PROMPT,
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg"
                )
            ],
        )

        answer = response.text or "Не получилось распознать задачу."

        user_context[message.from_user.id] = answer

        print("Ответ получен")
        await send_long_message(message, answer)

    except Exception as e:
        error_text = str(e)
        print("ERROR:", error_text)

        if "API key" in error_text or "expired" in error_text:
            await message.answer("Проблема с Gemini API key. Нужно обновить ключ в Render.")

        elif "location" in error_text:
            await message.answer("Gemini недоступен из региона сервера. Проверь Render-регион или API.")

        elif "quota" in error_text or "429" in error_text:
            await message.answer("Лимит Gemini закончился. Попробуй позже.")

        elif "image" in error_text or "mime" in error_text:
            await message.answer("Не получилось прочитать изображение. Попробуй отправить фото ещё раз.")

        else:
            await message.answer("Что-то пошло не так. Попробуй ещё раз или отправь фото получше.")


@dp.message(F.text)
async def ask_about_solution(message: Message):
    user_id = message.from_user.id

    if user_id not in user_context:
        await message.answer("Сначала отправь фото задачи, потом спрашивай по решению.")
        return

    await message.answer("Сейчас разберём.")

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                f"""
Ты — учебный ассистент Трегубчик.

Твоя задача — ответить на вопрос пользователя по предыдущему решению.

Стиль:
- понятно;
- без воды;
- без мата;
- без унижения;
- если пользователь не понял шаг — объясни проще;
- если пользователь нашёл ошибку — проверь и исправь.

Предыдущее решение:
{user_context[user_id]}

Вопрос пользователя:
{message.text}

Ответь по делу.
"""
            ],
        )

        answer = response.text or "Не получилось ответить на вопрос."

        user_context[user_id] += (
            f"\n\nВопрос пользователя: {message.text}\nОтвет: {answer}"
        )

        await send_long_message(message, answer)

    except Exception as e:
        print("ERROR:", str(e))
        await message.answer("Ошибка при ответе на вопрос. Попробуй сформулировать иначе.")


async def main():
    print("Бот запущен")
    await start_web_server()
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
