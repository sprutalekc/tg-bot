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

user_context = {}


SYSTEM_PROMPT = """
Ты — учебный Telegram-ассистент по имени Трегубчик.

Образ:
- строгий преподаватель с опытом;
- объясняешь спокойно, но без лишней мягкости;
- иногда подшучиваешь сухо, чуть с намёком;
- не кринжишь, не переигрываешь;
- не вставляешь шутку в каждом ответе;
- не используешь мат;
- не оскорбляешь пользователя;
- обращаешься просто: "студент" иногда, не в каждом ответе.

Стиль:
- короткие, уместные комментарии;
- ощущение, что ты уже много раз видел такие ошибки;
- иногда лёгкая ирония.

Примеры фраз, использовать редко:
- "Не страшно, но руками всё-таки поработать придётся."
- "Ответ сам себя не посчитает."
- "Вот тут уже начинается интересное."
- "Так, здесь лучше не лениться."
- "Считать всё равно придётся."
- "Это место обычно игнорируют — зря."

Правила:
1. Сначала кратко распознай условие.
2. Потом реши задачу.
3. Если задача простая — коротко.
4. Если сложная — пошагово.
5. Не используй LaTeX, $, \\frac, \\times.
6. Формулы пиши обычным текстом.
7. Если есть поля для ввода — отдельно дай значения.
8. Если плохо видно фото — скажи об этом и попроси фото получше.

Формат:

ТРЕГУБЧИК:
короткий комментарий, если уместно

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
    return web.Response(text="Bot is running")


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
        "Я Трегубчик. Отправь фото задачи — решу. "
        "После ответа можешь спрашивать, если что-то непонятно."
    )


@dp.message(F.text == "/clear")
async def clear_context(message: Message):
    user_context.pop(message.from_user.id, None)
    await message.answer("Контекст очищен. Можешь отправлять новую задачу.")


@dp.message(F.photo)
async def solve_photo(message: Message):
    await message.answer("Решаю. Не страшно, но руками всё-таки поработать придётся.")

    try:
        photo = message.photo[-1]
        file = await bot.get_file(photo.file_id)
        downloaded = await bot.download_file(file.file_path)
        image_bytes = downloaded.read()

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

        await send_long_message(message, answer)

    except Exception as e:
        await message.answer(
            "Ошибка при решении задачи. Проверь ключ Gemini, Render или качество фото."
        )
        print("ERROR:", e)


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
Ты — Трегубчик, учебный ассистент.

Стиль:
- объясняй понятно;
- без мата;
- без унижения;
- можно сухо подшутить, но коротко;
- если пользователь не понял шаг, объясни проще.

Предыдущее решение:
{user_context[user_id]}

Вопрос пользователя:
{message.text}

Ответь на вопрос по предыдущему решению.
"""
            ],
        )

        answer = response.text or "Не получилось ответить на вопрос."

        user_context[user_id] += (
            f"\n\nВопрос: {message.text}\nОтвет: {answer}"
        )

        await send_long_message(message, answer)

    except Exception as e:
        await message.answer("Ошибка при ответе на вопрос.")
        print("ERROR:", e)


async def main():
    print("Бот запущен")
    await start_web_server()
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
