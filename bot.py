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

if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN не найден в Environment Variables")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY не найден в Environment Variables")


# ───────────────────────────── init ─────────────────────────────

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

MODEL = "gpt-4o-mini"

# ───────────────────────────── persistent context ───────────────

CONTEXT_FILE = "user_context.json"


def load_context() -> dict:
    try:
        with open(CONTEXT_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
            return {int(k): v for k, v in raw.items()}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_context(ctx: dict):
    try:
        with open(CONTEXT_FILE, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in ctx.items()}, f, ensure_ascii=False)
    except Exception as e:
        print(f"Ошибка сохранения контекста: {e}")


user_context: dict = load_context()

# rate limiting: user_id -> список временных меток запросов
user_requests: dict[int, list[float]] = defaultdict(list)

RATE_LIMIT = 5        # максимум запросов...
RATE_WINDOW = 60.0    # ...за это количество секунд


# ───────────────────────────── prompts ──────────────────────────

SYSTEM_PROMPT = """
Ты — учебный Telegram-ассистент по имени Трегубчик.

Характер:
Строгий преподаватель с живым характером. Точный, без воды. Иногда — неожиданно смешной,
но юмор у тебя специфический: мрачноватый, абсурдный, иногда на грани. Не пошлый, не детский.
Шутишь редко, но метко — и именно тогда, когда студент этого не ждёт.
Не оскорбляешь, не хамишь, но можешь поддеть. Мат не используешь.

Примеры твоего юмора (не копируй, уловил дух):
- "Правильно. Хотя я видел, как люди неправильно решали это 40 минут подряд. Ты справился быстрее."
- "Ошибка в знаке. Классика. Из-за таких ошибок ракеты летят не туда."
- "Верно. Можешь идти праздновать. Или решить следующую задачу — на твоё усмотрение."
- "Это называется расходящийся ряд. Он уходит в бесконечность, как и некоторые студенты на пересдаче."

Важно: шутки — редко. В большинстве ответов — только точное решение и объяснение.
Юмор появляется сам, когда уместен, а не лепится к каждому ответу.

Правила форматирования (строго):
- НЕ используй LaTeX, символы $, \\frac, \\times, \\cdot, ^{} и подобные.
- НЕ используй Markdown: никаких **, *, ##, `, _подчёркиваний_.
- Формулы пиши обычным текстом: "a^2 + b^2 = c^2", "x = (-b + sqrt(D)) / 2a".
- Дроби пиши через слэш: "1/2", "3/4".
- Степени через ^: "x^2", "2^10".
- Текст как обычное сообщение — никакой разметки.

Правила ответа:
1. Кратко распознай условие задачи.
2. Реши задачу.
3. Простая задача — короткий ответ. Сложная — пошагово.
4. Если есть поля для ввода — в конце отдельно перечисли значения.
5. Если фото плохо видно — скажи об этом и попроси прислать лучше.

Формат ответа:

УСЛОВИЕ:
...

РЕШЕНИЕ:
...

ОТВЕТ:
...

ДЛЯ ЗАПОЛНЕНИЯ (если нужно):
...
"""

FOLLOWUP_SYSTEM = """
Ты — учебный ассистент Трегубчик. Строгий, точный, с живым характером.

Отвечай на вопрос пользователя по предыдущему решению.
Если не понял шаг — объясни проще. Если нашёл ошибку — проверь и исправь честно.

Про стиль:
- Никогда не заканчивай фразами "Если есть вопросы — спрашивай", "Не стесняйся", "Удачи!".
  Это не твой стиль.
- Если пользователь пишет что понял, благодарит, говорит "ок" — отвечай 1-2 фразами,
  каждый раз по-разному. Можно с иронией, можно с неожиданной мрачной шуткой,
  можно просто сухо. Главное — не одно и то же каждый раз.
  Примеры духа (не копируй дословно):
  "Разобрались. На экзамене я рядом не буду.",
  "Хорошо. Запомни — не для меня, для себя.",
  "Принято. Ракеты теперь будут лететь туда, куда надо.",
  "Отлично. Ещё 40 таких задач — и можно считать себя человеком."
- Если пользователь пишет что-то нейтральное — не требуй фото, отвечай по контексту.
- Юмор — редко, но метко. Мрачноватый, абсурдный, неожиданный. Не пошлый.

Правила форматирования (строго):
- НЕ используй LaTeX, символы $, \\frac, \\times и подобные.
- НЕ используй Markdown: никаких **, *, ##, `, _.
- Формулы обычным текстом: "x = (-b + sqrt(D)) / 2a", дроби через слэш.
- Текст как обычное сообщение.
"""


# ───────────────────────────── helpers ──────────────────────────

def clean_response(text: str) -> str:
    """Убирает остатки LaTeX и Markdown из ответа модели."""
    # LaTeX-окружения
    text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
    text = re.sub(r'\$.*?\$', '', text)
    text = re.sub(r'\\[\(\[].*?\\[\)\]]', '', text, flags=re.DOTALL)

    # LaTeX-команды
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', text)
    text = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', text)
    text = re.sub(r'\\cdot', '*', text)
    text = re.sub(r'\\times', '*', text)
    text = re.sub(r'\\pm', '±', text)
    text = re.sub(r'\\leq', '<=', text)
    text = re.sub(r'\\geq', '>=', text)
    text = re.sub(r'\\neq', '!=', text)
    text = re.sub(r'\\approx', '≈', text)
    text = re.sub(r'\\infty', '∞', text)
    text = re.sub(r'\\\w+\{([^}]*)\}', r'\1', text)  # \cmd{...} -> содержимое
    text = re.sub(r'\\\w+', '', text)                  # одиночные \cmd -> убрать

    # Markdown
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)
    text = re.sub(r'`{1,3}([^`]+)`{1,3}', r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # Лишние пустые строки
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()



def compress_image(image_bytes: bytes, max_size: int = 800, quality: int = 70) -> bytes:
    """Сжимает фото до max_size px по длинной стороне и качества quality."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        compressed = buf.getvalue()
        print(f"Фото: {len(image_bytes)//1024}KB -> {len(compressed)//1024}KB")
        return compressed
    except Exception as e:
        print(f"Ошибка сжатия: {e}")
        return image_bytes  # отдаём оригинал если что-то пошло не так

def is_rate_limited(user_id: int) -> bool:
    """Возвращает True, если пользователь превысил лимит запросов."""
    now = time.time()
    user_requests[user_id] = [
        t for t in user_requests[user_id] if now - t < RATE_WINDOW
    ]
    if len(user_requests[user_id]) >= RATE_LIMIT:
        return True
    user_requests[user_id].append(now)
    return False


async def send_typing_while(message: Message, coro):
    """Шлёт ChatAction параллельно с ожиданием корутины."""
    async def keep_typing():
        while True:
            await bot.send_chat_action(message.chat.id, "typing")
            await asyncio.sleep(4)

    typing_task = asyncio.create_task(keep_typing())
    try:
        result = await coro
    finally:
        typing_task.cancel()
    return result


async def generate_with_retry(messages: list, retries: int = 3):
    for i in range(retries):
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=2000,
            )
            return response
        except Exception as e:
            print(f"Ошибка OpenAI (попытка {i+1}): {e}")
            if i < retries - 1:
                await asyncio.sleep(2)
            else:
                return None


async def send_long_message(message: Message, text: str):
    if not text:
        await message.answer("Не получилось получить ответ.")
        return
    text = clean_response(text)
    for i in range(0, len(text), 4000):
        await message.answer(text[i:i + 4000])



# ───────────────────────────── phrases ─────────────────────────

THINKING_PHRASES = [
    "Смотрю...",
    "Считаю...",
    "Разбираю...",
    "Секунду...",
    "Думаю...",
    "Сейчас.",
    "Принял.",
    "Глянем.",
]

PHOTO_ACCEPT_PHRASES = [
    "Принял, смотрю...",
    "Фото получил. Считаю...",
    "Разбираю задачу...",
    "Сейчас посмотрим...",
    "Получил. Секунду...",
    "Глянем что тут...",
]

# ───────────────────────────── web server ───────────────────────

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


# ───────────────────────────── handlers ─────────────────────────

@dp.message(F.text == "/start")
async def start(message: Message):
    name = message.from_user.first_name or "студент"
    await message.answer(
        f"Здравствуй, {name}.\n\n"
        "Я Трегубчик — учебный ассистент.\n"
        "Присылай фото задачи, разберём.\n\n"
        "Если что-то непонятно в решении — просто напиши вопрос, "
        "я помню последнюю задачу.\n\n"
        "Команды:\n"
        "/help — что умею\n"
        "/clear — забыть последнюю задачу\n"
        "/about — о боте"
    )


@dp.message(F.text == "/help")
async def help_command(message: Message):
    await message.answer(
        "Что умею:\n\n"
        "1. Решаю задачи по фото — математика, физика, химия и другое.\n"
        "2. Объясняю каждый шаг понятным языком.\n"
        "3. Помню последнюю задачу: можешь спросить «почему так?» или «объясни шаг 2».\n"
        "4. Если нужно заполнить поля ответа — дам отдельный список значений.\n\n"
        "Советы:\n"
        "- Фото должно быть чётким, условие полностью в кадре.\n"
        "- Если ответ длинный — жди, не отправляй фото повторно.\n\n"
        "Команды:\n"
        "/start — начало\n"
        "/help — эта справка\n"
        "/clear — очистить контекст\n"
        "/about — о боте"
    )


@dp.message(F.text == "/about")
async def about_command(message: Message):
    await message.answer(
        "Трегубчик — учебный ассистент на основе GPT-4o mini.\n\n"
        "Создан, чтобы помогать разбирать задачи, а не решать их вместо тебя. "
        "Объясняю шаги, отвечаю на уточняющие вопросы, держу ответы чёткими.\n\n"
        "Лимит: не более 5 запросов в минуту."
    )


@dp.message(F.text == "/clear")
async def clear_context(message: Message):
    user_context.pop(message.from_user.id, None)
    save_context(user_context)
    await message.answer("Контекст сброшен. Присылай новую задачу.")


@dp.message(F.photo)
async def solve_photo(message: Message):
    user_id = message.from_user.id

    if is_rate_limited(user_id):
        await message.answer(
            "Полегче. Не более 5 запросов в минуту.\n"
            "Подожди немного и попробуй снова."
        )
        return

    await message.answer(random.choice(PHOTO_ACCEPT_PHRASES))

    try:
        photo = message.photo[-1]
        file = await bot.get_file(photo.file_id)
        downloaded = await bot.download_file(file.file_path)
        image_bytes = downloaded.read()
        image_bytes = compress_image(image_bytes)
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    }
                ],
            },
        ]

        response = await send_typing_while(message, generate_with_retry(messages))

        if not response:
            await message.answer(
                "OpenAI не отвечает. Подожди пару секунд и попробуй снова."
            )
            return

        answer = response.choices[0].message.content or "Не получилось распознать задачу."
        user_context[user_id] = answer
        save_context(user_context)

        await send_long_message(message, answer)

    except Exception as e:
        error_text = str(e)
        print("ERROR:", error_text)

        if "invalid_api_key" in error_text or "Incorrect API key" in error_text:
            await message.answer("Проблема с OpenAI API key. Нужно обновить ключ в Render.")
        elif "insufficient_quota" in error_text or "quota" in error_text:
            await message.answer("Лимит OpenAI исчерпан. Проверь баланс на platform.openai.com.")
        elif "rate_limit" in error_text or "429" in error_text:
            await message.answer("Слишком много запросов к OpenAI. Подожди немного.")
        elif "content_policy" in error_text:
            await message.answer("OpenAI отклонил запрос. Попробуй другое фото.")
        elif "image" in error_text or "mime" in error_text:
            await message.answer("Не смог прочитать изображение. Пришли фото ещё раз.")
        else:
            await message.answer("Что-то пошло не так. Попробуй снова или пришли фото получше.")


@dp.message(F.text)
async def ask_about_solution(message: Message):
    user_id = message.from_user.id

    if user_id not in user_context:
        await message.answer(
            "Сначала пришли фото задачи — потом задавай вопросы по решению."
        )
        return

    if is_rate_limited(user_id):
        await message.answer("Не более 5 запросов в минуту. Подожди немного.")
        return

    await message.answer(random.choice(THINKING_PHRASES))

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    FOLLOWUP_SYSTEM
                    + f"\n\nПредыдущее решение:\n{user_context[user_id]}"
                ),
            },
            {"role": "user", "content": message.text},
        ]

        response = await send_typing_while(message, generate_with_retry(messages))

        if not response:
            await message.answer("OpenAI не отвечает. Попробуй чуть позже.")
            return

        answer = response.choices[0].message.content or "Не получилось ответить на вопрос."
        user_context[user_id] += f"\n\nВопрос: {message.text}\nОтвет: {answer}"
        save_context(user_context)

        await send_long_message(message, answer)

    except Exception as e:
        print("ERROR:", str(e))
        await message.answer("Ошибка при ответе. Попробуй переформулировать вопрос.")


# ───────────────────────────── main ─────────────────────────────

async def main():
    print("Бот запущен")
    await start_web_server()
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
