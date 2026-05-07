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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

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

MODEL = "gpt-5-mini"

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
- Текст как обычное сообщение — никакой разметки.

Обозначения — пиши понятно, как говорят вслух:
- Корень: "корень из 25" или "√25", НЕ "sqrt(25)"
- Дроби: "1/2", "три четвёртых", НЕ "frac{1}{2}"
- Степени: "x в квадрате", "x^2", НЕ "x^{2}"
- Умножение: "·" или "×" или просто "3 на 4", НЕ "*"
- Интеграл: "интеграл от 0 до 1", НЕ "\int_0^1"
- Предел: "предел при x стремящемся к 0", НЕ "\lim_{x->0}"
- Бесконечность: "бесконечность" или "∞"
- Принадлежность: "x принадлежит R" или "x ∈ R"
- Для всех: "для всех x" или "∀x"
- Сумма: "сумма от i=1 до n"
- Если символ помогает понять — используй его (√, ∞, ∈, ∀, ±, ≤, ≥, ≠, ≈)
- Если нет подходящего символа — пиши словами, понятно и по-человечески

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
- Если пользователь благодарит или говорит "понял/ок/спасиб" — отвечай 1-2 фразами,
  каждый раз по-разному. Преимущественно сухо, с иронией или коротким напутствием.
  Шутку сюда — только если она сама просится. Чаще без неё.
  Примеры духа (не копируй дословно):
  "Разобрались. На экзамене я рядом не буду.",
  "Хорошо. Запомни — не для меня, для себя.",
  "Принято.",
  "Отлично. Ещё 40 таких задач — и можно считать себя человеком."
- Юмор уместен когда: студент долго не понимал и наконец дошло, задача была с подвохом,
  ошибка была классической и банальной. Не уместен как реакция на простое "спасибо".
- Юмор — мрачноватый, абсурдный, неожиданный. Не пошлый. Редко.
- Если пользователь пишет что-то нейтральное — не требуй фото, отвечай по контексту.

Правила форматирования (строго):
- НЕ используй LaTeX, символы $, \\frac, \\times и подобные.
- НЕ используй Markdown: никаких **, *, ##, `, _.
- Текст как обычное сообщение.
- Обозначения как в SYSTEM_PROMPT: "корень из x" или √x, НЕ sqrt(x).
  Пиши понятно — как объясняют вслух на паре.
"""



GRAPH_SYSTEM_PROMPT = """
Ты анализируешь задачу и определяешь: нужно ли построить график для её решения или объяснения.

Если график нужен — верни ТОЛЬКО JSON без какого-либо текста до или после, строго такого формата:
{
  "need_graph": true,
  "functions": [
    {"expr": "x**2 - 3*x + 2", "label": "f(x) = x² - 3x + 2"},
    {"expr": "2*x - 1", "label": "g(x) = 2x - 1"}
  ],
  "x_min": -2,
  "x_max": 5,
  "title": "Пересечение параболы и прямой",
  "mark_zeros": true,
  "mark_intersections": true
}

Если график НЕ нужен — верни ТОЛЬКО:
{"need_graph": false}

Правила для expr:
- Используй только Python/numpy синтаксис
- Степень: x**2, x**3
- Корень: np.sqrt(x)
- Тригонометрия: np.sin(x), np.cos(x), np.tan(x)
- Логарифм: np.log(x) — натуральный, np.log10(x) — десятичный
- Экспонента: np.exp(x)
- Абсолютное значение: np.abs(x)
- Константа e: np.e
- Пи: np.pi
- Не используй math.*, только np.*

x_min и x_max — разумный диапазон для задачи. Обычно от -10 до 10.
mark_zeros — отметить нули функции на графике.
mark_intersections — отметить точки пересечения функций.
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


def build_graph(graph_data: dict) -> bytes:
    """Строит график по данным от GPT и возвращает PNG bytes."""
    fig, ax = plt.subplots(figsize=(9, 6), dpi=130)

    # Стиль
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='#a0a0b0', labelsize=9)
    ax.xaxis.label.set_color('#a0a0b0')
    ax.yaxis.label.set_color('#a0a0b0')
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a2a4a')

    ax.axhline(0, color='#3a3a5a', linewidth=1.2, zorder=1)
    ax.axvline(0, color='#3a3a5a', linewidth=1.2, zorder=1)
    ax.grid(True, color='#2a2a4a', linewidth=0.6, linestyle='--', alpha=0.7)

    x_min = graph_data.get("x_min", -10)
    x_max = graph_data.get("x_max", 10)
    x = np.linspace(x_min, x_max, 1200)

    colors = ['#7eb8f7', '#f7a07e', '#7ef7a0', '#f7e07e', '#d07ef7']
    functions = graph_data.get("functions", [])
    ys = []

    for i, fn in enumerate(functions):
        expr = fn.get("expr", "")
        label = fn.get("label", f"f{i+1}(x)")
        color = colors[i % len(colors)]
        try:
            y = eval(expr, {"x": x, "np": np, "__builtins__": {}})
            y = np.where(np.abs(y) > 1e6, np.nan, y)  # убираем выбросы
            ax.plot(x, y, color=color, linewidth=2.2, label=label, zorder=3)
            ys.append(y)

            # Нули функции
            if graph_data.get("mark_zeros") and len(functions) == 1:
                sign_changes = np.where(np.diff(np.sign(y)))[0]
                for idx in sign_changes:
                    if not np.isnan(y[idx]) and not np.isnan(y[idx+1]):
                        x_zero = x[idx] - y[idx] * (x[idx+1] - x[idx]) / (y[idx+1] - y[idx])
                        ax.plot(x_zero, 0, 'o', color='#ffffff', markersize=6,
                                zorder=5, markeredgecolor=color, markeredgewidth=1.5)
                        ax.annotate(f'  {x_zero:.2f}', (x_zero, 0),
                                    color='#c0c0d0', fontsize=8, va='bottom')
        except Exception as e:
            print(f"Ошибка построения {expr}: {e}")

    # Точки пересечения
    if graph_data.get("mark_intersections") and len(ys) >= 2:
        try:
            diff = ys[0] - ys[1]
            sign_changes = np.where(np.diff(np.sign(diff)))[0]
            for idx in sign_changes:
                if not np.isnan(diff[idx]) and not np.isnan(diff[idx+1]):
                    x_int = x[idx] - diff[idx] * (x[idx+1] - x[idx]) / (diff[idx+1] - diff[idx])
                    y_int = float(eval(functions[0]["expr"],
                                       {"x": x_int, "np": np, "__builtins__": {}}))
                    ax.plot(x_int, y_int, '*', color='#ffffff', markersize=10,
                            zorder=6, markeredgecolor='#f7a07e', markeredgewidth=1)
                    ax.annotate(f'  ({x_int:.2f}; {y_int:.2f})', (x_int, y_int),
                                color='#c0c0d0', fontsize=8)
        except Exception as e:
            print(f"Ошибка пересечений: {e}")

    title = graph_data.get("title", "График")
    ax.set_title(title, color='#d0d0e8', fontsize=12, pad=12)
    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("y", fontsize=10, rotation=0, labelpad=12)

    if functions:
        legend = ax.legend(facecolor='#1a1a2e', edgecolor='#3a3a5a',
                           labelcolor='#c0c0d0', fontsize=9)

    # Умные пределы по y
    all_y = [y for y in ys if y is not None]
    if all_y:
        combined = np.concatenate(all_y)
        finite = combined[np.isfinite(combined)]
        if len(finite):
            margin = (finite.max() - finite.min()) * 0.15 or 1
            ax.set_ylim(finite.min() - margin, finite.max() + margin)

    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_major_locator(ticker.AutoLocator())

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


async def check_and_build_graph(image_b64: str, solution_text: str, message: Message):
    """Спрашивает GPT нужен ли график, и если да — строит и отправляет."""
    try:
        check_messages = [
            {"role": "system", "content": GRAPH_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                    {
                        "type": "text",
                        "text": f"Решение задачи:\n{solution_text}"
                    }
                ],
            },
        ]

        response = await client.chat.completions.create(
            model=MODEL,
            messages=check_messages,
            max_tokens=400,
        )

        raw = response.choices[0].message.content or ""
        raw = raw.strip().strip("```json").strip("```").strip()
        graph_data = json.loads(raw)

        if not graph_data.get("need_graph"):
            return

        png_bytes = await asyncio.get_event_loop().run_in_executor(
            None, build_graph, graph_data
        )

        from aiogram.types import BufferedInputFile
        await message.answer_photo(
            BufferedInputFile(png_bytes, filename="graph.png"),
            caption="График к задаче."
        )

    except json.JSONDecodeError:
        pass  # GPT вернул не JSON — значит график не нужен
    except Exception as e:
        print(f"Ошибка графика: {e}")


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

        # Строим график если нужен
        await check_and_build_graph(image_b64, answer, message)

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
