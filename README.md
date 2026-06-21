# 🧠 RAG_assist — AI-помощник на ваших данных

> AI, который отвечает не «из головы», а строго по вашим документам, инструкциям и базе знаний.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-green)](https://openai.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector-blueviolet)](https://docs.trychroma.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🔍 Что это и зачем нужно

**RAG (Retrieval-Augmented Generation)** — технология, позволяющая AI отвечать на вопросы, используя **ваши данные**, а не только общую тренировочную базу.

### 💼 Решаемые бизнес-задачи:
| Задача | Как решает RAG_assist |
|--------|----------------------|
| 📞 Поддержка клиентов | Бот отвечает по вашей базе знаний, с ссылками на источники |
| 📚 Внутренний поиск | Сотрудники находят информацию в документах за секунды |
| 📄 Анализ договоров | AI выделяет риски, условия, сроки — на основе ваших шаблонов |
| 🗂 Работа с инструкциями | Быстрые ответы по техническим регламентам, политикам, FAQ |

### ✨ Преимущества:
- ✅ **Точность**: ответы с цитатами из ваших документов, минимум «галлюцинаций»
- ✅ **Приватность**: векторная база работает локально, данные не уходят в облако (кроме запросов к LLM по вашему выбору)
- ✅ **Гибкость**: подключаем любые источники — PDF, DOCX, TXT, Notion, Confluence
- ✅ **Масштабируемость**: от прототипа на 10 документах до продакшена на 10 000+

---

## 🛠 Технологии

- 🐍 Python 3.9+
- 🧠 OpenAI API / локальные LLM (опционально)
- 🗄️ ChromaDB — векторная база для семантического поиска
- 📦 LangChain / LlamaIndex — оркестрация RAG-цепочек
- 🔐 `.env` — безопасное хранение ключей и конфигураций

---

Консольное приложение с архитектурой **Retrieval-Augmented Generation (RAG)**: ответы формируются на основе документов из векторного хранилища и языковой модели.

Проект содержит две независимые реализации:

| Режим | Папка | LLM | Embeddings |
|-------|-------|-----|------------|
| **API Mode** | `assistant_api/` | OpenAI (`gpt-4o-mini`) | OpenAI `text-embedding-3-small` |
| **GigaChat Mode** | `assistant_giga/` | GigaChat (Сбер) | GigaChat Embeddings |

## Возможности

- Загрузка и индексация документов в **ChromaDB**
- Семантический поиск релевантных фрагментов
- Генерация ответов с учётом контекста
- Кеширование ответов в **SQLite** (снижает расход API и ускоряет повторные запросы)
- Поддержка **HTTP/HTTPS прокси** для OpenAI (через `proxy_config.py`)
- Оценка качества RAG через **RAGAS** (`assistant_api/evaluate_ragas.py`)

## Структура проекта

```
PEr08/
├── assistant_api/          # RAG на OpenAI API
│   ├── app.py              # Консольное приложение
│   ├── rag_pipeline.py     # Основной pipeline
│   ├── vector_store.py     # ChromaDB + embeddings
│   ├── cache.py            # SQLite-кеш
│   ├── evaluate_ragas.py   # Оценка метрик RAGAS
│   ├── rag_test_questions.txt
│   └── data/               # Документы для индексации
├── assistant_giga/         # RAG на GigaChat
│   ├── app.py
│   ├── rag_pipeline.py
│   ├── vector_store.py
│   ├── gigachat_client.py
│   ├── cache.py
│   └── data/               # Документы для индексации
├── proxy_config.py         # Настройка прокси для OpenAI
├── requirements.txt
├── .env.example            # Шаблон переменных окружения
├── .env                    # Ключи API (создать локально, не коммитить!)
└── README.md
```

## Публикация на GitHub

```powershell
git init
git add .
git status   # убедитесь, что .env и venv не в списке
git commit -m "Initial commit: RAG assistant with OpenAI and GigaChat"
git branch -M main
git remote add origin https://github.com/<username>/<repo>.git
git push -u origin main
```

> **Важно:** файл `.env` с ключами API в репозиторий не попадает (см. `.gitignore`). Используйте `.env.example` как шаблон.

## Требования

- **Python 3.11** (рекомендуется; см. `установка python 3-11.txt`)
- Ключ OpenAI API — для `assistant_api`
- Ключи GigaChat — для `assistant_giga`

## Установка

### 1. Python 3.11

```powershell
winget install Python.Python.3.11
```

### 2. Виртуальное окружение

```powershell
cd d:\WBCODE\PEr08
py -3.11 -m venv venv_py311
.\venv_py311\Scripts\activate
```

### 3. Зависимости

```powershell
pip install -r requirements.txt
```

## Настройка

Скопируйте шаблон и заполните своими ключами:

```powershell
copy .env.example .env
```

Отредактируйте `.env`:

```env
# OpenAI (assistant_api)
OPENAI_API_KEY=sk-...

# GigaChat (assistant_giga)
GIGACHAT_AUTH_KEY=...
GIGACHAT_RQUID=...

# Прокси для OpenAI (опционально)
HTTP_PROXY=...

```

> **Важно:** не публикуйте `.env` в репозиторий — в нём хранятся секретные ключи.

### Прокси

Для доступа к OpenAI из регионов с ограничениями укажите `HTTP_PROXY` или `HTTPS_PROXY` в `.env`. Прокси применяется автоматически к:

- запросам OpenAI (чат и embeddings);
- оценке RAGAS.

GigaChat работает напрямую, без прокси.

## Запуск

### OpenAI (API Mode)

```powershell
cd assistant_api
python app.py
```

### GigaChat

```powershell
cd assistant_giga
python app.py
```

### Команды в консоли

| Команда | Описание |
|---------|----------|
| `stats` | Статистика: коллекция, кеш, модель, прокси |
| `clear` | Очистить кеш ответов |
| `exit` / `quit` | Выход |

## Оценка качества (RAGAS)

Скрипт оценивает RAG-систему на тестовых вопросах по метрикам **Faithfulness** и **Context Precision**:

```powershell
cd assistant_api
python evaluate_ragas.py
```

Оценка использует OpenAI API и может занять 1–2 минуты.

## Как это работает

```
Вопрос пользователя
       │
       ▼
  Проверка кеша ──► (найден) ──► Ответ
       │
       ▼ (не найден)
  Поиск в ChromaDB (top-k документов)
       │
       ▼
  Формирование промпта с контекстом
       │
       ▼
  Запрос к LLM (OpenAI / GigaChat)
       │
       ▼
  Сохранение в кеш ──► Ответ
```

Документы из `data/docs.txt` разбиваются на чанки (~500 символов с перекрытием), векторизуются и сохраняются в ChromaDB при первом запуске.

## Зависимости

Основные библиотеки: `openai`, `chromadb`, `python-dotenv`, `httpx`, `ragas`, `langchain`, `datasets`.

Полный список — в [`requirements.txt`](requirements.txt).

## Устранение неполадок

| Проблема | Решение |
|----------|---------|
| `OPENAI_API_KEY не установлен` | Проверьте `.env` в корне проекта |
| Ошибки сети к OpenAI | Включите прокси в `.env`, убедитесь что прокси-сервер запущен |
| Ошибки GigaChat | Проверьте `GIGACHAT_AUTH_KEY` и `GIGACHAT_RQUID` |
| Несовместимость пакетов | Используйте Python 3.11 и виртуальное окружение |
