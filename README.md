# Gigachat Agent API

HTTP API для интеллектуального агента GigaChat, решающего NP-полные задачи с использованием FastAPI и QUBO-солверов (npqtools).

## Структура проекта

```
├── configs/
│   └── config.yaml                # Конфигурация приложения
├── data/
│   └── database.db                # SQLite база данных (создаётся автоматически)
├── src/
│   ├── __init__.py                # Инициализация пакета
│   ├── __main__.py                # Точка входа (python -m src)
│   ├── main.py                    # FastAPI приложение и эндпоинты
│   ├── components/
│   │   ├── __init__.py
│   │   ├── gigachat.py            # Клиент GigaChat и двухфазные функции
│   │   └── models.py             # Pydantic модели запросов/ответов
│   └── lib/
│       ├── __init__.py
│       ├── compute.py             # QUBO-солверы (npqtools обёртки)
│       ├── database.py            # Модуль работы с БД (aiosqlite)
│       ├── prompts.py             # Промпты для GigaChat (классификация, извлечение, диалог)
│       └── task_processing.py     # Парсинг и решение задач
├── static/
│   └── index.html                 # Веб-интерфейс (SPA с авторизацией и чатами)
├── russian_trusted_root_ca_pem.crt # Сертификат Минцифры для GigaChat
├── requirements.txt               # Зависимости Python
└── README.md
```

## Конфигурация

Перед запуском необходимо настроить:

1. **Сертификат Минцифры**: Поместите файл `russian_trusted_root_ca_pem.crt` в корень проекта

2. **Конфигурация в `configs/config.yaml`**:
   ```yaml
   database:
     path: data/database.db

   server:
     host: "0.0.0.0"
     port: 8000
     reload: true

   gigachat:
     certificate_path: "russian_trusted_root_ca_pem.crt"
     api_key: "INSERT_YOUR_KEY"
     model: "GigaChat-2-Max"
     max_tokens: 80000
     max_loop_cycles: 6
   ```

   | Параметр | Описание |
   |---|---|
   | `database.path` | Путь к файлу SQLite базы данных |
   | `server.host` | Хост для запуска сервера |
   | `server.port` | Порт для запуска сервера |
   | `server.reload` | Автоперезагрузка при изменении кода |
   | `gigachat.certificate_path` | Путь к сертификату Минцифры |
   | `gigachat.api_key` | API-ключ GigaChat |
   | `gigachat.model` | Модель GigaChat |
   | `gigachat.max_tokens` | Лимит токенов для контекста |
   | `gigachat.max_loop_cycles` | Максимум итераций обработки запроса |

## Установка

```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск через модуль (рекомендуется)
python -m src

# Или напрямую через uvicorn
uvicorn src.main:app --reload
```

Сервер запустится на `http://localhost:8000`. Веб-интерфейс доступен по корневому URL.

## Архитектура обработки запросов

Агент использует **двухфазный** подход для обработки сообщений:

1. **Фаза 1 — Классификация**: GigaChat определяет тип запроса и возвращает числовой код:
   - `0` — обычный разговор (генерируется текстовый ответ)
   - `1` — задача коммивояжёра (TSP)
   - `2` — задача о максимальной клике
   - `3` — задача о рюкзаке (Knapsack)
   - `4` — арифметическое вычисление
   - `-1 N` — недостаточно данных для задачи типа N (запрос уточнения)

2. **Фаза 2 — Извлечение данных**: Если обнаружена задача (коды 1–4), GigaChat извлекает входные данные в простом текстовом формате, после чего задача решается соответствующим QUBO-солвером из `npqtools`.

Результат решения оформляется GigaChat в человекочитаемый ответ.

## API Endpoints

### Аутентификация

#### POST /register

Регистрация нового пользователя.

**Request:**
```json
{
  "username": "user1",
  "password": "secret"
}
```

**Response (200 OK):**
```json
{
  "user_id": 1,
  "username": "user1"
}
```

**Response (409 Conflict):**
```json
{
  "status": "error",
  "detail": "Username already exists"
}
```

#### POST /login

Аутентификация пользователя.

**Request:**
```json
{
  "username": "user1",
  "password": "secret"
}
```

**Response (200 OK):**
```json
{
  "user_id": 1,
  "username": "user1"
}
```

**Response (401 Unauthorized):**
```json
{
  "status": "error",
  "detail": "Invalid username or password"
}
```

### Управление чатами

#### POST /chats

Создание нового чата.

**Request:**
```json
{
  "user_id": 1,
  "title": "Новый чат"
}
```

**Response (200 OK):**
```json
{
  "id": 1,
  "title": "Новый чат",
  "created_at": ""
}
```

#### GET /chats/{user_id}

Получение списка чатов пользователя.

**Response (200 OK):**
```json
{
  "chats": [
    {
      "id": 1,
      "title": "Новый чат",
      "created_at": "2025-01-01 12:00:00"
    }
  ]
}
```

#### DELETE /chats

Удаление чата.

**Request:**
```json
{
  "user_id": 1,
  "chat_id": 1
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "message": "Chat deleted"
}
```

#### GET /chats/{chat_id}/messages?user_id={user_id}

Получение сообщений чата.

**Response (200 OK):**
```json
{
  "messages": [
    {"role": "user", "content": "Привет!"},
    {"role": "assistant", "content": "Здравствуйте!"}
  ]
}
```

### Сообщения

#### POST /new_message

Обработка сообщения пользователя через двухфазный агент.

**Request:**
```json
{
  "chat_id": 1,
  "user_id": 1,
  "text": "Вычисли 2 + 2 * 3"
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "gigachat_response": "Результат вычисления: 2 + 2 * 3 = 8"
}
```

**Response (403 Forbidden):**
```json
{
  "detail": "Chat does not belong to this user"
}
```

**Response (500 Error):**
```json
{
  "status": "error",
  "detail": "Error description"
}
```

### Служебные

#### GET /health

Проверка состояния сервера.

**Response:**
```json
{
  "status": "healthy"
}
```

#### GET /

Отдаёт веб-интерфейс (`static/index.html`).

## Поддерживаемые задачи

| Задача | task_id | Солвер | Входные данные |
|---|---|---|---|
| Задача коммивояжёра (TSP) | 1 | `QUBOSalesman` | Матрица расстояний |
| Максимальная клика | 2 | `QUBOClique` | Матрица смежности |
| Задача о рюкзаке | 3 | `QUBOKnapsack` | Предметы (вес, ценность) + вместимость |
| Арифметика | 4 | `eval()` | Математическое выражение |

## Примеры использования

### cURL

```bash
# Регистрация
curl -X POST "http://localhost:8000/register" \
  -H "Content-Type: application/json" \
  -d '{"username": "user1", "password": "secret"}'

# Вход
curl -X POST "http://localhost:8000/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "user1", "password": "secret"}'

# Создание чата
curl -X POST "http://localhost:8000/chats" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "title": "Мой чат"}'

# Отправка сообщения
curl -X POST "http://localhost:8000/new_message" \
  -H "Content-Type: application/json" \
  -d '{"chat_id": 1, "user_id": 1, "text": "Вычисли 15 + 27"}'

# Получение чатов
curl "http://localhost:8000/chats/1"

# Получение сообщений чата
curl "http://localhost:8000/chats/1/messages?user_id=1"

# Проверка здоровья
curl "http://localhost:8000/health"
```

### Python

```python
import requests

BASE = "http://localhost:8000"

# Регистрация
auth = requests.post(f"{BASE}/register", json={"username": "user1", "password": "secret"})
user = auth.json()

# Создание чата
chat = requests.post(f"{BASE}/chats", json={"user_id": user["user_id"], "title": "Тест"})
chat_id = chat.json()["id"]

# Отправка сообщения
response = requests.post(
    f"{BASE}/new_message",
    json={"chat_id": chat_id, "user_id": user["user_id"], "text": "Вычисли 2 + 2 * 3"}
)
print(response.json())
```

## API Документация

После запуска сервера доступна автоматическая документация:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## База данных

SQLite база данных создаётся автоматически при первом запуске в `data/database.db`.

### Таблица `users`

| Поле | Тип | Описание |
|---|---|---|
| `id` | INTEGER PRIMARY KEY AUTOINCREMENT | Уникальный идентификатор |
| `username` | TEXT NOT NULL UNIQUE | Имя пользователя |
| `password_hash` | TEXT NOT NULL | SHA-256 хеш пароля |
| `created_at` | TEXT NOT NULL | Дата создания |

### Таблица `items` (чаты)

| Поле | Тип | Описание |
|---|---|---|
| `id` | INTEGER PRIMARY KEY AUTOINCREMENT | ID чата |
| `user_id` | INTEGER NOT NULL | ID владельца (FK → users) |
| `context` | TEXT NOT NULL | Контекст диалога для GigaChat |
| `title` | TEXT NOT NULL | Заголовок чата |
| `messages_json` | TEXT NOT NULL | JSON-массив сообщений |
| `created_at` | TEXT NOT NULL | Дата создания |

### Логирование запросов

Запросы к GigaChat логируются в отдельную базу `logs/request_logs.db` (таблица `gigachat_logs`).

## Логирование

Уровень логирования настраивается через переменную окружения `LOG_LEVEL` (по умолчанию `INFO`):

```bash
LOG_LEVEL=DEBUG python -m src
```
