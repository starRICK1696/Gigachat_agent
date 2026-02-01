# Gigachat agent API

HTTP API для агента Gigachat с использованием FastAPI.

## Структура проекта

```
├── configs/
│   └── config.yaml              # Конфигурация приложения
├── data/
│   └── database.db              # SQLite база данных (создается автоматически)
├── src/
│   ├── __init__.py              # Инициализация пакета
│   ├── main.py                  # FastAPI приложение
│   ├── components/
│   │   ├── gigachat.py          # Клиент и функции GigaChat
│   │   └── models.py            # Pydantic модели
│   └── lib/
│       ├── database.py          # Модуль работы с базой данных
│       └── prompts.py           # Промпты для GigaChat
├── russian_trusted_root_ca.cer  # Сертификат Минцифры для Gigachat (нужно добавить)
├── requirements.txt             # Зависимости Python
└── README.md
```

## Конфигурация

Перед запуском необходимо настроить:

1. **Сертификат Минцифры**: Поместите файл `russian_trusted_root_ca.cer` в корень проекта

2. **Конфигурация в `configs/config.yaml`**:
   ```yaml
   gigachat:
     # Путь к сертификату (относительно корня проекта)
     certificate_path: "russian_trusted_root_ca.cer"
     # API ключ GigaChat - замените на ваш ключ
     api_key: "YOUR_GIGACHAT_API_KEY_HERE"
   ```

## Установка

```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск
uvicorn src.main:app --reload
```

Сервер запустится на `http://localhost:8000`

## API Endpoints

### POST /new_message

Вставляет или обновляет текст в базе данных по указанному ID.

**Request:**
```json
{
  "id": 1,
  "text": "Hello World"
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "gigachat_response": "Some text"
}
```

**Response (500 Error):**
```json
{
  "status": "error",
  "detail": "Error description"
}
```

### GET /health

Проверка состояния сервера.

**Response:**
```json
{
  "status": "healthy"
}
```

## Примеры использования

### cURL

```bash
# Вставка текста
curl -X POST "http://localhost:8000/new_message" \
  -H "Content-Type: application/json" \
  -d '{"id": 1, "text": "Привет, мир!"}'

# Проверка здоровья
curl "http://localhost:8000/health"
```

### Python

```python
import requests

response = requests.post(
    "http://localhost:8000/new_message",
    json={"id": 1, "text": "Привет, мир!"}
)
print(response.json())
```

## API Документация

После запуска сервера доступна автоматическая документация:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## База данных

SQLite база данных создается автоматически при первом запуске в папке `data/database.db`.

Схема таблицы `items`:
- `id` (INTEGER PRIMARY KEY) - уникальный идентификатор
- `context` (TEXT NOT NULL) - текстовое значение