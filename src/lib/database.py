"""Database module for SQLite operations."""

import aiosqlite
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

DATABASE_PATH = Path("data/database.db")
LOGS_DATABASE_PATH = Path("logs/request_logs.db")


async def init_db() -> None:
    """Initialize the database and create tables if they don't exist."""
    logger.debug(f"Initializing database at: {DATABASE_PATH}")
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    async with aiosqlite.connect(DATABASE_PATH) as db:
        logger.debug("Executing CREATE TABLE IF NOT EXISTS for 'items' table")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY,
                context TEXT NOT NULL
            )
        """)
        await db.commit()
        logger.info("Database initialized successfully")
    
    # Initialize logs database
    await init_logs_db()


async def init_logs_db() -> None:
    """Initialize the logs database for request/response logging."""
    logger.debug(f"Initializing logs database at: {LOGS_DATABASE_PATH}")
    LOGS_DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    async with aiosqlite.connect(LOGS_DATABASE_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS gigachat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                chat_id INTEGER,
                request_text TEXT NOT NULL,
                response_text TEXT,
                tokens_used INTEGER,
                is_json_response INTEGER,
                task_id INTEGER,
                task_result TEXT,
                error TEXT
            )
        """)
        await db.commit()
        logger.info("Logs database initialized successfully")


async def log_gigachat_request(
    chat_id: int,
    request_text: str,
    response_text: Optional[str] = None,
    tokens_used: Optional[int] = None,
    is_json_response: Optional[bool] = None,
    task_id: Optional[int] = None,
    task_result: Optional[str] = None,
    error: Optional[str] = None
) -> Optional[int]:
    """Log a GigaChat request and response to the logs database.
    
    Args:
        chat_id: The chat ID associated with the request.
        request_text: The full request text sent to GigaChat.
        response_text: The response received from GigaChat.
        tokens_used: Number of tokens used in the request.
        is_json_response: Whether the response was JSON.
        task_id: The task ID if response was a task.
        task_result: The result of task processing.
        error: Any error that occurred.
        
    Returns:
        The ID of the inserted log entry.
    """
    timestamp = datetime.now().isoformat()
    
    logger.debug(f"Logging GigaChat request for chat_id={chat_id}")
    logger.debug(f"Request length: {len(request_text)} chars, Response length: {len(response_text) if response_text else 0} chars")
    
    async with aiosqlite.connect(LOGS_DATABASE_PATH) as db:
        cursor = await db.execute(
            """INSERT INTO gigachat_logs
               (timestamp, chat_id, request_text, response_text, tokens_used,
                is_json_response, task_id, task_result, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                timestamp,
                chat_id,
                request_text,
                response_text,
                tokens_used,
                1 if is_json_response else 0 if is_json_response is not None else None,
                task_id,
                task_result,
                error
            )
        )
        await db.commit()
        log_id = cursor.lastrowid
        logger.info(f"Logged GigaChat request with id={log_id}")
        return log_id


async def get_recent_logs(limit: int = 50) -> list[dict]:
    """Retrieve recent log entries.
    
    Args:
        limit: Maximum number of entries to retrieve.
        
    Returns:
        List of log entries as dictionaries.
    """
    async with aiosqlite.connect(LOGS_DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM gigachat_logs ORDER BY id DESC LIMIT ?",
            (limit,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


async def insert_or_update_context(chat_id: int, context: str) -> None:
    """Insert or replace a row in the table.
    
    Args:
        chat_id: The ID of the row to insert/update.
        context: The context value to store.
    """
    logger.debug(f"INSERT OR REPLACE for chat_id={chat_id}, context length={len(context)} chars")
    logger.debug(f"Context preview: {context[:200]}{'...' if len(context) > 200 else ''}")
    
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO items (id, context) VALUES (?, ?)",
            (chat_id, context)
        )
        await db.commit()
        logger.info(f"Context saved for chat_id={chat_id}")


async def get_context_by_id(chat_id: int) -> str | None:
    """Retrieve context by ID from the items table.
    
    Args:
        chat_id: The ID of the row to retrieve.
        
    Returns:
        The context value if found, None otherwise.
    """
    logger.debug(f"SELECT context for chat_id={chat_id}")
    
    async with aiosqlite.connect(DATABASE_PATH) as db:
        async with db.execute(
            "SELECT context FROM items WHERE id = ?",
            (chat_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                logger.debug(f"Found context for chat_id={chat_id}, length={len(row[0])} chars")
                return row[0]
            else:
                logger.debug(f"No context found for chat_id={chat_id}")
                return None