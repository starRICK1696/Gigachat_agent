"""Database module for SQLite operations."""

import aiosqlite
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

DATABASE_PATH = Path("data/database.db")
LOGS_DATABASE_PATH = Path("logs/request_logs.db")


def hash_password(password: str) -> str:
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


async def init_db() -> None:
    """Initialize the database and create tables if they don't exist."""
    logger.debug(f"Initializing database at: {DATABASE_PATH}")
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    async with aiosqlite.connect(DATABASE_PATH) as db:
        logger.debug("Executing CREATE TABLE IF NOT EXISTS for 'items' table")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                context TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT 'Новый чат',
                messages_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
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
    """Update context for an existing chat.
    
    Args:
        chat_id: The ID of the chat to update.
        context: The context value to store.
    """
    logger.debug(f"UPDATE context for chat_id={chat_id}, context length={len(context)} chars")
    logger.debug(f"Context preview: {context[:200]}{'...' if len(context) > 200 else ''}")
    
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute(
            "UPDATE items SET context = ? WHERE id = ?",
            (context, chat_id)
        )
        await db.commit()
        logger.info(f"Context saved for chat_id={chat_id}")


async def update_chat_title(chat_id: int, title: str) -> None:
    """Update the title of a chat.
    
    Args:
        chat_id: The ID of the chat.
        title: The new title.
    """
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute(
            "UPDATE items SET title = ? WHERE id = ?",
            (title, chat_id)
        )
        await db.commit()
        logger.info(f"Title updated for chat_id={chat_id}")


async def get_messages_json(chat_id: int) -> str:
    """Get the messages JSON for a chat.
    
    Args:
        chat_id: The chat ID.
        
    Returns:
        JSON string of messages array.
    """
    async with aiosqlite.connect(DATABASE_PATH) as db:
        async with db.execute(
            "SELECT messages_json FROM items WHERE id = ?",
            (chat_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else "[]"


async def update_messages_json(chat_id: int, messages_json: str) -> None:
    """Update the messages JSON for a chat.
    
    Args:
        chat_id: The chat ID.
        messages_json: JSON string of messages array.
    """
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute(
            "UPDATE items SET messages_json = ? WHERE id = ?",
            (messages_json, chat_id)
        )
        await db.commit()
        logger.debug(f"Messages JSON updated for chat_id={chat_id}")


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


# ==================== User management ====================

async def create_user(username: str, password: str) -> Optional[int]:
    """Create a new user.
    
    Args:
        username: The username.
        password: The plain-text password (will be hashed).
        
    Returns:
        The new user's ID, or None if username already exists.
    """
    password_hash = hash_password(password)
    
    async with aiosqlite.connect(DATABASE_PATH) as db:
        try:
            cursor = await db.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, password_hash)
            )
            await db.commit()
            user_id = cursor.lastrowid
            logger.info(f"Created user '{username}' with id={user_id}")
            return user_id
        except aiosqlite.IntegrityError:
            logger.warning(f"User '{username}' already exists")
            return None


async def authenticate_user(username: str, password: str) -> Optional[int]:
    """Authenticate a user by username and password.
    
    Args:
        username: The username.
        password: The plain-text password.
        
    Returns:
        The user's ID if credentials are valid, None otherwise.
    """
    password_hash = hash_password(password)
    
    async with aiosqlite.connect(DATABASE_PATH) as db:
        async with db.execute(
            "SELECT id FROM users WHERE username = ? AND password_hash = ?",
            (username, password_hash)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                logger.info(f"User '{username}' authenticated successfully")
                return row[0]
            else:
                logger.warning(f"Authentication failed for user '{username}'")
                return None


# ==================== Chat management ====================

async def create_chat(user_id: int, title: str = "Новый чат") -> int:
    """Create a new chat for a user.
    
    Args:
        user_id: The user's ID.
        title: The chat title.
        
    Returns:
        The new chat's ID.
    """
    async with aiosqlite.connect(DATABASE_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO items (user_id, context, title) VALUES (?, ?, ?)",
            (user_id, "", title)
        )
        await db.commit()
        chat_id: int = cursor.lastrowid  # type: ignore[assignment]
        logger.info(f"Created chat id={chat_id} for user_id={user_id}")
        return chat_id


async def get_user_chats(user_id: int) -> list[dict]:
    """Get all chats for a user.
    
    Args:
        user_id: The user's ID.
        
    Returns:
        List of chat dicts with id, title, created_at.
    """
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, title, created_at FROM items WHERE user_id = ? ORDER BY id DESC",
            (user_id,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


async def delete_chat(chat_id: int, user_id: int) -> bool:
    """Delete a chat belonging to a user.
    
    Args:
        chat_id: The chat ID to delete.
        user_id: The user's ID (for ownership check).
        
    Returns:
        True if deleted, False if not found or not owned by user.
    """
    async with aiosqlite.connect(DATABASE_PATH) as db:
        cursor = await db.execute(
            "DELETE FROM items WHERE id = ? AND user_id = ?",
            (chat_id, user_id)
        )
        await db.commit()
        if cursor.rowcount > 0:
            logger.info(f"Deleted chat id={chat_id} for user_id={user_id}")
            return True
        else:
            logger.warning(f"Chat id={chat_id} not found for user_id={user_id}")
            return False


async def chat_belongs_to_user(chat_id: int, user_id: int) -> bool:
    """Check if a chat belongs to a user.
    
    Args:
        chat_id: The chat ID.
        user_id: The user's ID.
        
    Returns:
        True if the chat belongs to the user.
    """
    async with aiosqlite.connect(DATABASE_PATH) as db:
        async with db.execute(
            "SELECT 1 FROM items WHERE id = ? AND user_id = ?",
            (chat_id, user_id)
        ) as cursor:
            row = await cursor.fetchone()
            return row is not None