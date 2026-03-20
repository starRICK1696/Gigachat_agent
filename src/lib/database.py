"""Database module for SQLite operations."""

import aiosqlite
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DATABASE_PATH = Path("data/database.db")


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