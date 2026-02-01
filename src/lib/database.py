"""Database module for SQLite operations."""

import aiosqlite
from pathlib import Path


DATABASE_PATH = Path("data/database.db")


async def init_db() -> None:
    """Initialize the database and create tables if they don't exist."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY,
                context TEXT NOT NULL
            )
        """)
        await db.commit()


async def insert_or_update_context(chat_id: int, context: str) -> None:
    """Insert or replace a row in the table.
    
    Args:
        chat_id: The ID of the row to insert/update.
        context: The context value to store.
    """
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO items (id, context) VALUES (?, ?)",
            (chat_id, context)
        )
        await db.commit()


async def get_context_by_id(chat_id: int) -> str | None:
    """Retrieve context by ID from the items table.
    
    Args:
        chat_id: The ID of the row to retrieve.
        
    Returns:
        The context value if found, None otherwise.
    """
    async with aiosqlite.connect(DATABASE_PATH) as db:
        async with db.execute(
            "SELECT context FROM items WHERE id = ?",
            (chat_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None