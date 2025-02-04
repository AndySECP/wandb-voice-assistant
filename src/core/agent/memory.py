from typing import List, Dict, Any, Optional
from collections import deque
import sqlite3
from datetime import datetime
import logging
import os

logger = logging.getLogger(__name__)


## This script is not used anymore
class MemoryManager:
    def __init__(
        self, db_path: str = "memory.db", cache_size: int = 20, persist_after: int = 5
    ):
        self.db_path = db_path
        self.cache_size = cache_size
        self.persist_after = persist_after

        # In-memory storage using deque with specified cache size
        self.recent_messages = deque(maxlen=self.cache_size)
        self.messages_since_persist = 0

        # If database exists, drop it for fresh start
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

        self._init_db()

    def _init_db(self):
        """Initialize SQLite database with simplified schema"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()

            # Create simple conversations table
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """
            )

            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    async def add_message(self, role: str, content: str) -> None:
        """Add a new message to memory"""
        try:
            timestamp = datetime.now().isoformat()
            message = {"role": role, "content": content, "timestamp": timestamp}

            # Add to in-memory cache
            self.recent_messages.append(message)
            self.messages_since_persist += 1

            # Persist to database if threshold reached
            if self.messages_since_persist >= self.persist_after:
                await self._persist_to_db()
                self.messages_since_persist = 0

        except Exception as e:
            logger.error(f"Error adding message: {e}")
            raise

    async def _persist_to_db(self) -> None:
        """Persist recent messages to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()

            # Get messages that haven't been persisted
            messages_to_persist = list(self.recent_messages)[
                -self.messages_since_persist :
            ]

            # Insert messages
            c.executemany(
                "INSERT INTO conversations (role, content, timestamp) VALUES (?, ?, ?)",
                [
                    (msg["role"], msg["content"], msg["timestamp"])
                    for msg in messages_to_persist
                ],
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error persisting to database: {e}")
            raise

    def get_recent_context(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get recent conversation messages from cache"""
        try:
            # Use cache_size as default limit if none provided
            limit = limit or self.cache_size

            # Get messages from cache
            messages = list(self.recent_messages)[-limit:]

            # Return only role and content for conversation context
            return [
                {"role": msg["role"], "content": msg["content"]} for msg in messages
            ]
        except Exception as e:
            logger.error(f"Error getting recent context: {e}")
            return []

    def search_memory(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search through all memory (both cache and database)"""
        try:
            results = []

            # Search in-memory cache first
            for msg in self.recent_messages:
                if query.lower() in msg["content"].lower():
                    results.append(msg)

            # If we need more results, search database
            if len(results) < limit:
                conn = sqlite3.connect(self.db_path)
                c = conn.cursor()

                rows = c.execute(
                    """
                    SELECT role, content, timestamp 
                    FROM conversations 
                    WHERE content LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """,
                    (f"%{query}%", limit - len(results)),
                ).fetchall()

                conn.close()

                for row in rows:
                    results.append(
                        {"role": row[0], "content": row[1], "timestamp": row[2]}
                    )

            return results[:limit]
        except Exception as e:
            logger.error(f"Error searching memory: {e}")
            return []

    def clear_memory(self):
        """Clear both cache and database"""
        try:
            # Clear in-memory cache
            self.recent_messages.clear()
            self.messages_since_persist = 0

            # Clear database
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("DELETE FROM conversations")
            conn.commit()
            conn.close()

            logger.info("Memory cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing memory: {e}")
            raise
