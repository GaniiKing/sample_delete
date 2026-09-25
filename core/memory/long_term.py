import sqlite3
from datetime import datetime

class LongTermMemory:
    def __init__(self, db_path="divya_memory.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY,
            type TEXT,
            content TEXT,
            created_at TEXT
        )
        """)
        self.conn.commit()

    def store(self, memory_type, content):
        self.conn.execute(
            "INSERT INTO memory (type, content, created_at) VALUES (?, ?, ?)",
            (memory_type, content, datetime.now().isoformat())
        )
        self.conn.commit()

    def retrieve_all(self):
        cursor = self.conn.execute("SELECT type, content FROM memory")
        return cursor.fetchall()
