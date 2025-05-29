import sqlite3
from datetime import datetime

class ModelRegistry:
    def __init__(self, db_path='model_registry.db'):
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    version TEXT,
                    path TEXT,
                    metrics TEXT,
                    created_at TEXT
                )
            ''')

    def register_model(self, name, version, path, metrics):
        with self.conn:
            self.conn.execute(
                'INSERT INTO models (name, version, path, metrics, created_at) VALUES (?, ?, ?, ?, ?)',
                (name, version, path, str(metrics), datetime.now().isoformat())
            )

    def get_model(self, name, version=None):
        cur = self.conn.cursor()
        if version:
            cur.execute('SELECT * FROM models WHERE name=? AND version=?', (name, version))
        else:
            cur.execute('SELECT * FROM models WHERE name=? ORDER BY created_at DESC LIMIT 1', (name,))
        return cur.fetchone()

    def list_models(self):
        cur = self.conn.cursor()
        cur.execute('SELECT * FROM models ORDER BY created_at DESC')
        return cur.fetchall() 