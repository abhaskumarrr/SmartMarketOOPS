import sqlite3
from datetime import datetime
import torch
import joblib
import os
import json
import logging

logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self, db_path='model_registry.db'):
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    version TEXT NOT NULL UNIQUE,
                    path TEXT NOT NULL,
                    metrics TEXT,
                    metadata TEXT,
                    preprocessor_path TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

    def save_model(self, model, symbol: str, model_type: str, metrics: dict, metadata: dict, preprocessor=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"{symbol.replace('/', '_')}_{model_type}_{timestamp}"
        model_filename = f"{version}.pt"
        model_path = os.path.join(self.model_dir, model_filename)
        
        logger.info(f"Saving model to {model_path}")
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved successfully to {model_path}")

        preprocessor_path = None
        if preprocessor:
            preprocessor_filename = f"{version}_preprocessor.joblib"
            preprocessor_path = os.path.join(self.model_dir, preprocessor_filename)
            logger.info(f"Saving preprocessor to {preprocessor_path}")
            joblib.dump(preprocessor, preprocessor_path)
            logger.info(f"Preprocessor saved successfully to {preprocessor_path}")

        self.register_model(
            symbol=symbol,
            model_type=model_type,
            version=version,
            path=model_path,
            metrics=json.dumps(metrics),
            metadata=json.dumps(metadata),
            preprocessor_path=preprocessor_path
        )
        return version

    def register_model(self, symbol, model_type, version, path, metrics, metadata, preprocessor_path):
        with self.conn:
            self.conn.execute(
                'INSERT INTO models (symbol, model_type, version, path, metrics, metadata, preprocessor_path) VALUES (?, ?, ?, ?, ?, ?, ?)',
                (symbol, model_type, version, path, metrics, metadata, preprocessor_path)
            )

    def get_model(self, symbol, version=None):
        cur = self.conn.cursor()
        if version:
            cur.execute('SELECT * FROM models WHERE symbol=? AND version=?', (symbol, version))
        else:
            cur.execute('SELECT * FROM models WHERE symbol=? ORDER BY created_at DESC LIMIT 1', (symbol,))
        row = cur.fetchone()
        if row:
            # Reconstruct model object (requires model definition)
            # For now, just return path and metadata
            return {
                "symbol": row[1],
                "model_type": row[2],
                "version": row[3],
                "path": row[4],
                "metrics": json.loads(row[5]),
                "metadata": json.loads(row[6]),
                "preprocessor_path": row[7]
            }
        return None

    def list_models(self):
        cur = self.conn.cursor()
        cur.execute('SELECT * FROM models ORDER BY created_at DESC')
        return cur.fetchall() 