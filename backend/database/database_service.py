#!/usr/bin/env python3
"""
Comprehensive Database Service for SmartMarket Trading System
Integrates Redis (real-time data) and QuestDB (time-series persistence)
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import redis.asyncio as redis
import psycopg2
from psycopg2.extras import RealDictCursor
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseService:
    """Unified database service for real-time and persistent storage"""
    
    def __init__(self):
        self.redis_client = None
        self.questdb_pool = None
        self.is_connected = False
        
        # Configuration
        self.redis_config = {
            'host': 'localhost',
            'port': 6379,
            'db': 0,
            'decode_responses': True
        }
        
        self.questdb_config = {
            'host': 'localhost',
            'port': 8812,
            'user': 'admin',
            'password': 'quest',
            'database': 'qdb'
        }
        
    async def initialize(self):
        """Initialize all database connections"""
        try:
            # Initialize Redis connection
            await self._init_redis()
            
            # Initialize QuestDB connection
            await self._init_questdb()
            
            # Create tables if they don't exist
            await self._create_tables()
            
            self.is_connected = True
            logger.info("✅ Database service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database service: {e}")
            raise
    
    async def _init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(**self.redis_config)
            
            # Test connection
            await self.redis_client.ping()
            logger.info("✅ Redis connection established")
            
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            raise
    
    async def _init_questdb(self):
        """Initialize QuestDB connection"""
        try:
            # QuestDB uses PostgreSQL wire protocol
            import asyncpg
            
            self.questdb_pool = await asyncpg.create_pool(
                host=self.questdb_config['host'],
                port=self.questdb_config['port'],
                user=self.questdb_config['user'],
                password=self.questdb_config['password'],
                database=self.questdb_config['database'],
                min_size=1,
                max_size=10
            )
            
            logger.info("✅ QuestDB connection established")
            
        except Exception as e:
            logger.warning(f"⚠️ QuestDB connection failed, using fallback: {e}")
            # Continue without QuestDB for now
            self.questdb_pool = None
    
    async def _create_tables(self):
        """Create necessary tables in QuestDB"""
        if not self.questdb_pool:
            logger.warning("⚠️ Skipping table creation - QuestDB not available")
            return
            
        tables = [
            # Market data table
            """
            CREATE TABLE IF NOT EXISTS market_data (
                timestamp TIMESTAMP,
                symbol SYMBOL,
                price DOUBLE,
                volume DOUBLE,
                bid DOUBLE,
                ask DOUBLE,
                change_24h DOUBLE,
                source STRING
            ) timestamp(timestamp) PARTITION BY DAY;
            """,
            
            # Trading signals table
            """
            CREATE TABLE IF NOT EXISTS trading_signals (
                timestamp TIMESTAMP,
                symbol SYMBOL,
                signal_type STRING,
                price DOUBLE,
                confidence DOUBLE,
                quality DOUBLE,
                source STRING
            ) timestamp(timestamp) PARTITION BY DAY;
            """,
            
            # Trades table
            """
            CREATE TABLE IF NOT EXISTS trades (
                timestamp TIMESTAMP,
                trade_id STRING,
                symbol SYMBOL,
                side STRING,
                quantity DOUBLE,
                price DOUBLE,
                status STRING,
                pnl DOUBLE,
                fees DOUBLE
            ) timestamp(timestamp) PARTITION BY DAY;
            """,
            
            # Portfolio snapshots table
            """
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                timestamp TIMESTAMP,
                total_value DOUBLE,
                available_balance DOUBLE,
                unrealized_pnl DOUBLE,
                realized_pnl DOUBLE,
                positions_count INT
            ) timestamp(timestamp) PARTITION BY DAY;
            """
        ]
        
        try:
            async with self.questdb_pool.acquire() as conn:
                for table_sql in tables:
                    await conn.execute(table_sql)
            
            logger.info("✅ QuestDB tables created/verified")
            
        except Exception as e:
            logger.error(f"❌ Failed to create QuestDB tables: {e}")
    
    # Redis Operations (Real-time data)
    async def store_market_data_redis(self, symbol: str, data: Dict[str, Any]):
        """Store real-time market data in Redis"""
        try:
            key = f"market_data:{symbol}"
            
            # Add timestamp
            data['timestamp'] = time.time()
            data['symbol'] = symbol
            
            # Store as JSON with expiration (1 hour)
            await self.redis_client.setex(
                key, 
                3600,  # 1 hour expiration
                json.dumps(data)
            )
            
            # Also add to a stream for real-time updates
            stream_key = f"market_stream:{symbol}"
            await self.redis_client.xadd(
                stream_key,
                data,
                maxlen=1000  # Keep last 1000 entries
            )
            
            logger.debug(f"📊 Stored market data for {symbol} in Redis")
            
        except Exception as e:
            logger.error(f"❌ Failed to store market data in Redis: {e}")
    
    async def store_trading_signal_redis(self, signal: Dict[str, Any]):
        """Store trading signal in Redis"""
        try:
            signal_id = f"signal:{signal['symbol']}:{int(time.time())}"
            
            # Store signal with expiration (24 hours)
            await self.redis_client.setex(
                signal_id,
                86400,  # 24 hours
                json.dumps(signal)
            )
            
            # Add to signals stream
            await self.redis_client.xadd(
                "trading_signals_stream",
                signal,
                maxlen=5000  # Keep last 5000 signals
            )
            
            logger.debug(f"🎯 Stored trading signal for {signal['symbol']} in Redis")
            
        except Exception as e:
            logger.error(f"❌ Failed to store trading signal in Redis: {e}")
    
    async def get_latest_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest market data from Redis"""
        try:
            key = f"market_data:{symbol}"
            data = await self.redis_client.get(key)
            
            if data:
                return json.loads(data)
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get market data from Redis: {e}")
            return None
    
    async def get_recent_signals(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent trading signals from Redis stream"""
        try:
            # Read from trading signals stream
            messages = await self.redis_client.xrevrange(
                "trading_signals_stream",
                count=limit
            )
            
            signals = []
            for msg_id, fields in messages:
                signal = dict(fields)
                signal['id'] = msg_id
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Failed to get recent signals from Redis: {e}")
            return []
    
    # QuestDB Operations (Persistent time-series data)
    async def store_market_data_questdb(self, symbol: str, data: Dict[str, Any]):
        """Store market data in QuestDB for long-term analysis"""
        if not self.questdb_pool:
            return
            
        try:
            async with self.questdb_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO market_data (
                        timestamp, symbol, price, volume, bid, ask, change_24h, source
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                    datetime.utcnow(),
                    symbol,
                    float(data.get('price', 0)),
                    float(data.get('volume', 0)),
                    float(data.get('bid', 0)),
                    float(data.get('ask', 0)),
                    float(data.get('change_24h', 0)),
                    data.get('source', 'delta_exchange')
                )
            
            logger.debug(f"💾 Stored market data for {symbol} in QuestDB")
            
        except Exception as e:
            logger.error(f"❌ Failed to store market data in QuestDB: {e}")
    
    async def store_trading_signal_questdb(self, signal: Dict[str, Any]):
        """Store trading signal in QuestDB"""
        if not self.questdb_pool:
            return
            
        try:
            async with self.questdb_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO trading_signals (
                        timestamp, symbol, signal_type, price, confidence, quality, source
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                    datetime.utcnow(),
                    signal['symbol'],
                    signal['signal_type'],
                    float(signal['price']),
                    float(signal.get('confidence', 0.5)),
                    float(signal.get('quality', 0.5)) if isinstance(signal.get('quality'), (int, float)) else 0.5,
                    signal.get('source', 'ai_model')
                )
            
            logger.debug(f"💾 Stored trading signal for {signal['symbol']} in QuestDB")
            
        except Exception as e:
            logger.error(f"❌ Failed to store trading signal in QuestDB: {e}")
    
    async def get_historical_data(self, symbol: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical market data from QuestDB"""
        if not self.questdb_pool:
            return []
            
        try:
            async with self.questdb_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM market_data 
                    WHERE symbol = $1 
                    AND timestamp > dateadd('h', -$2, now())
                    ORDER BY timestamp DESC
                    LIMIT 1000
                """, symbol, hours)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"❌ Failed to get historical data from QuestDB: {e}")
            return []
    
    async def cleanup(self):
        """Clean up database connections"""
        try:
            if self.redis_client:
                await self.redis_client.close()
                
            if self.questdb_pool:
                await self.questdb_pool.close()
                
            logger.info("✅ Database connections closed")
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

# Global database service instance
db_service = DatabaseService()

async def main():
    """Test the database service"""
    try:
        await db_service.initialize()
        
        # Test Redis operations
        test_data = {
            'price': 105000.50,
            'volume': 1234.56,
            'bid': 104999.00,
            'ask': 105001.00,
            'change_24h': 2.5
        }
        
        await db_service.store_market_data_redis('BTCUSD', test_data)
        retrieved = await db_service.get_latest_market_data('BTCUSD')
        logger.info(f"✅ Redis test successful: {retrieved}")
        
        # Test QuestDB operations
        await db_service.store_market_data_questdb('BTCUSD', test_data)
        
        # Test trading signal
        test_signal = {
            'symbol': 'BTCUSD',
            'signal_type': 'buy',
            'price': 105000.50,
            'confidence': 0.85,
            'quality': 0.9
        }
        
        await db_service.store_trading_signal_redis(test_signal)
        await db_service.store_trading_signal_questdb(test_signal)
        
        signals = await db_service.get_recent_signals(5)
        logger.info(f"✅ Signals test successful: {len(signals)} signals retrieved")
        
    except Exception as e:
        logger.error(f"❌ Database test failed: {e}")
    finally:
        await db_service.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
