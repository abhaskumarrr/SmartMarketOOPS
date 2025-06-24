#!/usr/bin/env python3
"""
SmartMarketOOPS Simple Version - Debug and Fix Issues
Simplified version to identify and fix problems
"""

import os
import sys
import logging
import asyncio
import signal
from pathlib import Path
from typing import Dict, Any
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import ccxt
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/simple.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Global system state
system_state = {
    'is_running': False,
    'balance': 10000.0,
    'market_data': {},
    'exchanges': {},
    'last_update': 0
}

async def fetch_market_data():
    """Simple market data fetching"""
    try:
        logger.info("🔄 Fetching market data...")
        
        # Initialize Binance exchange
        if 'binance' not in system_state['exchanges']:
            system_state['exchanges']['binance'] = ccxt.binance({
                'enableRateLimit': True,
                'timeout': 10000,
                'sandbox': False
            })
            logger.info("✅ Binance exchange initialized")
        
        exchange = system_state['exchanges']['binance']
        
        # Fetch BTC and ETH prices
        symbols = ['BTC/USDT', 'ETH/USDT']
        
        for symbol in symbols:
            try:
                ticker = exchange.fetch_ticker(symbol)
                system_state['market_data'][symbol] = {
                    'symbol': symbol,
                    'price': ticker['close'],
                    'volume': ticker['baseVolume'],
                    'timestamp': ticker['timestamp'],
                    'change_24h': ticker['percentage']
                }
                logger.info(f"📊 {symbol}: ${ticker['close']:,.2f}")
                
            except Exception as e:
                logger.error(f"❌ Error fetching {symbol}: {e}")
        
        system_state['last_update'] = asyncio.get_event_loop().time()
        logger.info("✅ Market data updated successfully")
        
    except Exception as e:
        logger.error(f"❌ Error in fetch_market_data: {e}")
        import traceback
        traceback.print_exc()

async def market_data_loop():
    """Background task to fetch market data periodically"""
    while system_state['is_running']:
        try:
            await fetch_market_data()
            await asyncio.sleep(30)  # Update every 30 seconds
        except Exception as e:
            logger.error(f"❌ Error in market data loop: {e}")
            await asyncio.sleep(60)  # Wait longer on error

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("🚀 Starting SmartMarketOOPS Simple Version")
    
    try:
        system_state['is_running'] = True
        
        # Start market data background task
        market_task = asyncio.create_task(market_data_loop())
        logger.info("✅ Market data task started")
        
        # Initial data fetch
        await fetch_market_data()
        
        logger.info("✅ System ready")
        yield
        
    except Exception as e:
        logger.error(f"❌ Error in lifespan: {e}")
        import traceback
        traceback.print_exc()
        yield
    finally:
        system_state['is_running'] = False
        logger.info("✅ System stopped")

# Create FastAPI app
app = FastAPI(
    title="SmartMarketOOPS Simple",
    description="Simplified version for debugging",
    version="1.0.0-simple",
    docs_url="/docs",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
try:
    static_dir = Path("static")
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
except Exception as e:
    logger.warning(f"⚠️ Could not mount static files: {e}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """System health check"""
    return {
        "status": "healthy" if system_state['is_running'] else "stopped",
        "timestamp": system_state.get('last_update', 0),
        "market_data_count": len(system_state.get('market_data', {})),
        "exchanges": list(system_state.get('exchanges', {}).keys())
    }

# Portfolio endpoint
@app.get("/api/portfolio")
async def get_portfolio():
    """Portfolio data"""
    return {
        "totalValue": system_state['balance'],
        "previousValue": system_state['balance'] * 0.98,
        "totalPnL": system_state['balance'] * 0.02,
        "dailyPnL": system_state['balance'] * 0.01,
        "dailyPnLPercentage": 1.0,
        "totalReturn": system_state['balance'] * 0.02,
        "totalReturnPercentage": 2.0,
        "winRate": 65.0,
        "sharpeRatio": 1.5,
        "maxDrawdown": -5.0,
        "positions": 2,
        "lastUpdate": system_state.get('last_update', 0)
    }

# Market data endpoint
@app.get("/api/market-data")
async def get_market_data():
    """Real-time market data"""
    market_data = []
    
    for symbol, data in system_state.get('market_data', {}).items():
        market_data.append({
            "symbol": symbol.replace('/', ''),
            "price": round(data.get('price', 0), 2),
            "volume": round(data.get('volume', 0), 2),
            "change_24h": round(data.get('change_24h', 0), 2),
            "timestamp": data.get('timestamp', 0),
            "exchange": "binance"
        })
    
    return {
        "data": market_data,
        "count": len(market_data),
        "lastUpdate": system_state.get('last_update', 0)
    }

# Trading signals endpoint
@app.get("/api/signals")
async def get_trading_signals():
    """Simple trading signals"""
    signals = []
    
    for symbol, data in system_state.get('market_data', {}).items():
        # Simple signal logic based on 24h change
        change = data.get('change_24h', 0)
        
        if change > 5:
            action = 'buy'
            confidence = min(75 + change, 95)
        elif change < -5:
            action = 'sell'
            confidence = min(75 + abs(change), 95)
        else:
            action = 'hold'
            confidence = 50
        
        signals.append({
            "id": str(hash(symbol) % 10000),
            "symbol": symbol.replace('/', ''),
            "action": action,
            "confidence": round(confidence, 1),
            "price": round(data.get('price', 0), 2),
            "timestamp": data.get('timestamp', 0),
            "source": "simple_analysis",
            "status": "active"
        })
    
    return signals

# Performance metrics endpoint
@app.get("/api/performance")
async def get_performance():
    """Performance metrics"""
    return {
        "totalTrades": 10,
        "successfulTrades": 7,
        "winRate": 70.0,
        "totalPnL": 200.0,
        "averageTradeReturn": 2.0,
        "sharpeRatio": 1.5,
        "maxDrawdown": -5.0,
        "volatility": 10.0,
        "systemUptime": 99.5,
        "averageLatency": 25,
        "errorRate": 0.1,
        "throughput": 50,
        "modelAccuracy": 70.0,
        "modelPrecision": 68.0,
        "modelRecall": 72.0,
        "modelF1Score": 70.0
    }

# Root endpoint
@app.get("/")
async def read_root():
    """Serve frontend"""
    from fastapi.responses import FileResponse
    static_file = Path("static/index.html")
    if static_file.exists():
        return FileResponse(str(static_file))
    else:
        return {
            "message": "SmartMarketOOPS Simple Version",
            "status": "running",
            "docs": "/docs",
            "health": "/health"
        }

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)

def main():
    """Main entry point"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    host = "0.0.0.0"
    port = 8000
    
    logger.info(f"🚀 Starting SmartMarketOOPS Simple on {host}:{port}")
    
    uvicorn.run(
        "start_simple:app",
        host=host,
        port=port,
        workers=1,
        log_level="info",
        reload=False,
        access_log=True
    )

if __name__ == "__main__":
    main()
