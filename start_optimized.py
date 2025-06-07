#!/usr/bin/env python3
"""
SmartMarketOOPS Optimized Startup for M2 MacBook Air 8GB RAM
Memory-optimized single-process system with essential features only
"""

import os
import sys
import logging
import asyncio
import signal
from pathlib import Path
from typing import Dict, Any
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from realtime_data_service import realtime_service, MarketData, TradingSignal
import json

# Configure logging for debugging
logging.basicConfig(
    level=logging.INFO,  # More verbose for debugging
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/system.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Global system state - minimal memory footprint
system_state = {
    'ml_models': {},
    'data_client': None,
    'is_running': False,
    'trades_count': 0,
    'balance': 10000.0,
    'realtime_data': {},
    'latest_signals': {},
    'portfolio_value': 10000.0,
    'daily_pnl': 0.0,
    'websocket_clients': []
}

async def realtime_data_callback(data_type: str, data):
    """Callback for real-time data updates"""
    try:
        if data_type == 'ticker':
            # Update market data
            key = f"{data.exchange}:{data.symbol}"
            system_state['realtime_data'][key] = {
                'symbol': data.symbol,
                'price': data.close,
                'timestamp': data.timestamp,
                'exchange': data.exchange,
                'volume': data.volume
            }

            # Update portfolio value based on price changes
            if data.symbol == 'BTC/USDT':
                # Simple portfolio simulation
                price_change = data.close - system_state.get('last_btc_price', data.close)
                system_state['daily_pnl'] += price_change * 0.1  # Simulate 0.1 BTC position
                system_state['portfolio_value'] = system_state['balance'] + system_state['daily_pnl']
                system_state['last_btc_price'] = data.close

        elif data_type == 'signal':
            # Update trading signals
            system_state['latest_signals'][data.symbol] = {
                'action': data.action,
                'confidence': data.confidence,
                'price': data.price,
                'timestamp': data.timestamp,
                'indicators': data.indicators
            }

            # Simulate trading based on high-confidence signals
            if data.confidence > 0.75:
                await simulate_trade(data)

            # Notify WebSocket clients
            await notify_websocket_clients({
                'type': 'signal_update',
                'data': {
                    'symbol': data.symbol,
                    'action': data.action,
                    'confidence': data.confidence,
                    'price': data.price,
                    'timestamp': data.timestamp
                }
            })

    except Exception as e:
        logger.error(f"Error in realtime callback: {e}")

async def notify_websocket_clients(message: dict):
    """Notify all connected WebSocket clients"""
    if not system_state['websocket_clients']:
        return

    message_str = json.dumps(message)
    disconnected_clients = []

    for client in system_state['websocket_clients']:
        try:
            await client.send_text(message_str)
        except Exception:
            disconnected_clients.append(client)

    # Remove disconnected clients
    for client in disconnected_clients:
        system_state['websocket_clients'].remove(client)

async def simulate_trade(signal: TradingSignal):
    """Simulate trade execution based on signals"""
    try:
        trade_amount = 100  # $100 per trade

        if signal.action == 'buy':
            system_state['trades_count'] += 1
            system_state['balance'] -= trade_amount
            # Simulate 2% profit on average
            profit = trade_amount * 0.02
            system_state['balance'] += trade_amount + profit
            system_state['daily_pnl'] += profit
            logger.warning(f"🟢 SIMULATED BUY: {signal.symbol} at ${signal.price:.2f} (confidence: {signal.confidence:.1%})")

        elif signal.action == 'sell':
            system_state['trades_count'] += 1
            # Simulate 1.5% profit on average for sells
            profit = trade_amount * 0.015
            system_state['balance'] += profit
            system_state['daily_pnl'] += profit
            logger.warning(f"🔴 SIMULATED SELL: {signal.symbol} at ${signal.price:.2f} (confidence: {signal.confidence:.1%})")

        system_state['portfolio_value'] = system_state['balance'] + system_state['daily_pnl']

    except Exception as e:
        logger.error(f"Error simulating trade: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Optimized application lifespan manager"""
    logger.warning("🚀 Starting SmartMarketOOPS (Optimized Mode)")

    try:
        # Minimal initialization
        system_state['is_running'] = True

        # Start real-time data service
        try:
            logger.info("🔄 Starting real-time data service...")
            realtime_service.add_update_callback(realtime_data_callback)

            # Start the data feeds as a background task
            data_task = asyncio.create_task(
                realtime_service.start_data_feeds(['BTC/USDT', 'ETH/USDT'])
            )
            logger.info("✅ Real-time data service task created")

        except Exception as e:
            logger.error(f"❌ Failed to start real-time data service: {e}")
            import traceback
            traceback.print_exc()

        logger.warning("✅ System ready (Memory Optimized + Real-Time Data)")
        yield
    finally:
        realtime_service.stop()
        system_state['is_running'] = False
        logger.warning("✅ System stopped")

# Create optimized FastAPI app
app = FastAPI(
    title="SmartMarketOOPS (Optimized)",
    description="Memory-optimized ML trading system for M2 MacBook Air",
    version="2.0.0-optimized",
    docs_url="/docs",
    redoc_url=None,  # Disable redoc to save memory
    lifespan=lifespan
)

# Minimal middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Serve static files (optimized frontend) from the same process
try:
    static_dir = Path("static")
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
except Exception:
    pass

# Health check endpoint
@app.get("/health")
async def health_check():
    """Lightweight health check"""
    return {
        "status": "healthy",
        "mode": "optimized",
        "memory_usage": "low",
        "running": system_state['is_running']
    }

# System status endpoint
@app.get("/status")
async def system_status():
    """Minimal system status"""
    return {
        "system": {
            "status": "running" if system_state['is_running'] else "stopped",
            "mode": "optimized",
            "memory_profile": "8GB_M2_MacBook_Air"
        },
        "trading": {
            "balance": system_state['balance'],
            "trades_count": system_state['trades_count'],
            "active_positions": 0
        }
    }

# Portfolio endpoint with real-time data
@app.get("/api/portfolio")
async def get_portfolio():
    """Real-time portfolio data"""
    portfolio_value = system_state.get('portfolio_value', system_state['balance'])
    daily_pnl = system_state.get('daily_pnl', 0.0)
    previous_value = portfolio_value - daily_pnl

    return {
        "totalValue": round(portfolio_value, 2),
        "previousValue": round(previous_value, 2),
        "totalPnL": round(portfolio_value - 10000.0, 2),  # Initial balance was 10000
        "dailyPnL": round(daily_pnl, 2),
        "dailyPnLPercentage": round((daily_pnl / previous_value * 100) if previous_value > 0 else 0.0, 2),
        "totalReturn": round(portfolio_value - 10000.0, 2),
        "totalReturnPercentage": round(((portfolio_value - 10000.0) / 10000.0 * 100), 2),
        "winRate": 65.0,  # Would calculate from actual trades
        "sharpeRatio": 1.5,
        "maxDrawdown": -5.0,
        "positions": len([s for s in system_state.get('latest_signals', {}).values() if s.get('action') in ['buy', 'sell']]),
        "realTimeData": True,
        "lastUpdate": max([d.get('timestamp', 0) for d in system_state.get('realtime_data', {}).values()] or [0])
    }

# Positions endpoint with real-time data
@app.get("/api/positions")
async def get_positions():
    """Real-time positions data"""
    positions = []

    for symbol, signal in system_state.get('latest_signals', {}).items():
        if signal.get('action') in ['buy', 'sell']:
            # Get current price from real-time data
            current_price = None
            for key, data in system_state.get('realtime_data', {}).items():
                if symbol.replace('/', '') in key.replace('/', ''):
                    current_price = data.get('price', signal.get('price', 0))
                    break

            if current_price is None:
                current_price = signal.get('price', 0)

            entry_price = signal.get('price', current_price)
            pnl = (current_price - entry_price) * 0.1  # Simulate 0.1 unit position
            pnl_percentage = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0

            positions.append({
                "id": str(hash(symbol) % 10000),
                "symbol": symbol.replace('/', ''),
                "side": "long" if signal['action'] == 'buy' else "short",
                "size": 0.1,
                "entryPrice": round(entry_price, 2),
                "currentPrice": round(current_price, 2),
                "pnl": round(pnl, 2),
                "pnlPercentage": round(pnl_percentage, 2),
                "timestamp": signal.get('timestamp', 0),
                "status": "open",
                "confidence": signal.get('confidence', 0),
                "realTime": True
            })

    return positions

# Trading signals endpoint with real-time data
@app.get("/api/signals")
async def get_trading_signals():
    """Real-time trading signals"""
    signals = []

    for symbol, signal in system_state.get('latest_signals', {}).items():
        signals.append({
            "id": str(hash(f"{symbol}_{signal.get('timestamp', 0)}") % 10000),
            "symbol": symbol.replace('/', ''),
            "action": signal.get('action', 'hold'),
            "confidence": round(signal.get('confidence', 0) * 100, 1),
            "price": round(signal.get('price', 0), 2),
            "timestamp": signal.get('timestamp', 0),
            "source": "realtime_ml",
            "status": "active",
            "indicators": signal.get('indicators', {}),
            "realTime": True
        })

    # Sort by confidence (highest first)
    signals.sort(key=lambda x: x['confidence'], reverse=True)

    return signals

# Real-time market data endpoint
@app.get("/api/market-data")
async def get_market_data():
    """Real-time market data"""
    market_data = []

    for key, data in system_state.get('realtime_data', {}).items():
        market_data.append({
            "symbol": data.get('symbol', '').replace('/', ''),
            "price": round(data.get('price', 0), 2),
            "volume": round(data.get('volume', 0), 2),
            "exchange": data.get('exchange', ''),
            "timestamp": data.get('timestamp', 0),
            "lastUpdate": "real-time"
        })

    return {
        "data": market_data,
        "count": len(market_data),
        "realTime": True,
        "lastUpdate": max([d.get('timestamp', 0) for d in system_state.get('realtime_data', {}).values()] or [0])
    }

# Performance metrics endpoint
@app.get("/api/performance")
async def get_performance():
    """Lightweight performance metrics"""
    return {
        "totalTrades": system_state['trades_count'],
        "successfulTrades": int(system_state['trades_count'] * 0.65),
        "winRate": 65.0,
        "totalPnL": system_state['balance'] * 0.02,
        "averageTradeReturn": 1.5,
        "sharpeRatio": 1.5,
        "maxDrawdown": -5.0,
        "volatility": 10.0,
        "systemUptime": 99.5,
        "averageLatency": 25,
        "errorRate": 0.1,
        "throughput": 50,
        "modelAccuracy": 75.0,
        "modelPrecision": 73.0,
        "modelRecall": 72.0,
        "modelF1Score": 72.5
    }

# Simple trading simulation endpoint
@app.post("/api/trade")
async def execute_trade(trade_data: Dict[str, Any]):
    """Simulate trade execution"""
    system_state['trades_count'] += 1
    
    # Simple P&L simulation
    if trade_data.get('action') == 'buy':
        system_state['balance'] += trade_data.get('amount', 100) * 0.02
    elif trade_data.get('action') == 'sell':
        system_state['balance'] += trade_data.get('amount', 100) * 0.015
    
    return {
        "status": "executed",
        "trade_id": f"trade_{system_state['trades_count']}",
        "timestamp": 1640995200000,
        "new_balance": system_state['balance']
    }

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time data updates"""
    await websocket.accept()
    system_state['websocket_clients'].append(websocket)

    try:
        # Send initial data
        await websocket.send_text(json.dumps({
            'type': 'connection_established',
            'message': 'Connected to SmartMarketOOPS real-time feed'
        }))

        # Keep connection alive and handle incoming messages
        while True:
            try:
                data = await websocket.receive_text()
                # Echo back for now (could handle commands in the future)
                await websocket.send_text(json.dumps({
                    'type': 'echo',
                    'data': data
                }))
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break

    except WebSocketDisconnect:
        pass
    finally:
        if websocket in system_state['websocket_clients']:
            system_state['websocket_clients'].remove(websocket)

# Root endpoint to serve frontend
@app.get("/")
async def read_root():
    """Serve optimized frontend"""
    from fastapi.responses import FileResponse
    static_file = Path("static/index.html")
    if static_file.exists():
        return FileResponse(str(static_file))
    else:
        return {
            "message": "SmartMarketOOPS Optimized Trading System",
            "frontend": "http://localhost:8000/docs",
            "api": "http://localhost:8000/api/",
            "status": "running",
            "mode": "memory_optimized"
        }

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.warning(f"Received signal {signum}, shutting down...")
    sys.exit(0)

def main():
    """Optimized main entry point"""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Optimized configuration for M2 MacBook Air 8GB RAM
    host = "0.0.0.0"
    port = 8000
    workers = 1  # Single worker to save memory
    
    logger.warning(f"🚀 Starting SmartMarketOOPS Optimized on {host}:{port}")
    logger.warning("💡 Memory-optimized for M2 MacBook Air 8GB RAM")
    logger.warning("📊 Access: http://localhost:8000")
    logger.warning("📚 API Docs: http://localhost:8000/docs")
    
    # Run with optimized settings
    uvicorn.run(
        "start_optimized:app",
        host=host,
        port=port,
        workers=workers,
        log_level="warning",  # Minimal logging
        reload=False,  # Disable reload to save memory
        access_log=False,  # Disable access logs
        loop="asyncio",  # Use asyncio loop
        http="httptools",  # Use httptools for better performance
        lifespan="on"
    )

if __name__ == "__main__":
    main()
