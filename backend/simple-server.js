/**
 * Simple Backend Server for Phase 1 Testing
 * Provides basic paper trading API endpoints
 */

const express = require('express');
const cors = require('cors');
const path = require('path');

// Load environment variables
require('dotenv').config({
  path: path.resolve(__dirname, '.env')
});

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors({
  origin: process.env.CORS_ORIGIN || 'http://localhost:3000',
  credentials: true
}));
app.use(express.json());

// Request logging
app.use((req, res, next) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.path}`);
  next();
});

// Mock paper trading state
let paperTradingState = {
  portfolio: {
    balance: 2000,
    totalPnL: 125.50,
    currentBalance: 2125.50,
    positions: [
      {
        id: 'pos_1',
        symbol: 'ETH/USDT',
        side: 'buy',
        size: 0.5,
        entryPrice: 2580.00,
        currentPrice: 2620.00,
        unrealizedPnL: 20.00,
        stopLoss: 2450.00,
        takeProfitLevels: [
          { percentage: 25, ratio: 2, price: 2700.00, executed: false },
          { percentage: 50, ratio: 3, price: 2800.00, executed: false },
          { percentage: 75, ratio: 5, price: 3000.00, executed: false }
        ],
        openTime: new Date(Date.now() - 3600000).toISOString(),
        status: 'open'
      }
    ],
    trades: [
      {
        id: 'trade_1',
        symbol: 'BTC/USDT',
        side: 'buy',
        size: 0.01,
        entryPrice: 45000,
        exitPrice: 46500,
        pnl: 15.00,
        timestamp: new Date(Date.now() - 7200000).toISOString()
      }
    ]
  },
  lastUpdate: new Date()
};

// Mock market data with current realistic prices (Dec 2024)
const marketData = {
  'ETH/USDT': {
    symbol: 'ETH/USDT',
    price: 2567.58,
    change: '+32.15',
    changePercent: 1.27,
    volume: 125000,
    high24h: 2610.00,
    low24h: 2520.00,
    timestamp: new Date().toISOString()
  },
  'BTC/USDT': {
    symbol: 'BTC/USDT',
    price: 105444.00,
    change: '+1850.00',
    changePercent: 1.78,
    volume: 85000,
    high24h: 106200.00,
    low24h: 103800.00,
    timestamp: new Date().toISOString()
  },
  'SOL/USDT': {
    symbol: 'SOL/USDT',
    price: 154.24,
    change: '+2.18',
    changePercent: 1.43,
    volume: 95000,
    high24h: 158.50,
    low24h: 151.00,
    timestamp: new Date().toISOString()
  }
};

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({
    success: true,
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    environment: process.env.NODE_ENV || 'development'
  });
});

// Paper trading endpoints
app.get('/api/paper-trading/portfolio', (req, res) => {
  try {
    console.log('📊 Portfolio request received');
    
    // Update unrealized P&L for open positions
    const updatedPositions = paperTradingState.portfolio.positions.map(position => {
      if (position.status === 'open' && marketData[position.symbol]) {
        const currentPrice = marketData[position.symbol].price;
        const priceDiff = currentPrice - position.entryPrice;
        const unrealizedPnL = position.side === 'buy' 
          ? priceDiff * position.size 
          : -priceDiff * position.size;
        
        return {
          ...position,
          currentPrice,
          unrealizedPnL: parseFloat(unrealizedPnL.toFixed(2))
        };
      }
      return position;
    });

    const totalUnrealizedPnL = updatedPositions
      .filter(p => p.status === 'open')
      .reduce((sum, p) => sum + (p.unrealizedPnL || 0), 0);

    const response = {
      success: true,
      data: {
        ...paperTradingState.portfolio,
        positions: updatedPositions,
        totalUnrealizedPnL: parseFloat(totalUnrealizedPnL.toFixed(2)),
        currentBalance: paperTradingState.portfolio.balance + 
                       paperTradingState.portfolio.totalPnL + 
                       totalUnrealizedPnL,
        lastUpdate: new Date()
      }
    };

    console.log('✅ Portfolio data sent successfully');
    res.json(response);
  } catch (error) {
    console.error('❌ Portfolio error:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

app.get('/api/paper-trading/market-data', (req, res) => {
  try {
    console.log('📈 Market data request received');
    
    // Add some random price fluctuation
    Object.keys(marketData).forEach(symbol => {
      const data = marketData[symbol];
      const fluctuation = (Math.random() - 0.5) * 0.02; // ±1% fluctuation
      const newPrice = data.price * (1 + fluctuation);
      const change = newPrice - data.price;
      const changePercent = (change / data.price) * 100;
      
      marketData[symbol] = {
        ...data,
        price: parseFloat(newPrice.toFixed(2)),
        change: change >= 0 ? `+${change.toFixed(2)}` : change.toFixed(2),
        changePercent: parseFloat(changePercent.toFixed(2)),
        timestamp: new Date().toISOString()
      };
    });

    const response = {
      success: true,
      data: marketData
    };

    console.log('✅ Market data sent successfully');
    res.json(response);
  } catch (error) {
    console.error('❌ Market data error:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Chart data endpoint
app.get('/api/paper-trading/chart-data', (req, res) => {
  try {
    const { symbol = 'ETH/USDT', timeframe = '1m', limit = 100 } = req.query;
    console.log(`📊 Chart data request: ${symbol} ${timeframe} (${limit} candles)`);
    
    // Generate mock candlestick data
    const candles = [];
    const now = Date.now();
    const timeframeMs = 60000; // 1 minute
    const basePrice = marketData[symbol]?.price || 105444;
    let currentPrice = basePrice;

    for (let i = parseInt(limit) - 1; i >= 0; i--) {
      const time = now - (i * timeframeMs);
      const volatility = 0.01; // 1% volatility
      const change = (Math.random() - 0.5) * volatility * currentPrice;
      const open = currentPrice;
      const close = open + change;
      const high = Math.max(open, close) + Math.random() * 0.005 * currentPrice;
      const low = Math.min(open, close) - Math.random() * 0.005 * currentPrice;
      const volume = Math.random() * 1000000 + 500000;

      candles.push({
        time,
        open: parseFloat(open.toFixed(2)),
        high: parseFloat(high.toFixed(2)),
        low: parseFloat(low.toFixed(2)),
        close: parseFloat(close.toFixed(2)),
        volume: Math.floor(volume)
      });

      currentPrice = close;
    }

    const response = {
      success: true,
      data: {
        symbol,
        timeframe,
        candles
      }
    };

    console.log('✅ Chart data sent successfully');
    res.json(response);
  } catch (error) {
    console.error('❌ Chart data error:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Server error:', err);
  res.status(500).json({
    success: false,
    error: 'Internal server error',
    message: err.message
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    success: false,
    error: 'Not found',
    message: `Route ${req.method} ${req.path} not found`
  });
});

// Start server
app.listen(PORT, () => {
  console.log('🚀 SmartMarketOOPS Backend Server Started');
  console.log('─'.repeat(50));
  console.log(`📡 Server running on port ${PORT}`);
  console.log(`🌐 Environment: ${process.env.NODE_ENV || 'development'}`);
  console.log(`🔗 CORS Origin: ${process.env.CORS_ORIGIN || 'http://localhost:3000'}`);
  console.log('─'.repeat(50));
  console.log('📊 Available endpoints:');
  console.log('  GET /api/health');
  console.log('  GET /api/paper-trading/portfolio');
  console.log('  GET /api/paper-trading/market-data');
  console.log('  GET /api/paper-trading/chart-data');
  console.log('─'.repeat(50));
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully');
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('SIGINT received, shutting down gracefully');
  process.exit(0);
});
