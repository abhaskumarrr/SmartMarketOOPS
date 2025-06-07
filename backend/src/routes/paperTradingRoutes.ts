import express from 'express';
import { marketDataService } from '../services/marketDataService';
import { logger } from '../utils/logger';

const router = express.Router();

// In-memory storage for paper trading data with REAL market data simulation
let paperTradingState = {
  portfolio: {
    balance: 1000, // $1000 initial balance as requested
    totalPnL: 0,
    positions: [],
    trades: [],
    isActive: true // Enable trading by default
  },
  config: {
    initialCapital: 1000, // $1000 initial capital
    leverage: 3,
    riskPerTrade: 0.02,
    assets: ['ETH/USDT', 'BTC/USDT', 'BTCUSD', 'ETHUSD'], // Support both formats
    stopLossPercentage: 0.025
  },
  lastUpdate: new Date()
};

// Start with clean slate - no positions, $1000 balance
paperTradingState.portfolio.positions = [];
paperTradingState.portfolio.trades = [];

// Get portfolio overview
router.get('/portfolio', async (req, res) => {
  try {
    // Get current prices for unrealized P&L calculation
    const positions = [];
    
    for (const position of paperTradingState.portfolio.positions) {
      if (position.status === 'open') {
        try {
          const currentPrice = await getCurrentPrice(position.symbol);
          const unrealizedPnL = position.side === 'buy' 
            ? (currentPrice - position.entryPrice) * position.size
            : (position.entryPrice - currentPrice) * position.size;
          
          positions.push({
            ...position,
            currentPrice,
            unrealizedPnL
          });
        } catch (error) {
          logger.error(`Error getting price for ${position.symbol}:`, error);
          positions.push({
            ...position,
            currentPrice: position.entryPrice,
            unrealizedPnL: 0
          });
        }
      } else {
        positions.push(position);
      }
    }
    
    const totalUnrealizedPnL = positions
      .filter(p => p.status === 'open')
      .reduce((sum, p) => sum + (p.unrealizedPnL || 0), 0);
    
    res.json({
      success: true,
      data: {
        ...paperTradingState.portfolio,
        positions,
        totalUnrealizedPnL,
        currentBalance: paperTradingState.portfolio.balance + paperTradingState.portfolio.totalPnL + totalUnrealizedPnL,
        lastUpdate: new Date()
      }
    });
  } catch (error) {
    logger.error('Error getting portfolio:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Get real-time market data
router.get('/market-data', async (req, res) => {
  try {
    const symbols = ['ETH/USDT', 'BTC/USDT'];
    const marketData = {};
    
    for (const symbol of symbols) {
      try {
        const ticker = await marketDataService.getMarketData(symbol);

        marketData[symbol] = {
          symbol,
          price: ticker?.price || 0,
          change: ticker?.change || '0.00',
          changePercent: ticker?.changePercent || 0,
          volume: ticker?.volume || 0,
          high24h: ticker?.high24h || ticker?.price || 0,
          low24h: ticker?.low24h || ticker?.price || 0,
          timestamp: new Date()
        };
      } catch (error) {
        logger.error(`Error getting market data for ${symbol}:`, error);
        marketData[symbol] = {
          symbol,
          price: 0,
          change: '0.00',
          changePercent: 0,
          volume: 0,
          high24h: 0,
          low24h: 0,
          timestamp: new Date()
        };
      }
    }
    
    res.json({
      success: true,
      data: marketData
    });
  } catch (error) {
    logger.error('Error getting market data:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Get trading system status
router.get('/status', async (req, res) => {
  try {
    const status = {
      isConnected: true,
      isTrading: paperTradingState.portfolio.isActive,
      exchange: 'Delta Exchange Indian Testnet',
      lastUpdate: paperTradingState.lastUpdate,
      config: paperTradingState.config,
      nextCycleIn: 30 - (Date.now() % 30000) / 1000, // Seconds until next 30s cycle
      systemHealth: 'healthy'
    };
    
    res.json({
      success: true,
      data: status
    });
  } catch (error) {
    logger.error('Error getting status:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Get historical price data for charts
router.get('/chart-data/:symbol', async (req, res) => {
  try {
    const { symbol } = req.params;
    const { timeframe = '1m', limit = '100' } = req.query;
    const limitNum = parseInt(limit as string) || 100;
    
    // Generate mock historical data (in production, fetch from exchange)
    const now = Date.now();
    const interval = timeframe === '1m' ? 60000 : 300000; // 1min or 5min
    const data = [];
    
    let basePrice = symbol === 'ETH/USDT' ? 2567 : 105444;
    
    for (let i = limitNum; i >= 0; i--) {
      const timestamp = now - (i * interval);
      const volatility = 0.002; // 0.2% volatility
      const change = (Math.random() - 0.5) * volatility;
      
      const open = basePrice;
      const close = basePrice * (1 + change);
      const high = Math.max(open, close) * (1 + Math.random() * 0.001);
      const low = Math.min(open, close) * (1 - Math.random() * 0.001);
      const volume = Math.random() * 1000;
      
      data.push({
        timestamp,
        open: parseFloat(open.toFixed(2)),
        high: parseFloat(high.toFixed(2)),
        low: parseFloat(low.toFixed(2)),
        close: parseFloat(close.toFixed(2)),
        volume: parseFloat(volume.toFixed(2))
      });
      
      basePrice = close;
    }
    
    res.json({
      success: true,
      data: {
        symbol,
        timeframe,
        candles: data
      }
    });
  } catch (error) {
    logger.error('Error getting chart data:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Execute a simulated trade with real market data
router.post('/trade', async (req, res) => {
  try {
    const { symbol, side, size, orderType = 'market_order' } = req.body;

    if (!symbol || !side || !size) {
      return res.status(400).json({
        success: false,
        error: 'Missing required fields: symbol, side, size'
      });
    }

    // Get current market price
    const currentPrice = await getCurrentPrice(symbol);
    if (!currentPrice || currentPrice === 0) {
      return res.status(400).json({
        success: false,
        error: `Unable to get current price for ${symbol}`
      });
    }

    // Calculate trade value
    const tradeValue = currentPrice * parseFloat(size);
    const commission = tradeValue * 0.001; // 0.1% commission

    // Check if we have enough balance for buy orders
    if (side === 'buy' && (tradeValue + commission) > paperTradingState.portfolio.balance) {
      return res.status(400).json({
        success: false,
        error: 'Insufficient balance for trade'
      });
    }

    // Create trade record
    const trade = {
      id: `trade_${Date.now()}`,
      symbol,
      side,
      size: parseFloat(size),
      price: currentPrice,
      orderType,
      status: 'filled',
      timestamp: new Date().toISOString(),
      commission,
      value: tradeValue
    };

    // Update portfolio
    if (side === 'buy') {
      paperTradingState.portfolio.balance -= (tradeValue + commission);
    } else {
      paperTradingState.portfolio.balance += (tradeValue - commission);
    }

    // Add to trades history
    paperTradingState.portfolio.trades.push(trade);

    // Update positions (simplified - just track the trade)
    const existingPositionIndex = paperTradingState.portfolio.positions.findIndex(
      p => p.symbol === symbol && p.status === 'open'
    );

    if (existingPositionIndex >= 0) {
      const position = paperTradingState.portfolio.positions[existingPositionIndex];
      if (position.side === side) {
        // Add to existing position
        const totalSize = position.size + parseFloat(size);
        const totalValue = (position.entryPrice * position.size) + (currentPrice * parseFloat(size));
        position.entryPrice = totalValue / totalSize;
        position.size = totalSize;
      } else {
        // Opposite side - close or reduce position
        if (position.size > parseFloat(size)) {
          position.size -= parseFloat(size);
        } else {
          // Close position
          position.status = 'closed';
          position.closePrice = currentPrice;
          position.closeTime = new Date().toISOString();
        }
      }
    } else {
      // Create new position
      const newPosition = {
        id: `pos_${Date.now()}`,
        symbol,
        side,
        size: parseFloat(size),
        entryPrice: currentPrice,
        openTime: new Date().toISOString(),
        status: 'open'
      };
      paperTradingState.portfolio.positions.push(newPosition);
    }

    paperTradingState.lastUpdate = new Date();

    logger.info(`🎯 Simulated trade executed: ${side.toUpperCase()} ${size} ${symbol} @ $${currentPrice}`);

    res.json({
      success: true,
      data: {
        trade,
        orderId: trade.id,
        status: 'filled',
        executedPrice: currentPrice,
        executedSize: size,
        commission,
        timestamp: trade.timestamp
      }
    });
  } catch (error) {
    logger.error('Error executing simulated trade:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Update paper trading state (called by trading system)
router.post('/update', async (req, res) => {
  try {
    const { portfolio, config } = req.body;

    if (portfolio) {
      paperTradingState.portfolio = { ...paperTradingState.portfolio, ...portfolio };
    }

    if (config) {
      paperTradingState.config = { ...paperTradingState.config, ...config };
    }

    paperTradingState.lastUpdate = new Date();

    res.json({
      success: true,
      message: 'Paper trading state updated'
    });
  } catch (error) {
    logger.error('Error updating paper trading state:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Helper function to get current price with fallback to realistic mock data
async function getCurrentPrice(symbol: string): Promise<number> {
  try {
    // Try to get real market data first
    const marketData = await marketDataService.getMarketData(symbol);
    if (marketData?.price && marketData.price > 0) {
      return marketData.price;
    }
  } catch (error) {
    logger.error(`Error fetching price for ${symbol}:`, error);
  }

  // Fallback to realistic mock prices when external APIs are having issues
  const mockPrices: { [key: string]: number } = {
    'BTC/USDT': 105563.43,
    'ETH/USDT': 2579.39,
    'BTCUSD': 105563.43,
    'ETHUSD': 2579.39,
    'BTC/USD': 105563.43,
    'ETH/USD': 2579.39
  };

  const price = mockPrices[symbol] || mockPrices[symbol.replace('USD', '/USDT')] || 50000;
  logger.info(`📊 Using mock price for ${symbol}: $${price} (external APIs having issues)`);
  return price;
}

export default router;
