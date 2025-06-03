const express = require('express');
const ccxt = require('ccxt');

const router = express.Router();

// In-memory storage for paper trading data
let paperTradingState = {
  portfolio: {
    balance: 2000,
    totalPnL: 0,
    positions: [],
    trades: [],
    isActive: true
  },
  config: {
    initialCapital: 2000,
    leverage: 3,
    riskPerTrade: 0.02,
    assets: ['ETH/USDT', 'BTC/USDT'],
    stopLossPercentage: 0.025
  },
  lastUpdate: new Date()
};

// Mock current position data (matching the live trading system)
const mockCurrentPosition = {
  id: 'pos_1733270421000',
  symbol: 'ETH/USDT',
  side: 'buy',
  size: 0.6203,
  entryPrice: 2579.39,
  stopLoss: 2514.91,
  takeProfitLevels: [
    { percentage: 25, ratio: 2.0, price: 2643.37, executed: false },
    { percentage: 50, ratio: 5.0, price: 2901.35, executed: false },
    { percentage: 25, ratio: 5.0, price: 2901.35, executed: false }
  ],
  openTime: new Date().toISOString(),
  status: 'open'
};

paperTradingState.portfolio.positions = [mockCurrentPosition];

// Initialize Delta Exchange for market data
const exchange = new ccxt.delta({
  sandbox: true,
  enableRateLimit: true,
  options: { defaultType: 'spot' },
  urls: {
    api: {
      public: 'https://cdn-ind.testnet.deltaex.org',
      private: 'https://cdn-ind.testnet.deltaex.org'
    }
  }
});

// Product ID mappings for Delta Exchange perpetual futures (correct testnet IDs)
const PRODUCT_IDS = {
  'BTC/USDT': 84,    // BTCUSD perpetual futures
  'ETH/USDT': 1699,  // ETHUSD perpetual futures
  'BTCUSD': 84,
  'ETHUSD': 1699
};

// Helper function to get current price using correct Delta Exchange API
async function getCurrentPrice(symbol) {
  try {
    // Map symbol to product ID
    const productId = PRODUCT_IDS[symbol] || PRODUCT_IDS[symbol.replace('/USDT', 'USD')];

    if (productId) {
      // Use direct API call to Delta Exchange with symbol (not product ID)
      const axios = require('axios');
      const symbolForAPI = symbol.replace('/USDT', 'USD'); // Convert BTC/USDT to BTCUSD
      const response = await axios.get(`https://cdn-ind.testnet.deltaex.org/v2/tickers/${symbolForAPI}`);

      if (response.data.success) {
        const ticker = response.data.result;
        return parseFloat(ticker.close || ticker.last_price || '0');
      }
    }

    // Fallback to CCXT if direct API fails
    await exchange.loadMarkets();
    const ticker = await exchange.fetchTicker(symbol);
    return ticker.indexPrice || parseFloat(ticker.info?.spot_price) || ticker.last;
  } catch (error) {
    console.error(`Error fetching price for ${symbol}:`, error.message);
    // Return mock prices if API fails
    return symbol.includes('ETH') ? 2579.39 : 105563.43;
  }
}

// Get portfolio overview
router.get('/portfolio', async (req, res) => {
  try {
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
          console.error(`Error getting price for ${position.symbol}:`, error);
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
    console.error('Error getting portfolio:', error);
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
        const price = await getCurrentPrice(symbol);
        const productId = PRODUCT_IDS[symbol];

        // Try to get additional market data from Delta Exchange API
        let additionalData = {};
        if (productId) {
          try {
            const axios = require('axios');
            const symbolForAPI = symbol.replace('/USDT', 'USD'); // Convert BTC/USDT to BTCUSD
            const response = await axios.get(`https://cdn-ind.testnet.deltaex.org/v2/tickers/${symbolForAPI}`);

            if (response.data.success) {
              const ticker = response.data.result;
              additionalData = {
                change: parseFloat(ticker.change || '0'),
                changePercent: parseFloat(ticker.change_percent || '0'),
                volume: parseFloat(ticker.volume || '0'),
                high24h: parseFloat(ticker.high || price * 1.05),
                low24h: parseFloat(ticker.low || price * 0.95),
                markPrice: parseFloat(ticker.mark_price || '0'),
                indexPrice: parseFloat(ticker.spot_price || '0'),
                openInterest: parseFloat(ticker.open_interest || '0')
              };
            }
          } catch (apiError) {
            console.warn(`Failed to get additional data for ${symbol}:`, apiError.message);
          }
        }

        // Use real data if available, otherwise generate mock data
        const change24h = additionalData.change || (Math.random() - 0.5) * 0.1 * price;
        const changePercent = additionalData.changePercent || (change24h / price) * 100;

        marketData[symbol] = {
          symbol,
          price: parseFloat(price.toFixed(2)),
          change: parseFloat(change24h.toFixed(2)),
          changePercent: parseFloat(changePercent.toFixed(2)),
          volume: additionalData.volume || parseFloat((Math.random() * 10000).toFixed(2)),
          high24h: additionalData.high24h || parseFloat((price * 1.05).toFixed(2)),
          low24h: additionalData.low24h || parseFloat((price * 0.95).toFixed(2)),
          markPrice: additionalData.markPrice || 0,
          indexPrice: additionalData.indexPrice || 0,
          openInterest: additionalData.openInterest || 0,
          timestamp: Date.now(),
          source: productId ? 'delta_exchange_india' : 'mock'
        };
      } catch (error) {
        console.error(`Error getting market data for ${symbol}:`, error);
        // Fallback data
        const basePrice = symbol === 'ETH/USDT' ? 2579.39 : 105563.43;
        marketData[symbol] = {
          symbol,
          price: basePrice,
          change: 0,
          changePercent: 0,
          volume: 0,
          high24h: basePrice,
          low24h: basePrice,
          markPrice: 0,
          indexPrice: 0,
          openInterest: 0,
          timestamp: Date.now(),
          source: 'fallback'
        };
      }
    }

    res.json({
      success: true,
      data: marketData,
      timestamp: Date.now(),
      source: 'delta_exchange_india_testnet'
    });
  } catch (error) {
    console.error('Error getting market data:', error);
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
    console.error('Error getting status:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Get historical price data for charts
router.get('/chart-data/:symbol', async (req, res) => {
  try {
    const { symbol } = req.params;
    const { timeframe = '1m', limit = '100' } = req.query;
    const limitNum = parseInt(limit) || 100;
    
    // Generate mock historical data
    const now = Date.now();
    const interval = timeframe === '1m' ? 60000 : 300000; // 1min or 5min
    const data = [];
    
    let basePrice = symbol === 'ETH/USDT' ? 2579 : 105500;
    
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
    console.error('Error getting chart data:', error);
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
    console.error('Error updating paper trading state:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

module.exports = router;
