/**
 * Market Controller
 * Handles market data and price information
 */

const prisma = require('../utils/prismaClient');
const { createError } = require('../middleware/errorHandler');

/**
 * Get real-time price data
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @param {Function} next - Express next function
 */
const getRealTimePrice = async (req, res, next) => {
  try {
    const { symbol = 'BTCUSD' } = req.query;
    
    // TODO: In production, fetch this from external market data API
    // For now, generate realistic mock data
    const basePrice = getBasePrice(symbol);
    const now = Date.now();
    
    const mockPrice = {
      symbol,
      price: basePrice + (Math.random() * 100 - 50),
      timestamp: new Date(now).toISOString(),
      change: {
        '1h': (Math.random() * 2 - 1).toFixed(2),
        '24h': (Math.random() * 5 - 2.5).toFixed(2),
        '7d': (Math.random() * 15 - 7.5).toFixed(2)
      },
      volume: {
        '24h': Math.floor(Math.random() * 500000000 + 1000000000)
      }
    };
    
    res.json({
      success: true,
      data: mockPrice
    });
  } catch (error) {
    next(createError(`Failed to fetch price data: ${error.message}`, 500));
  }
};

/**
 * Get historical OHLCV data
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @param {Function} next - Express next function
 */
const getHistoricalData = async (req, res, next) => {
  try {
    const { 
      symbol = 'BTCUSD', 
      interval = '1h', 
      limit = 100,
      start = Date.now() - (100 * 3600000) // Default to 100 hours ago for hourly data
    } = req.query;
    
    // TODO: In production, fetch this from database or external API
    // For now, generate realistic mock data
    const basePrice = getBasePrice(symbol);
    const intervalMs = getIntervalMs(interval);
    const mockData = [];
    
    // Generate candles with somewhat realistic price movements
    let lastClose = basePrice;
    for (let i = 0; i < limit; i++) {
      const timestamp = new Date(parseInt(start) + (i * intervalMs));
      
      // Create some randomness but maintain a trend
      const volatility = getVolatility(symbol, interval);
      const changePercent = (Math.random() * volatility * 2) - volatility;
      const change = lastClose * (changePercent / 100);
      
      // Calculate OHLCV values
      const close = lastClose + change;
      const open = lastClose;
      const high = Math.max(open, close) + (Math.random() * Math.abs(change));
      const low = Math.min(open, close) - (Math.random() * Math.abs(change));
      const volume = Math.floor(Math.random() * getBaseVolume(symbol, interval));
      
      mockData.push({
        timestamp: timestamp.toISOString(),
        open,
        high,
        low,
        close,
        volume
      });
      
      lastClose = close;
    }
    
    res.json({
      success: true,
      data: {
        symbol,
        interval,
        candles: mockData
      }
    });
  } catch (error) {
    next(createError(`Failed to fetch historical data: ${error.message}`, 500));
  }
};

/**
 * Get market pairs data (available trading pairs)
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @param {Function} next - Express next function
 */
const getMarketPairs = async (req, res, next) => {
  try {
    // Mock market pairs data
    const mockPairs = [
      {
        symbol: 'BTCUSD',
        baseAsset: 'BTC',
        quoteAsset: 'USD',
        price: 46250.75,
        change24h: 2.35,
        volume24h: 1250000000,
        marketCap: 889753000000
      },
      {
        symbol: 'ETHUSD',
        baseAsset: 'ETH',
        quoteAsset: 'USD',
        price: 3456.25,
        change24h: 1.75,
        volume24h: 850000000,
        marketCap: 415240000000
      },
      {
        symbol: 'LTCUSD',
        baseAsset: 'LTC',
        quoteAsset: 'USD',
        price: 185.60,
        change24h: -0.85,
        volume24h: 125000000,
        marketCap: 13250000000
      },
      {
        symbol: 'XRPUSD',
        baseAsset: 'XRP',
        quoteAsset: 'USD',
        price: 0.65,
        change24h: 0.35,
        volume24h: 75000000,
        marketCap: 31560000000
      }
    ];
    
    res.json({
      success: true,
      data: mockPairs
    });
  } catch (error) {
    next(createError(`Failed to fetch market pairs: ${error.message}`, 500));
  }
};

// Helper functions to generate realistic market data
function getBasePrice(symbol) {
  const prices = {
    'BTCUSD': 46000,
    'ETHUSD': 3450,
    'LTCUSD': 185,
    'XRPUSD': 0.65,
    'BNBUSD': 550,
    'ADAUSD': 1.25,
    'DOGEUSD': 0.12,
    'SOLUSD': 110
  };
  
  return prices[symbol] || 100; // Default to 100 for unknown symbols
}

function getVolatility(symbol, interval) {
  // Base volatility percentages by interval
  const baseVolatility = {
    '1m': 0.15,
    '5m': 0.3,
    '15m': 0.5,
    '30m': 0.8,
    '1h': 1.2,
    '4h': 2.5,
    '1d': 4.0,
    '1w': 10.0
  };
  
  // Adjust for asset (some are more volatile)
  const assetMultiplier = {
    'BTCUSD': 1.0,
    'ETHUSD': 1.2,
    'LTCUSD': 1.3,
    'XRPUSD': 1.5,
    'DOGEUSD': 2.0,
    'SOLUSD': 1.6
  };
  
  const baseVol = baseVolatility[interval] || 1.0;
  const multiplier = assetMultiplier[symbol] || 1.0;
  
  return baseVol * multiplier;
}

function getIntervalMs(interval) {
  const intervals = {
    '1m': 60 * 1000,
    '5m': 5 * 60 * 1000,
    '15m': 15 * 60 * 1000,
    '30m': 30 * 60 * 1000,
    '1h': 60 * 60 * 1000,
    '4h': 4 * 60 * 60 * 1000,
    '1d': 24 * 60 * 60 * 1000,
    '1w': 7 * 24 * 60 * 60 * 1000
  };
  
  return intervals[interval] || 60 * 60 * 1000; // Default to 1h
}

function getBaseVolume(symbol, interval) {
  // Base volume by symbol
  const baseVolume = {
    'BTCUSD': 1000000,
    'ETHUSD': 800000,
    'LTCUSD': 200000,
    'XRPUSD': 300000,
    'DOGEUSD': 500000,
    'SOLUSD': 400000
  };
  
  // Adjust volume by timeframe
  const intervalMultiplier = {
    '1m': 0.01,
    '5m': 0.05,
    '15m': 0.15,
    '30m': 0.3,
    '1h': 1.0,
    '4h': 4.0,
    '1d': 24.0,
    '1w': 168.0
  };
  
  const base = baseVolume[symbol] || 500000;
  const multiplier = intervalMultiplier[interval] || 1.0;
  
  return base * multiplier;
}

module.exports = {
  getRealTimePrice,
  getHistoricalData,
  getMarketPairs
}; 