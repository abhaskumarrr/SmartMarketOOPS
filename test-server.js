#!/usr/bin/env node

const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = 3001;

// Middleware
app.use(cors());
app.use(express.json());

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
      const symbolForAPI = symbol.replace('/USDT', 'USD'); // Convert BTC/USDT to BTCUSD
      const response = await axios.get(`https://cdn-ind.testnet.deltaex.org/v2/tickers/${symbolForAPI}`);
      
      if (response.data.success) {
        const ticker = response.data.result;
        return parseFloat(ticker.close || ticker.last_price || '0');
      }
    }
    
    // Return mock prices if API fails
    return symbol.includes('ETH') ? 2579.39 : 105563.43;
  } catch (error) {
    console.error(`Error fetching price for ${symbol}:`, error.message);
    // Return mock prices if API fails
    return symbol.includes('ETH') ? 2579.39 : 105563.43;
  }
}

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    service: 'SmartMarketOOPS Test Server',
    version: '1.0.0'
  });
});

// Get real-time market data
app.get('/api/paper-trading/market-data', async (req, res) => {
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

// Mock portfolio endpoint
app.get('/api/paper-trading/portfolio', (req, res) => {
  res.json({
    success: true,
    data: {
      balance: 10000.00,
      totalValue: 10000.00,
      totalPnL: 0.00,
      positions: [],
      timestamp: Date.now()
    }
  });
});

// Test Delta Exchange API directly
app.get('/api/test/delta-api', async (req, res) => {
  try {
    console.log('🧪 Testing Delta Exchange API...');
    
    const results = {
      products: null,
      btcTicker: null,
      ethTicker: null,
      errors: []
    };
    
    // Test 1: Get products
    try {
      const response = await axios.get('https://cdn-ind.testnet.deltaex.org/v2/products');
      if (response.data.success) {
        const products = response.data.result;
        const btcProduct = products.find(p => p.symbol === 'BTCUSD');
        const ethProduct = products.find(p => p.symbol === 'ETHUSD');
        
        results.products = {
          total: products.length,
          btc: btcProduct ? { id: btcProduct.id, symbol: btcProduct.symbol } : null,
          eth: ethProduct ? { id: ethProduct.id, symbol: ethProduct.symbol } : null
        };
      }
    } catch (error) {
      results.errors.push(`Products API: ${error.message}`);
    }
    
    // Test 2: Get BTC ticker
    try {
      const response = await axios.get('https://cdn-ind.testnet.deltaex.org/v2/tickers/BTCUSD');
      if (response.data.success) {
        const ticker = response.data.result;
        results.btcTicker = {
          price: ticker.close || ticker.last_price,
          volume: ticker.volume,
          high: ticker.high,
          low: ticker.low,
          markPrice: ticker.mark_price,
          indexPrice: ticker.spot_price
        };
      }
    } catch (error) {
      results.errors.push(`BTC Ticker: ${error.message}`);
    }
    
    // Test 3: Get ETH ticker
    try {
      const response = await axios.get('https://cdn-ind.testnet.deltaex.org/v2/tickers/ETHUSD');
      if (response.data.success) {
        const ticker = response.data.result;
        results.ethTicker = {
          price: ticker.close || ticker.last_price,
          volume: ticker.volume,
          high: ticker.high,
          low: ticker.low,
          markPrice: ticker.mark_price,
          indexPrice: ticker.spot_price
        };
      }
    } catch (error) {
      results.errors.push(`ETH Ticker: ${error.message}`);
    }
    
    res.json({
      success: true,
      data: results,
      timestamp: Date.now()
    });
  } catch (error) {
    console.error('Error testing Delta API:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Test Server running on http://localhost:${PORT}`);
  console.log(`📊 Market Data: http://localhost:${PORT}/api/paper-trading/market-data`);
  console.log(`🧪 Delta API Test: http://localhost:${PORT}/api/test/delta-api`);
  console.log(`🏥 Health Check: http://localhost:${PORT}/health`);
});

module.exports = app;
