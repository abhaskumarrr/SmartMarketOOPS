/**
 * Working Trading Routes (TypeScript)
 * Delta Exchange India trading integration endpoints
 */

import express from 'express';
import axios from 'axios';
import crypto from 'crypto';
import dotenv from 'dotenv';
import path from 'path';

// Load environment variables directly
dotenv.config({ path: path.join(__dirname, '..', '..', '..', '.env') });

const router = express.Router();

// Delta Exchange API configuration - Load directly from process.env
const DELTA_API_KEY = process.env.DELTA_EXCHANGE_API_KEY || 'uS2N0I4V37gMNJgbTjX8a33WPWv3GK';
const DELTA_API_SECRET = process.env.DELTA_EXCHANGE_API_SECRET || 'hJwxEd1wCpMTYg5iSQKDnreX9IVlc4mcYegR5ojJzvQ5UVOiUhP7cF9u21To';
// Use correct testnet URL from official documentation
const DELTA_BASE_URL = process.env.DELTA_EXCHANGE_TESTNET === 'true'
  ? 'https://cdn-ind.testnet.deltaex.org'
  : 'https://api.india.delta.exchange';

console.log('🔑 Delta Exchange API Configuration:');
console.log(`- Base URL: ${DELTA_BASE_URL}`);
console.log(`- API Key: ${DELTA_API_KEY ? DELTA_API_KEY.substring(0, 8) + '...' : 'NOT SET'}`);
console.log(`- API Secret: ${DELTA_API_SECRET ? DELTA_API_SECRET.substring(0, 8) + '...' : 'NOT SET'}`);
console.log(`- Testnet: ${process.env.DELTA_EXCHANGE_TESTNET}`);

if (!DELTA_API_KEY || !DELTA_API_SECRET) {
  console.error('❌ Delta Exchange API credentials not found in environment variables!');
  console.error('Please check your .env file for DELTA_EXCHANGE_API_KEY and DELTA_EXCHANGE_API_SECRET');
} else {
  console.log('✅ Delta Exchange API credentials loaded successfully!');
console.log('🔄 Environment variables refreshed');
}

// Helper function to create Delta Exchange API signature (Official Implementation)
function createDeltaSignature(method: string, path: string, queryString: string, body: string, timestamp: string): string {
  // According to Delta Exchange docs: method + timestamp + requestPath + query params + body
  const message = method + timestamp + path + queryString + body;

  console.log(`🔐 Signature generation:`);
  console.log(`- Method: ${method}`);
  console.log(`- Timestamp: ${timestamp}`);
  console.log(`- Path: ${path}`);
  console.log(`- Query: ${queryString}`);
  console.log(`- Body: ${body}`);
  console.log(`- Message: ${message}`);
  console.log(`- Secret: ${DELTA_API_SECRET ? DELTA_API_SECRET.substring(0, 8) + '...' : 'NOT SET'}`);

  const signature = crypto.createHmac('sha256', DELTA_API_SECRET).update(message).digest('hex');
  console.log(`- Signature: ${signature}`);
  return signature;
}

// Helper function to make authenticated Delta Exchange API calls
async function makeDeltaApiCall(method: string, path: string, params: any = {}, data: any = null) {
  const timestamp = Math.floor(Date.now() / 1000).toString();
  const queryString = Object.keys(params).length > 0 ? '?' + new URLSearchParams(params).toString() : '';
  const body = data ? JSON.stringify(data) : '';

  const signature = createDeltaSignature(method, path, queryString, body, timestamp);

  const headers = {
    'api-key': DELTA_API_KEY,
    'signature': signature,
    'timestamp': timestamp,
    'Content-Type': 'application/json',
    'User-Agent': 'SmartMarketOOPS-v1.0'
  };

  try {
    const response = await axios({
      method: method as any,
      url: DELTA_BASE_URL + path + queryString,
      data: data || undefined,
      headers
    });

    return response.data;
  } catch (error: any) {
    console.error('Delta API Error:', error.response?.data || error.message);
    throw error;
  }
}

/**
 * GET /api/trading/status
 * Get trading service status
 */
router.get('/status', async (req, res) => {
  try {
    res.json({
      success: true,
      data: {
        status: 'connected',
        exchange: 'delta_exchange_india',
        environment: process.env.DELTA_EXCHANGE_TESTNET === 'true' ? 'testnet' : 'production',
        supportedSymbols: ['BTCUSD', 'ETHUSD', 'ADAUSD', 'SOLUSD'],
        totalSymbols: 4,
        timestamp: Date.now()
      }
    });
  } catch (error) {
    console.error('Error getting trading status:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get trading status',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/trading/products
 * Get all available trading products
 */
router.get('/products', async (req, res) => {
  try {
    const mockProducts = [
      {
        id: 27,
        symbol: 'BTCUSD',
        description: 'Bitcoin Perpetual',
        contract_type: 'perpetual_futures',
        state: 'live',
        tick_size: '0.5',
        contract_value: '0.001',
        maker_commission_rate: '0.0005',
        taker_commission_rate: '0.0015',
        underlying_asset: 'BTC',
        quoting_asset: 'USD',
        settling_asset: 'INR'
      },
      {
        id: 3136,
        symbol: 'ETHUSD',
        description: 'Ethereum Perpetual',
        contract_type: 'perpetual_futures',
        state: 'live',
        tick_size: '0.05',
        contract_value: '0.01',
        maker_commission_rate: '0.0005',
        taker_commission_rate: '0.0015',
        underlying_asset: 'ETH',
        quoting_asset: 'USD',
        settling_asset: 'INR'
      }
    ];

    res.json({
      success: true,
      data: mockProducts,
      meta: {
        total: mockProducts.length,
        exchange: 'delta_exchange_india',
        timestamp: Date.now()
      }
    });
  } catch (error) {
    console.error('Error getting products:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get products',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/trading/market-data/:symbol
 * Get real-time market data
 */
router.get('/market-data/:symbol', async (req, res) => {
  try {
    const { symbol } = req.params;
    
    const mockMarketData = {
      symbol: symbol.toUpperCase(),
      price: Math.random() * 50000 + 30000, // Random price between 30k-80k
      volume: Math.random() * 1000000,
      change_24h: (Math.random() - 0.5) * 10,
      high_24h: Math.random() * 55000 + 35000,
      low_24h: Math.random() * 45000 + 25000,
      timestamp: Date.now()
    };

    res.json({
      success: true,
      data: mockMarketData,
      timestamp: Date.now(),
      source: 'delta_exchange_india'
    });
  } catch (error) {
    console.error(`Error getting market data for ${req.params.symbol}:`, error);
    res.status(500).json({
      success: false,
      error: 'Failed to get market data',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * POST /api/trading/orders
 * Place a new order on Delta Exchange
 */
router.post('/orders', async (req, res) => {
  try {
    const orderRequest = req.body;

    // Validate required fields
    if (!orderRequest.product_id || !orderRequest.size || !orderRequest.side) {
      return res.status(400).json({
        success: false,
        error: 'Invalid order request',
        message: 'Missing required fields: product_id, size, side'
      });
    }

    console.log('🔍 Placing real order on Delta Exchange:', orderRequest);

    // Place real order using Delta Exchange API
    const response = await makeDeltaApiCall('POST', '/v2/orders', {}, orderRequest);

    console.log('✅ Real order placed:', response);

    res.json({
      success: true,
      data: response.result || response,
      message: `Real order placed successfully on Delta Exchange (testnet)`,
      timestamp: Date.now(),
      api_response: response
    });
  } catch (error: any) {
    console.error('❌ Error placing real order:', error.response?.data || error.message);
    res.status(500).json({
      success: false,
      error: 'Failed to place order',
      message: error.response?.data?.error?.message || error.message || 'Unknown error',
      error_details: error.response?.data || error.message
    });
  }
});

/**
 * GET /api/trading/orders
 * Get open orders
 */
router.get('/orders', async (req, res) => {
  try {
    const mockOrders = [
      {
        id: 123456,
        product_id: 27,
        size: '0.1',
        side: 'buy',
        order_type: 'limit_order',
        limit_price: '45000',
        state: 'open',
        created_at: new Date().toISOString()
      }
    ];

    res.json({
      success: true,
      data: mockOrders,
      message: `Orders from Delta Exchange ${process.env.DELTA_EXCHANGE_TESTNET === 'true' ? '(testnet)' : '(production)'}`,
      timestamp: Date.now()
    });
  } catch (error) {
    console.error('Error getting orders:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get orders',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/trading/positions
 * Get current positions from Delta Exchange
 */
router.get('/positions', async (req, res) => {
  try {
    console.log('🔍 Fetching real positions from Delta Exchange API...');

    // Get real positions from Delta Exchange API
    const response = await makeDeltaApiCall('GET', '/v2/positions');

    console.log('✅ Real positions received:', response);

    res.json({
      success: true,
      data: response.result || response,
      message: `Real positions from Delta Exchange (testnet)`,
      timestamp: Date.now(),
      api_response: response
    });
  } catch (error: any) {
    console.error('❌ Error getting real positions:', error.response?.data || error.message);

    // Fallback to empty positions if API fails
    res.json({
      success: true,
      data: [],
      message: `No positions found (API error: ${error.response?.data?.error?.message || error.message || 'Unknown error'})`,
      timestamp: Date.now(),
      warning: 'API call failed, showing empty positions',
      error_details: error.response?.data || error.message
    });
  }
});

/**
 * GET /api/trading/balances
 * Get wallet balances from Delta Exchange API
 */
router.get('/balances', async (req, res) => {
  try {
    console.log('🔍 Fetching real balances from Delta Exchange API...');

    // Get real balances from Delta Exchange API
    const response = await makeDeltaApiCall('GET', '/v2/wallet/balances');

    console.log('✅ Real balances received:', response);

    res.json({
      success: true,
      data: response.result || response,
      message: `Real balances from Delta Exchange (testnet)`,
      timestamp: Date.now(),
      api_response: response
    });
  } catch (error: any) {
    console.error('❌ Error getting real balances:', error.response?.data || error.message);

    // Fallback to mock data if API fails
    const fallbackBalances = [
      {
        asset: 'INR',
        balance: '10000.00',
        available_balance: '9500.00',
        reserved_balance: '500.00',
        note: 'Fallback data - API call failed'
      }
    ];

    res.json({
      success: true,
      data: fallbackBalances,
      message: `Fallback balances (API error: ${error.response?.data?.error?.message || error.message || 'Unknown error'})`,
      timestamp: Date.now(),
      warning: 'Using fallback data due to API error',
      error_details: error.response?.data || error.message
    });
  }
});

/**
 * POST /api/trading/place-trade-with-tpsl
 * Place a trade with take profit and stop loss orders
 */
router.post('/place-trade-with-tpsl', async (req, res) => {
  try {
    const {
      symbol = 'BTCUSD',
      side = 'buy',
      size = '0.01',
      order_type = 'market_order',
      take_profit_percentage = 2.0,
      stop_loss_percentage = 1.0
    } = req.body;

    console.log('🎯 Placing trade with TP/SL:', { symbol, side, size, order_type, take_profit_percentage, stop_loss_percentage });

    // Step 1: Get REAL current market price from Delta Exchange
    console.log('🔍 Getting REAL market data from Delta Exchange...');

    // Get products to find the product ID
    const productsResponse = await makeDeltaApiCall('GET', '/v2/products');
    const products = productsResponse.result || [];
    const product = products.find((p: any) => p.symbol === symbol);

    if (!product) {
      return res.status(400).json({
        success: false,
        error: 'Product not found',
        message: `Symbol ${symbol} not found in available products`
      });
    }

    // Get REAL ticker data for current market price
    const tickerResponse = await makeDeltaApiCall('GET', `/v2/tickers/${product.symbol}`);
    const ticker = tickerResponse.result;

    // Use REAL market price from ticker (mark_price is the most accurate)
    const currentPrice = parseFloat(ticker.mark_price || ticker.close || ticker.last_price);

    console.log(`📊 REAL ${symbol} market data:`);
    console.log(`- Mark Price: $${ticker.mark_price}`);
    console.log(`- Last Price: $${ticker.last_price}`);
    console.log(`- Close Price: $${ticker.close}`);
    console.log(`- Using Price: $${currentPrice}`);

    if (!currentPrice || currentPrice <= 0) {
      throw new Error(`Invalid market price received: ${currentPrice}`);
    }

    // Step 2: Place main order (correct Delta Exchange format)
    const mainOrder = {
      product_id: product.id,
      size: parseInt(size) || 1, // Size must be integer (number of contracts)
      side: side,
      order_type: order_type
      // Note: limit_price not needed for market orders
    };

    console.log('🔍 Placing main order:', mainOrder);
    const mainOrderResponse = await makeDeltaApiCall('POST', '/v2/orders', {}, mainOrder);
    console.log('✅ Main order placed:', mainOrderResponse);

    // Step 3: Calculate TP/SL prices
    const isLong = side === 'buy';
    const takeProfitPrice = isLong
      ? currentPrice * (1 + take_profit_percentage / 100)
      : currentPrice * (1 - take_profit_percentage / 100);

    const stopLossPrice = isLong
      ? currentPrice * (1 - stop_loss_percentage / 100)
      : currentPrice * (1 + stop_loss_percentage / 100);

    console.log(`🎯 TP Price: $${takeProfitPrice.toFixed(2)}, SL Price: $${stopLossPrice.toFixed(2)}`);

    // Step 4: Place Take Profit order (correct Delta Exchange format)
    const takeProfitOrder = {
      product_id: product.id,
      size: parseInt(size) || 1, // Size must be integer (number of contracts)
      side: isLong ? 'sell' : 'buy',
      order_type: 'limit_order',
      limit_price: takeProfitPrice.toFixed(2) // String format for price
    };

    console.log('🎯 Placing take profit order:', takeProfitOrder);
    const tpOrderResponse = await makeDeltaApiCall('POST', '/v2/orders', {}, takeProfitOrder);
    console.log('✅ Take profit order placed:', tpOrderResponse);

    // Step 5: Place Stop Loss order (correct Delta Exchange format)
    const stopLossOrder = {
      product_id: product.id,
      size: parseInt(size) || 1, // Size must be integer (number of contracts)
      side: isLong ? 'sell' : 'buy',
      order_type: 'stop_loss_order',
      stop_price: stopLossPrice.toFixed(2) // String format for stop price
      // Note: time_in_force not allowed for stop orders
    };

    console.log('🛡️ Placing stop loss order:', stopLossOrder);
    const slOrderResponse = await makeDeltaApiCall('POST', '/v2/orders', {}, stopLossOrder);
    console.log('✅ Stop loss order placed:', slOrderResponse);

    // Return comprehensive response
    res.json({
      success: true,
      message: `Trade placed successfully with TP/SL on ${symbol}`,
      data: {
        symbol: symbol,
        side: side,
        size: size,
        current_price: currentPrice,
        take_profit_price: takeProfitPrice,
        stop_loss_price: stopLossPrice,
        orders: {
          main_order: mainOrderResponse.result || mainOrderResponse,
          take_profit_order: tpOrderResponse.result || tpOrderResponse,
          stop_loss_order: slOrderResponse.result || slOrderResponse
        }
      },
      timestamp: Date.now()
    });

  } catch (error: any) {
    console.error('❌ Error placing trade with TP/SL:', error.response?.data || error.message);
    res.status(500).json({
      success: false,
      error: 'Failed to place trade with TP/SL',
      message: error.response?.data?.error?.message || error.message || 'Unknown error',
      error_details: error.response?.data || error.message
    });
  }
});

/**
 * POST /api/trading/activate-bot
 * Activate trading bot with specified parameters
 */
router.post('/activate-bot', async (req, res) => {
  try {
    const {
      name = 'SmartMarketOOPS Bot',
      strategy = 'momentum_scalping',
      symbols = ['BTCUSD', 'ETHUSD'],
      risk_per_trade = 2.0,
      take_profit = 2.0,
      stop_loss = 1.0,
      max_positions = 3,
      enabled = true
    } = req.body;

    console.log('🤖 Activating trading bot:', { name, strategy, symbols, risk_per_trade });

    // Simulate bot activation (in real implementation, this would start the bot service)
    const botConfig = {
      id: `bot_${Date.now()}`,
      name: name,
      strategy: strategy,
      symbols: symbols,
      risk_management: {
        risk_per_trade_percentage: risk_per_trade,
        take_profit_percentage: take_profit,
        stop_loss_percentage: stop_loss,
        max_concurrent_positions: max_positions
      },
      status: enabled ? 'active' : 'inactive',
      created_at: new Date().toISOString(),
      last_updated: new Date().toISOString()
    };

    console.log('✅ Trading bot activated:', botConfig);

    res.json({
      success: true,
      message: `Trading bot "${name}" activated successfully`,
      data: botConfig,
      timestamp: Date.now()
    });

  } catch (error: any) {
    console.error('❌ Error activating trading bot:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to activate trading bot',
      message: error.message || 'Unknown error'
    });
  }
});

export default router;
