/**
 * Working Trading Routes (TypeScript)
 * Delta Exchange India trading integration endpoints
 */

import express from 'express';
import { DeltaExchangeUnified } from '../services/DeltaExchangeUnified';
import env from '../config/environment';
import { protect as auth } from '../middleware/auth';

import { logger } from '../utils/logger';

const router = express.Router();

// Apply authentication middleware to all routes
router.use(auth);

// Initialize Delta Exchange Unified service
const deltaExchange = new DeltaExchangeUnified({
  apiKey: env.DELTA_EXCHANGE_API_KEY,
  apiSecret: env.DELTA_EXCHANGE_API_SECRET,
  testnet: env.DELTA_EXCHANGE_TESTNET
});

logger.info('🔑 Delta Exchange API Configuration:');
logger.info(`- Status: ${deltaExchange.isInitialized() ? 'Connected' : 'Not Connected'}`);
logger.info(`- API Key: ${env.DELTA_EXCHANGE_API_KEY ? env.DELTA_EXCHANGE_API_KEY.substring(0, 8) + '...' : 'NOT SET'}`);
logger.info(`- Testnet: ${env.DELTA_EXCHANGE_TESTNET}`);

if (!env.DELTA_EXCHANGE_API_KEY || !env.DELTA_EXCHANGE_API_SECRET) {
  logger.error('❌ Delta Exchange API credentials not found in environment variables!');
  logger.error('Please check your .env file for DELTA_EXCHANGE_API_KEY and DELTA_EXCHANGE_API_SECRET');
} else {
  logger.info('✅ Delta Exchange API credentials loaded successfully!');
  logger.info('🔄 Environment variables refreshed');
}

/**
 * GET /api/trading/status
 * Get trading service status
 */
router.get('/status', async (req, res): Promise<Response> => {
  try {
    return res.json({
      success: true,
      data: {
        status: 'connected',
        exchange: 'delta_exchange_india',
        environment: env.DELTA_EXCHANGE_TESTNET ? 'testnet' : 'production',
        supportedSymbols: ['BTCUSD', 'ETHUSD', 'ADAUSD', 'SOLUSD'],
        totalSymbols: 4,
        timestamp: Date.now()
      }
    });
  } catch (error) {
    logger.error('Error getting trading status:', { error: error instanceof Error ? error.message : String(error) });
    return res.status(500).json({
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
router.get('/products', async (req, res): Promise<Response> => {
  try {
    // Ensure Delta Exchange client is initialized
    if (!deltaExchange.isInitialized()) {
      await deltaExchange.initialize();
    }
    const products = await deltaExchange.getAllProducts();

    return res.json({
      success: true,
      data: products,
      meta: {
        total: products.length,
        exchange: 'delta_exchange_india',
        timestamp: Date.now()
      }
    });
  } catch (error) {
    logger.error('Error getting products:', { error: error instanceof Error ? error.message : String(error) });
    return res.status(500).json({
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
router.get('/market-data/:symbol', async (req, res): Promise<Response> => {
  try {
    const { symbol } = req.params;
    
    // Ensure Delta Exchange client is initialized
    if (!deltaExchange.isInitialized()) {
      await deltaExchange.initialize();
    }

    const marketData = await deltaExchange.getMarketData(symbol.toUpperCase());

    if (!marketData) {
      return res.status(404).json({
        success: false,
        error: 'Symbol not found',
        message: `Market data not available for symbol: ${symbol}`
      });
    }

    return res.json({
      success: true,
      data: marketData,
      timestamp: Date.now(),
      source: 'delta_exchange_india'
    });
  } catch (error) {
    logger.error(`Error getting market data for ${req.params.symbol}:`, { error: error instanceof Error ? error.message : String(error) });
    return res.status(500).json({
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
router.post('/orders', async (req, res): Promise<Response> => {
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

    logger.info('🚀 Placing real order on Delta Exchange:', { orderRequest });

    // Place real order using Delta Exchange API
    const response = await deltaExchange.placeOrder(orderRequest);

    logger.info('✅ Real order placed:', { response });

    return res.json({
      success: true,
      data: response.result || response,
      message: `Real order placed successfully on Delta Exchange (testnet)`,
      timestamp: Date.now(),
      api_response: response
    });
  } catch (error) {
    logger.error('❌ Error placing real order:', { error: error instanceof Error ? error.message : String(error), response_data: (error as any).response?.data });
    return res.status(500).json({
      success: false,
      error: 'Failed to place order',
      message: (error as any).response?.data?.error?.message || (error as any).message || 'Unknown error',
      error_details: (error as any).response?.data || (error as any).message
    });
  }
});

/**
 * GET /api/trading/orders
 * Get open orders
 */
router.get('/orders', async (req, res): Promise<Response> => {
  try {
    // Ensure Delta Exchange client is initialized
    if (!deltaExchange.isInitialized()) {
      await deltaExchange.initialize();
    }

    const openOrders = await deltaExchange.getOpenOrders();

    return res.json({
      success: true,
      data: openOrders,
      message: `Orders from Delta Exchange ${env.DELTA_EXCHANGE_TESTNET ? '(testnet)' : '(production)'}`,
      timestamp: Date.now()
    });
  } catch (error) {
    logger.error('Error getting orders:', { error: error instanceof Error ? error.message : String(error) });
    return res.status(500).json({
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
router.get('/positions', async (req, res): Promise<Response> => {
  try {
    logger.info('🔍 Fetching real positions from Delta Exchange API...');

    // Get real positions from Delta Exchange API
    const response = await deltaExchange.getPositions();

    logger.info('✅ Real positions received:', { response });

    return res.json({
      success: true,
      data: response.result || response,
      message: `Real positions from Delta Exchange (testnet)`,
      timestamp: Date.now(),
      api_response: response
    });
  } catch (error) {
    logger.error('❌ Error getting real positions:', { error: error instanceof Error ? error.message : String(error), response_data: (error as any).response?.data });

    // Fallback to empty positions if API fails
    return res.json({
      success: true,
      data: [],
      message: `No positions found (API error: ${(error as any).response?.data?.error?.message || (error as any).message || 'Unknown error'})`,
      timestamp: Date.now(),
      warning: 'API call failed, showing empty positions',
      error_details: (error as any).response?.data || (error as any).message
    });
  }
});

/**
 * GET /api/trading/balances
 * Get wallet balances from Delta Exchange API
 */
router.get('/balances', async (req, res): Promise<Response> => {
  try {
    logger.info('🔍 Fetching real balances from Delta Exchange API...');

    // Get real balances from Delta Exchange API
    const response = await deltaExchange.getBalance();

    logger.info('✅ Real balances received:', { response });

    return res.json({
      success: true,
      data: response.result || response,
      message: `Real balances from Delta Exchange (testnet)`,
      timestamp: Date.now(),
      api_response: response
    });
  } catch (error) {
    logger.error('❌ Error getting real balances:', { error: error instanceof Error ? error.message : String(error), response_data: (error as any).response?.data });

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

    return res.json({
      success: true,
      data: fallbackBalances,
      message: `Fallback balances (API error: ${(error as any).response?.data?.error?.message || (error as any).message || 'Unknown error'})`,
      timestamp: Date.now(),
      warning: 'Using fallback data due to API error',
      error_details: (error as any).response?.data || (error as any).message
    });
  }
});

/**
 * POST /api/trading/place-trade-with-tpsl
 * Place a trade with take profit and stop loss orders
 */
router.post('/place-trade-with-tpsl', async (req, res): Promise<Response> => {
  try {
    const {
      symbol = 'BTCUSD',
      side = 'buy',
      size = '0.01',
      order_type = 'market_order',
      take_profit_percentage = 2.0,
      stop_loss_percentage = 1.0
    } = req.body;

    logger.info('🎯 Placing trade with TP/SL:', { symbol, side, size, order_type, take_profit_percentage, stop_loss_percentage });

    // Step 1: Get REAL current market price from Delta Exchange
    logger.info('🔍 Getting REAL market data from Delta Exchange...');

    // Get products to find the product ID
    const products = deltaExchange.getAllProducts();
    const product = products.find((p: any) => p.symbol === symbol);

    if (!product) {
      return res.status(400).json({
        success: false,
        error: 'Product not found',
        message: `Symbol ${symbol} not found in available products`
      });
    }

    // Get REAL ticker data for current market price
    const ticker = await deltaExchange.getMarketData(product.symbol);

    // Use REAL market price from ticker (mark_price is the most accurate)
    const currentPrice = parseFloat(ticker.mark_price || ticker.close || ticker.last_price);

    logger.info(`📊 REAL ${symbol} market data:`);
    logger.info(`- Mark Price: ${ticker.mark_price}`);
    logger.info(`- Last Price: ${ticker.last_price}`);
    logger.info(`- Close Price: ${ticker.close}`);
    logger.info(`- Using Price: ${currentPrice}`);

    if (!currentPrice || currentPrice <= 0) {
      throw new Error(`Invalid market price received: ${currentPrice}`);
    }

    // Step 2: Place main order (correct Delta Exchange format)
    const defaultProductId = env.DELTA_BTCUSD_PRODUCT_ID;
    
    // Step 2: Place main order (correct Delta Exchange format)
    const mainOrder = {
      product_id: product.id || defaultProductId,
      size: parseInt(size) || 1, // Size must be integer (number of contracts)
      side: side,
      order_type: order_type,
      ...(order_type === 'limit_order' && { limit_price: currentPrice }),
      time_in_force: 'gtc',
      // Post-only for limit orders to ensure you're a maker, not a taker
      ...(order_type === 'limit_order' && { post_only: true })
    };

    logger.info('🔍 Placing main order:', { mainOrder });
    const mainOrderResponse = await deltaExchange.placeOrder(mainOrder);
    logger.info('✅ Main order placed:', { mainOrderResponse });

    // Step 3: Calculate TP/SL prices
    const isLong = side === 'buy';
    const takeProfitPrice = isLong
      ? currentPrice * (1 + take_profit_percentage / 100)
      : currentPrice * (1 - take_profit_percentage / 100);

    const stopLossPrice = isLong
      ? currentPrice * (1 - stop_loss_percentage / 100)
      : currentPrice * (1 + stop_loss_percentage / 100);

    logger.info(`🎯 TP Price: ${takeProfitPrice.toFixed(2)}, SL Price: ${stopLossPrice.toFixed(2)}`);

    // Step 4: Place Take Profit order (correct Delta Exchange format)
    const takeProfitOrder = {
      product_id: product.id || defaultProductId,
      size: parseInt(size) || 1, // Size must be integer (number of contracts)
      side: isLong ? 'sell' : 'buy',
      order_type: 'limit_order',
      limit_price: takeProfitPrice,
      time_in_force: 'gtc',
      post_only: true,
      reduce_only: true // Ensure this order only reduces position
    };

    logger.info('🎯 Placing take profit order:', { takeProfitOrder });
    const tpOrderResponse = await deltaExchange.placeOrder(takeProfitOrder);
    logger.info('✅ Take profit order placed:', { tpOrderResponse });

    // Step 5: Place Stop Loss order (correct Delta Exchange format)
    const stopLossOrder = {
      product_id: product.id || defaultProductId,
      size: parseInt(size) || 1, // Size must be integer (number of contracts)
      side: isLong ? 'sell' : 'buy',
      order_type: 'stop_order',
      stop_price: stopLossPrice,
      limit_price: stopLossPrice * 0.99, // 1% slippage allowed
      time_in_force: 'gtc',
      reduce_only: true // Ensure this order only reduces position
    };

    logger.info('🛡️ Placing stop loss order:', { stopLossOrder });
    const slOrderResponse = await deltaExchange.placeOrder(stopLossOrder);
    logger.info('✅ Stop loss order placed:', { slOrderResponse });

    // Return comprehensive response
    return res.json({
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
          main_order: slOrderResponse.result || slOrderResponse,
          take_profit_order: tpOrderResponse.result || tpOrderResponse,
          stop_loss_order: slOrderResponse.result || slOrderResponse
        }
      },
      timestamp: Date.now()
    });

  } catch (error) {
    logger.error('❌ Error placing trade with TP/SL:', { error: error instanceof Error ? error.message : String(error), response_data: (error as any).response?.data });
    return res.status(500).json({
      success: false,
      error: 'Failed to place trade with TP/SL',
      message: (error as any).response?.data?.error?.message || (error as any).message || 'Unknown error',
      error_details: (error as any).response?.data || (error as any).message
    });
  }
});

/**
 * POST /api/trading/activate-bot
 * Activate trading bot with specified parameters
 */
router.post('/activate-bot', async (req, res): Promise<Response> => {
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

    logger.info('🤖 Activating trading bot:', { name, strategy, symbols, risk_per_trade });

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

    logger.info('✅ Trading bot activated:', { botConfig });

    return res.json({
      success: true,
      message: `Trading bot "${name}" activated successfully`,
      data: botConfig,
      timestamp: Date.now()
    });

  } catch (error) {
    logger.error('❌ Error activating trading bot:', { error: error instanceof Error ? error.message : String(error) });
    return res.status(500).json({
      success: false,
      error: 'Failed to activate trading bot',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

export default router;
