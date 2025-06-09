/**
 * Trading Routes
 * Delta Exchange India trading integration endpoints
 */

import express from 'express';
import { DeltaExchangeService, DeltaCredentials, OrderRequest } from '../services/deltaExchangeService';

// Simple console logger
const logger = {
  info: (message: string, ...args: any[]) => console.log(`[INFO] ${message}`, ...args),
  error: (message: string, ...args: any[]) => console.error(`[ERROR] ${message}`, ...args),
  warn: (message: string, ...args: any[]) => console.warn(`[WARN] ${message}`, ...args),
  debug: (message: string, ...args: any[]) => console.log(`[DEBUG] ${message}`, ...args)
};

const router = express.Router();

// Global Delta Exchange service instance
let deltaService: any = null;

/**
 * Initialize Delta Exchange service with credentials
 */
const initializeDeltaService = () => {
  if (!deltaService) {
    const credentials: DeltaCredentials = {
      apiKey: process.env.DELTA_EXCHANGE_API_KEY || '',
      apiSecret: process.env.DELTA_EXCHANGE_API_SECRET || '',
      testnet: process.env.DELTA_EXCHANGE_TESTNET === 'true'
    };

    // Validate credentials
    if (!credentials.apiKey || !credentials.apiSecret) {
      logger.error('❌ Delta Exchange credentials not configured');
      throw new Error('Delta Exchange API credentials are required');
    }

    deltaService = new DeltaExchangeService(credentials);
    logger.info('🔄 Delta Exchange service initialized with real credentials');
    logger.info(`🌐 Environment: ${credentials.testnet ? 'TESTNET' : 'PRODUCTION'}`);
  }
  return deltaService;
};

/**
 * GET /api/trading/status
 * Get trading service status
 */
router.get('/status', async (req, res) => {
  try {
    const service = initializeDeltaService();
    const isReady = service.isReady();
    const supportedSymbols = service.getSupportedSymbols();

    res.json({
      success: true,
      data: {
        status: isReady ? 'connected' : 'initializing',
        exchange: 'delta_exchange_india',
        environment: process.env.DELTA_EXCHANGE_TESTNET === 'true' ? 'testnet' : 'production',
        supportedSymbols: supportedSymbols.slice(0, 10), // Show first 10
        totalSymbols: supportedSymbols.length,
        timestamp: Date.now()
      }
    });
  } catch (error) {
    logger.error('Error getting trading status:', error);
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
    const service = initializeDeltaService();
    const products = service.getAllProducts();

    // Filter and format products
    const formattedProducts = products.map(product => ({
      id: product.id,
      symbol: product.symbol,
      description: product.description,
      contract_type: product.contract_type,
      state: product.state,
      tick_size: product.tick_size,
      contract_value: product.contract_value,
      maker_commission_rate: product.maker_commission_rate,
      taker_commission_rate: product.taker_commission_rate,
      settlement_time: product.settlement_time,
      underlying_asset: product.underlying_asset,
      quoting_asset: product.quoting_asset,
      settling_asset: product.settling_asset
    }));

    res.json({
      success: true,
      data: formattedProducts,
      meta: {
        total: formattedProducts.length,
        exchange: 'delta_exchange_india',
        timestamp: Date.now()
      }
    });
  } catch (error) {
    logger.error('Error getting products:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get products',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/trading/products/:symbol
 * Get specific product information
 */
router.get('/products/:symbol', async (req, res) => {
  try {
    const { symbol } = req.params;
    const service = initializeDeltaService();
    const product = service.getProduct(symbol.toUpperCase());

    if (!product) {
      return res.status(404).json({
        success: false,
        error: 'Product not found',
        message: `Product not available for symbol: ${symbol}`
      });
    }

    res.json({
      success: true,
      data: product,
      timestamp: Date.now()
    });
  } catch (error) {
    logger.error(`Error getting product ${req.params.symbol}:`, error);
    res.status(500).json({
      success: false,
      error: 'Failed to get product',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/trading/market-data/:symbol
 * Get real-time market data from Delta Exchange
 */
router.get('/market-data/:symbol', async (req, res) => {
  try {
    const { symbol } = req.params;
    const service = initializeDeltaService();
    const marketData = await service.getMarketData(symbol.toUpperCase());

    if (!marketData) {
      return res.status(404).json({
        success: false,
        error: 'Market data not found',
        message: `Market data not available for symbol: ${symbol}`
      });
    }

    res.json({
      success: true,
      data: marketData,
      timestamp: Date.now(),
      source: 'delta_exchange_india'
    });
  } catch (error) {
    logger.error(`Error getting market data for ${req.params.symbol}:`, error);
    res.status(500).json({
      success: false,
      error: 'Failed to get market data',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/trading/market-data
 * Get market data for multiple symbols
 */
router.get('/market-data', async (req, res) => {
  try {
    const service = initializeDeltaService();
    const symbols = req.query.symbols
      ? (req.query.symbols as string).split(',').map(s => s.trim().toUpperCase())
      : ['BTCUSD', 'ETHUSD', 'ADAUSD', 'SOLUSD', 'DOTUSD'];

    const marketData = await service.getMultipleMarketData(symbols);

    res.json({
      success: true,
      data: marketData,
      meta: {
        symbols_requested: symbols,
        symbols_returned: marketData.length,
        timestamp: Date.now(),
        source: 'delta_exchange_india'
      }
    });
  } catch (error) {
    logger.error('Error getting multiple market data:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get market data',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * POST /api/trading/orders
 * Place a new order
 */
router.post('/orders', async (req, res) => {
  try {
    const service = initializeDeltaService();

    if (!service.isReady()) {
      return res.status(503).json({
        success: false,
        error: 'Trading service not ready',
        message: 'Delta Exchange service is still initializing'
      });
    }

    const orderRequest: OrderRequest = req.body;

    // Validate required fields
    if (!orderRequest.product_id || !orderRequest.size || !orderRequest.side) {
      return res.status(400).json({
        success: false,
        error: 'Invalid order request',
        message: 'Missing required fields: product_id, size, side'
      });
    }

    // Place real order using Delta Exchange API
    const order = await service.placeOrder(orderRequest);

    res.json({
      success: true,
      data: order,
      message: `Order placed successfully on Delta Exchange ${process.env.DELTA_EXCHANGE_TESTNET === 'true' ? '(testnet)' : '(production)'}`,
      timestamp: Date.now()
    });
  } catch (error) {
    logger.error('Error placing order:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to place order',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * DELETE /api/trading/orders/:orderId
 * Cancel an order
 */
router.delete('/orders/:orderId', async (req, res) => {
  try {
    const { orderId } = req.params;
    const { product_id } = req.query;
    const service = initializeDeltaService();

    if (!product_id) {
      return res.status(400).json({
        success: false,
        error: 'Missing product_id',
        message: 'product_id query parameter is required'
      });
    }

    // Cancel real order using Delta Exchange API
    const success = await service.cancelOrder(parseInt(product_id as string), parseInt(orderId));

    if (success) {
      res.json({
        success: true,
        message: `Order cancelled successfully on Delta Exchange ${process.env.DELTA_EXCHANGE_TESTNET === 'true' ? '(testnet)' : '(production)'}`,
        timestamp: Date.now()
      });
    } else {
      res.status(400).json({
        success: false,
        error: 'Failed to cancel order',
        message: 'Order cancellation was not successful'
      });
    }
  } catch (error) {
    logger.error('Error cancelling order:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to cancel order',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/trading/orders
 * Get open orders
 */
router.get('/orders', async (req, res) => {
  try {
    const service = initializeDeltaService();
    const { product_id } = req.query;

    // Get real orders from Delta Exchange API
    const orders = await service.getOpenOrders(product_id ? parseInt(product_id as string) : undefined);

    res.json({
      success: true,
      data: orders,
      message: `Orders from Delta Exchange ${process.env.DELTA_EXCHANGE_TESTNET === 'true' ? '(testnet)' : '(production)'}`,
      timestamp: Date.now()
    });
  } catch (error) {
    logger.error('Error getting orders:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get orders',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/trading/positions
 * Get current positions
 */
router.get('/positions', async (req, res) => {
  try {
    const service = initializeDeltaService();

    // Get real positions from Delta Exchange API
    const positions = await service.getPositions();

    res.json({
      success: true,
      data: positions,
      message: `Positions from Delta Exchange ${process.env.DELTA_EXCHANGE_TESTNET === 'true' ? '(testnet)' : '(production)'}`,
      timestamp: Date.now()
    });
  } catch (error) {
    logger.error('Error getting positions:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get positions',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/trading/balances
 * Get wallet balances
 */
router.get('/balances', async (req, res) => {
  try {
    const service = initializeDeltaService();

    // Get real balances from Delta Exchange API
    const balances = await service.getBalances();

    res.json({
      success: true,
      data: balances,
      message: `Balances from Delta Exchange ${process.env.DELTA_EXCHANGE_TESTNET === 'true' ? '(testnet)' : '(production)'}`,
      timestamp: Date.now()
    });
  } catch (error) {
    logger.error('Error getting balances:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get balances',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

export default router;
