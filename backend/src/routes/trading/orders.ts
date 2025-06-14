import express from 'express';
import { validateOrderParams } from '../../schemas/orderValidation';
import { DeltaExchangeUnified } from '../../services/DeltaExchangeUnified';
import logger from '../../utils/logger';

const router = express.Router();

// Create DeltaExchangeUnified instance with test credentials
const deltaExchange = new DeltaExchangeUnified({
  apiKey: process.env.DELTA_API_KEY || '',
  apiSecret: process.env.DELTA_API_SECRET || '',
  testnet: process.env.DELTA_TESTNET === 'true'
});

// Initialize the DeltaExchangeUnified client
(async () => {
  try {
    await deltaExchange.initialize();
    logger.info('Delta Exchange client initialized successfully');
  } catch (error) {
    logger.error('Failed to initialize Delta Exchange client:', error);
  }
})();

/**
 * @route POST /api/trading/orders
 * @description Place a new order on Delta Exchange
 * @access Private
 */
router.post('/', async (req, res) => {
  try {
    // Validate the order parameters
    const { error, value } = validateOrderParams(req.body);
    
    if (error) {
      logger.warn('Invalid order parameters:', error.message);
      return res.status(400).json({
        success: false,
        error: 'Validation Error',
        message: error.message
      });
    }
    
    // Extract order parameters
    const { 
      product_id, 
      side, 
      order_type, 
      size,
      limit_price,
      leverage,
      reduce_only,
      post_only,
      client_order_id
    } = value;
    
    // Ensure Delta Exchange client is initialized
    if (!deltaExchange.isInitialized()) {
      await deltaExchange.initialize();
    }
    
    // Place order
    const orderResult = await deltaExchange.placeOrder({
      product_id,
      size,
      side,
      order_type,
      ...(order_type === 'limit_order' && { limit_price }),
      ...(leverage && { leverage }),
      ...(reduce_only !== undefined && { reduce_only }),
      ...(post_only !== undefined && { post_only }),
      ...(client_order_id && { client_order_id })
    });
    
    logger.info(`Order placed successfully: ${JSON.stringify(orderResult)}`);
    
    return res.status(200).json({
      success: true,
      data: orderResult,
      message: 'Order placed successfully'
    });
  } catch (error) {
    logger.error('Error placing order:', error);
    
    // Handle different error types
    if (error.response && error.response.data) {
      return res.status(error.response.status || 400).json({
        success: false,
        error: 'Delta Exchange API Error',
        message: error.response.data.message || 'Failed to place order',
        code: error.response.data.code
      });
    }
    
    return res.status(500).json({
      success: false,
      error: 'Order Placement Failed',
      message: error.message || 'An unexpected error occurred'
    });
  }
});

/**
 * @route GET /api/trading/orders
 * @description Get all open orders
 * @access Private
 */
router.get('/', async (req, res) => {
  try {
    // Ensure Delta Exchange client is initialized
    if (!deltaExchange.isInitialized()) {
      await deltaExchange.initialize();
    }
    
    const openOrders = await deltaExchange.getOpenOrders();
    
    return res.status(200).json({
      success: true,
      data: openOrders,
      message: 'Open orders retrieved successfully'
    });
  } catch (error) {
    logger.error('Error fetching open orders:', error);
    
    return res.status(500).json({
      success: false,
      error: 'Failed to Fetch Orders',
      message: error.message || 'An unexpected error occurred'
    });
  }
});

/**
 * @route DELETE /api/trading/orders/:orderId
 * @description Cancel an order by ID
 * @access Private
 */
router.delete('/:orderId', async (req, res) => {
  try {
    const { orderId } = req.params;
    
    if (!orderId) {
      return res.status(400).json({
        success: false,
        error: 'Missing Order ID',
        message: 'Order ID is required'
      });
    }
    
    // Ensure Delta Exchange client is initialized
    if (!deltaExchange.isInitialized()) {
      await deltaExchange.initialize();
    }
    
    const result = await deltaExchange.cancelOrder(orderId);
    
    return res.status(200).json({
      success: true,
      data: result,
      message: 'Order cancelled successfully'
    });
  } catch (error) {
    logger.error(`Error cancelling order ${req.params.orderId}:`, error);
    
    return res.status(500).json({
      success: false,
      error: 'Failed to Cancel Order',
      message: error.message || 'An unexpected error occurred'
    });
  }
});

export default router; 