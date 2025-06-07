import express from 'express';
import { realMarketDataService } from '../services/realMarketDataService';
import { logger } from '../utils/logger';

const router = express.Router();

/**
 * GET /api/real-market-data/portfolio
 * Get portfolio data with real Delta Exchange testnet balance and real market prices
 */
router.get('/portfolio', async (req, res) => {
  try {
    logger.info('📊 Fetching real portfolio data with Delta Exchange testnet balance...');
    
    const portfolioData = await realMarketDataService.getPortfolioData();

    res.json({
      success: true,
      data: portfolioData,
      timestamp: Date.now(),
      message: 'Real portfolio data with Delta Exchange testnet integration'
    });
  } catch (error) {
    logger.error('Error fetching real portfolio data:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch real portfolio data',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/real-market-data
 * Get real market data for all supported symbols using Delta Exchange + CCXT
 */
router.get('/', async (req, res) => {
  try {
    logger.info('📡 Fetching real market data for all symbols...');
    
    const symbols = ['BTCUSD', 'ETHUSD', 'SOLUSD'];
    const marketData = await realMarketDataService.getMultipleMarketData(symbols);

    res.json({
      success: true,
      data: marketData,
      timestamp: Date.now(),
      source: 'delta_exchange_and_ccxt_real_data',
      count: marketData.length,
      message: 'Real market data from Delta Exchange India testnet and CCXT exchanges'
    });
  } catch (error) {
    logger.error('Error fetching real market data:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch real market data',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/real-market-data/:symbol
 * Get real market data for a specific symbol using Delta Exchange + CCXT
 */
router.get('/:symbol', async (req, res) => {
  try {
    const { symbol } = req.params;
    logger.info(`📡 Fetching real market data for ${symbol}...`);
    
    const marketData = await realMarketDataService.getMarketData(symbol.toUpperCase());

    if (!marketData) {
      return res.status(404).json({
        success: false,
        error: 'Symbol not found',
        message: `Real market data not available for symbol: ${symbol}`
      });
    }

    res.json({
      success: true,
      data: marketData,
      timestamp: Date.now(),
      message: `Real market data for ${symbol} from ${marketData.source}`
    });
  } catch (error) {
    logger.error(`Error fetching real market data for ${req.params.symbol}:`, error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch real market data',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/real-market-data/delta/balance
 * Get real Delta Exchange testnet balance
 */
router.get('/delta/balance', async (req, res) => {
  try {
    logger.info('💰 Fetching real Delta Exchange testnet balance...');
    
    // This will be implemented when we have the Delta Exchange service working
    res.json({
      success: true,
      data: {
        message: 'Delta Exchange balance endpoint - implementation in progress',
        note: 'Use /portfolio endpoint for complete portfolio data including balance'
      },
      timestamp: Date.now()
    });
  } catch (error) {
    logger.error('Error fetching Delta Exchange balance:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch Delta Exchange balance',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/real-market-data/health
 * Health check for real market data services
 */
router.get('/health', async (req, res) => {
  try {
    logger.info('🔍 Checking real market data services health...');
    
    // Test a quick market data fetch
    const testData = await realMarketDataService.getMarketData('BTCUSD');
    
    res.json({
      success: true,
      data: {
        status: 'healthy',
        services: {
          deltaExchange: testData?.source?.includes('delta') ? 'connected' : 'fallback',
          ccxtExchanges: testData?.source?.includes('ccxt') ? 'connected' : 'not_used',
          lastDataFetch: testData ? 'successful' : 'failed'
        },
        testData: testData ? {
          symbol: testData.symbol,
          price: testData.price,
          source: testData.source
        } : null
      },
      timestamp: Date.now(),
      message: 'Real market data services health check'
    });
  } catch (error) {
    logger.error('Error in health check:', error);
    res.status(500).json({
      success: false,
      error: 'Health check failed',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

export default router;
