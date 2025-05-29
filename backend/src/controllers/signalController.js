/**
 * Signal Controller
 * Handles trading signals and strategies
 */

const prisma = require('../utils/prismaClient');
const { createError } = require('../middleware/errorHandler');

/**
 * Get latest trading signals
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @param {Function} next - Express next function
 */
const getLatestSignals = async (req, res, next) => {
  try {
    const { symbol = 'BTCUSD', timeframe = '1h' } = req.query;
    
    // TODO: In production, fetch this from signal generator service
    // For now, return mock data
    const mockSignals = [
      {
        id: 'sig_1',
        time: new Date().toISOString(),
        symbol: symbol,
        timeframe: timeframe,
        type: 'buy',
        strength: 'strong',
        confidence: 0.85,
        price: 46250.75,
        strategy: 'MACD+RSI',
        indicators: {
          macd: { signal: 'buy', value: 245.32 },
          rsi: { signal: 'buy', value: 32.5 }
        }
      },
      {
        id: 'sig_2',
        time: new Date(Date.now() - 3600000).toISOString(),
        symbol: symbol,
        timeframe: timeframe,
        type: 'hold',
        strength: 'neutral',
        confidence: 0.6,
        price: 46150.25,
        strategy: 'MACD+RSI',
        indicators: {
          macd: { signal: 'neutral', value: 110.27 },
          rsi: { signal: 'neutral', value: 48.2 }
        }
      }
    ];
    
    res.json({
      success: true,
      data: mockSignals
    });
  } catch (error) {
    next(createError(`Failed to fetch signals: ${error.message}`, 500));
  }
};

/**
 * Get trading signals history
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @param {Function} next - Express next function
 */
const getSignalHistory = async (req, res, next) => {
  try {
    const { symbol = 'BTCUSD', limit = 20, offset = 0 } = req.query;
    
    // TODO: In production, fetch this from database
    // For now, return mock data
    const mockHistory = [];
    
    // Generate some historical signals
    const now = Date.now();
    for (let i = 0; i < limit; i++) {
      const signalTime = new Date(now - (i * 3600000));
      const signalTypes = ['buy', 'sell', 'hold'];
      const randomType = signalTypes[Math.floor(Math.random() * signalTypes.length)];
      const confidence = 0.5 + (Math.random() * 0.4);
      
      mockHistory.push({
        id: `sig_hist_${i + 1}`,
        time: signalTime.toISOString(),
        symbol: symbol,
        timeframe: '1h',
        type: randomType,
        strength: confidence > 0.7 ? 'strong' : 'medium',
        confidence: confidence,
        price: 45000 + (Math.random() * 2000),
        strategy: 'MACD+RSI',
        result: i < 10 ? {
          success: Math.random() > 0.3,
          pnl: (Math.random() * 4) - 1.5
        } : null
      });
    }
    
    res.json({
      success: true,
      data: mockHistory,
      meta: {
        total: 156,
        limit,
        offset
      }
    });
  } catch (error) {
    next(createError(`Failed to fetch signal history: ${error.message}`, 500));
  }
};

/**
 * Get available trading strategies
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @param {Function} next - Express next function
 */
const getStrategies = async (req, res, next) => {
  try {
    // Mock trading strategies
    const mockStrategies = [
      {
        id: 'macd_rsi',
        name: 'MACD+RSI Strategy',
        description: 'Combines MACD and RSI indicators for momentum-based signals',
        performance: {
          winRate: 0.72,
          avgPnl: 1.34,
          sharpeRatio: 1.8
        },
        parameters: [
          { name: 'macdFast', value: 12 },
          { name: 'macdSlow', value: 26 },
          { name: 'macdSignal', value: 9 },
          { name: 'rsiPeriod', value: 14 },
          { name: 'rsiOverbought', value: 70 },
          { name: 'rsiOversold', value: 30 }
        ]
      },
      {
        id: 'bollinger_breakout',
        name: 'Bollinger Breakout',
        description: 'Generates signals on price breakouts from Bollinger Bands',
        performance: {
          winRate: 0.68,
          avgPnl: 1.85,
          sharpeRatio: 1.65
        },
        parameters: [
          { name: 'period', value: 20 },
          { name: 'standardDeviations', value: 2 },
          { name: 'confirmationCandles', value: 2 }
        ]
      }
    ];
    
    res.json({
      success: true,
      data: mockStrategies
    });
  } catch (error) {
    next(createError(`Failed to fetch strategies: ${error.message}`, 500));
  }
};

module.exports = {
  getLatestSignals,
  getSignalHistory,
  getStrategies
}; 