/**
 * Backtest Service
 * Handles backtest creation, storage, and management
 */

import prisma from '../../utils/prismaClient';
import { Backtest } from '../../../generated/prisma';
import { logger } from '../../utils/logger';

export interface BacktestConfig {
  botId: string;
  symbol: string;
  timeframe: string;
  startDate: Date;
  endDate: Date;
  initialCapital: number;
  leverage: number;
  riskPerTrade: number;
  commission: number;
  modelName?: string;
  modelVersion?: string;
  optimizationMethod?: string;
}

export interface BacktestPerformance {
  totalReturn: number;
  totalReturnPercent: number;
  annualizedReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  maxDrawdownPercent: number;
  winRate: number;
  profitFactor: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  averageWin: number;
  averageLoss: number;
  largestWin: number;
  largestLoss: number;
}

export interface BacktestTrade {
  id: string;
  side: 'buy' | 'sell';
  entryTime: string;
  exitTime: string;
  entryPrice: number;
  exitPrice: number;
  size: number;
  pnl: number;
  pnlPercent: number;
  fees: number;
  status: string;
  symbol: string;
  strategy: string;
}

/**
 * Create a new backtest record
 */
export const createBacktest = async (
  config: BacktestConfig,
  performance: BacktestPerformance,
  trades?: BacktestTrade[]
): Promise<Backtest> => {
  try {
    const backtest = await prisma.backtest.create({
      data: {
        botId: config.botId,
        symbol: config.symbol,
        timeframe: config.timeframe,
        startDate: config.startDate,
        endDate: config.endDate,
        performance: performance as any,
        config: config as any,
        trades: trades as any,
        status: 'completed',
        duration: Date.now() - config.startDate.getTime(),
        modelName: config.modelName,
        modelVersion: config.modelVersion,
        optimizationMethod: config.optimizationMethod
      }
    });

    logger.info(`Created backtest ${backtest.id} for bot ${config.botId}`);
    return backtest;
  } catch (error) {
    logger.error('Error creating backtest:', error);
    throw error;
  }
};

/**
 * Update backtest status
 */
export const updateBacktestStatus = async (
  backtestId: string,
  status: 'running' | 'completed' | 'failed',
  performance?: BacktestPerformance,
  trades?: BacktestTrade[]
): Promise<Backtest> => {
  try {
    const updateData: any = { status };
    
    if (performance) {
      updateData.performance = performance;
    }
    
    if (trades) {
      updateData.trades = trades;
    }
    
    if (status === 'completed' || status === 'failed') {
      updateData.duration = Date.now();
    }

    const backtest = await prisma.backtest.update({
      where: { id: backtestId },
      data: updateData
    });

    logger.info(`Updated backtest ${backtestId} status to ${status}`);
    return backtest;
  } catch (error) {
    logger.error('Error updating backtest status:', error);
    throw error;
  }
};

/**
 * Get backtest by ID
 */
export const getBacktestById = async (backtestId: string): Promise<Backtest | null> => {
  try {
    return await prisma.backtest.findUnique({
      where: { id: backtestId },
      include: {
        bot: {
          select: {
            id: true,
            name: true,
            strategy: true,
            userId: true
          }
        }
      }
    });
  } catch (error) {
    logger.error('Error getting backtest by ID:', error);
    throw error;
  }
};

/**
 * Get backtests for a bot
 */
export const getBacktestsForBot = async (
  botId: string,
  limit: number = 10,
  offset: number = 0,
  status?: string
): Promise<{
  backtests: Backtest[];
  total: number;
  limit: number;
  offset: number;
  hasMore: boolean;
}> => {
  try {
    const where: any = { botId };
    if (status) {
      where.status = status;
    }

    const [backtests, total] = await Promise.all([
      prisma.backtest.findMany({
        where,
        orderBy: { createdAt: 'desc' },
        skip: offset,
        take: limit,
        include: {
          bot: {
            select: {
              id: true,
              name: true,
              strategy: true
            }
          }
        }
      }),
      prisma.backtest.count({ where })
    ]);

    return {
      backtests,
      total,
      limit,
      offset,
      hasMore: offset + limit < total
    };
  } catch (error) {
    logger.error('Error getting backtests for bot:', error);
    throw error;
  }
};

/**
 * Get backtests for a user
 */
export const getBacktestsForUser = async (
  userId: string,
  limit: number = 20,
  offset: number = 0,
  symbol?: string,
  timeframe?: string
): Promise<{
  backtests: Backtest[];
  total: number;
  limit: number;
  offset: number;
  hasMore: boolean;
}> => {
  try {
    const where: any = {
      bot: {
        userId
      }
    };

    if (symbol) {
      where.symbol = symbol;
    }

    if (timeframe) {
      where.timeframe = timeframe;
    }

    const [backtests, total] = await Promise.all([
      prisma.backtest.findMany({
        where,
        orderBy: { createdAt: 'desc' },
        skip: offset,
        take: limit,
        include: {
          bot: {
            select: {
              id: true,
              name: true,
              strategy: true,
              symbol: true
            }
          }
        }
      }),
      prisma.backtest.count({ where })
    ]);

    return {
      backtests,
      total,
      limit,
      offset,
      hasMore: offset + limit < total
    };
  } catch (error) {
    logger.error('Error getting backtests for user:', error);
    throw error;
  }
};

/**
 * Delete a backtest
 */
export const deleteBacktest = async (backtestId: string, userId: string): Promise<boolean> => {
  try {
    // First verify the user owns the bot that this backtest belongs to
    const backtest = await prisma.backtest.findUnique({
      where: { id: backtestId },
      include: {
        bot: {
          select: {
            userId: true
          }
        }
      }
    });

    if (!backtest) {
      throw new Error('Backtest not found');
    }

    if (backtest.bot.userId !== userId) {
      throw new Error('Access denied');
    }

    await prisma.backtest.delete({
      where: { id: backtestId }
    });

    logger.info(`Deleted backtest ${backtestId}`);
    return true;
  } catch (error) {
    logger.error('Error deleting backtest:', error);
    throw error;
  }
};

/**
 * Get best performing backtests
 */
export const getBestBacktests = async (
  userId: string,
  metric: 'totalReturn' | 'sharpeRatio' | 'winRate' = 'sharpeRatio',
  limit: number = 10
): Promise<Backtest[]> => {
  try {
    // Use raw query to sort by JSON field
    const backtests = await prisma.$queryRaw`
      SELECT b.*, bot.name as bot_name, bot.strategy as bot_strategy
      FROM "Backtest" b
      JOIN "Bot" bot ON b."botId" = bot.id
      WHERE bot."userId" = ${userId}
      AND b.status = 'completed'
      ORDER BY CAST(b.performance->>${metric} AS FLOAT) DESC
      LIMIT ${limit}
    `;

    return backtests as Backtest[];
  } catch (error) {
    logger.error('Error getting best backtests:', error);
    throw error;
  }
};

/**
 * Create backtest comparison
 */
export const createBacktestComparison = async (
  name: string,
  description: string | null,
  backtestIds: string[]
): Promise<any> => {
  try {
    // Get the backtests to compare
    const backtests = await prisma.backtest.findMany({
      where: {
        id: {
          in: backtestIds
        }
      },
      include: {
        bot: {
          select: {
            name: true,
            strategy: true
          }
        }
      }
    });

    if (backtests.length !== backtestIds.length) {
      throw new Error('Some backtests not found');
    }

    // Analyze and compare the backtests
    const results = analyzeBacktestComparison(backtests);

    // Create the comparison record
    const comparison = await prisma.backtestComparison.create({
      data: {
        name,
        description,
        results: results as any,
        winner: results.winner
      }
    });

    // Create the junction table entries
    await Promise.all(
      backtestIds.map(backtestId =>
        prisma.backtestComparisonBacktest.create({
          data: {
            backtestId,
            backtestComparisonId: comparison.id
          }
        })
      )
    );

    logger.info(`Created backtest comparison ${comparison.id} with ${backtestIds.length} backtests`);
    return comparison;
  } catch (error) {
    logger.error('Error creating backtest comparison:', error);
    throw error;
  }
};

/**
 * Analyze backtest comparison
 */
function analyzeBacktestComparison(backtests: any[]): any {
  const metrics = ['totalReturn', 'sharpeRatio', 'maxDrawdown', 'winRate'];
  const analysis: any = {
    summary: {},
    detailed: {},
    winner: null,
    scores: {}
  };

  // Calculate scores for each backtest
  backtests.forEach(backtest => {
    let score = 0;
    const performance = backtest.performance;

    // Scoring algorithm (weighted)
    score += (performance.totalReturn || 0) * 0.3;
    score += (performance.sharpeRatio || 0) * 0.4;
    score += (1 - Math.abs(performance.maxDrawdown || 0)) * 0.2;
    score += (performance.winRate || 0) * 0.1;

    analysis.scores[backtest.id] = score;
    analysis.detailed[backtest.id] = {
      botName: backtest.bot.name,
      strategy: backtest.bot.strategy,
      performance: performance,
      score: score
    };
  });

  // Find winner
  const winnerEntry = Object.entries(analysis.scores).reduce((a: any, b: any) => 
    a[1] > b[1] ? a : b
  );
  analysis.winner = winnerEntry[0];

  // Summary statistics
  metrics.forEach(metric => {
    const values = backtests.map(b => b.performance[metric] || 0);
    analysis.summary[metric] = {
      min: Math.min(...values),
      max: Math.max(...values),
      avg: values.reduce((a, b) => a + b, 0) / values.length,
      std: Math.sqrt(values.reduce((a, b) => a + Math.pow(b - analysis.summary[metric]?.avg || 0, 2), 0) / values.length)
    };
  });

  return analysis;
}

/**
 * Get backtest comparison by ID
 */
export const getBacktestComparison = async (comparisonId: string): Promise<any> => {
  try {
    return await prisma.backtestComparison.findUnique({
      where: { id: comparisonId },
      include: {
        backtests: {
          include: {
            backtest: {
              include: {
                bot: {
                  select: {
                    name: true,
                    strategy: true
                  }
                }
              }
            }
          }
        }
      }
    });
  } catch (error) {
    logger.error('Error getting backtest comparison:', error);
    throw error;
  }
};