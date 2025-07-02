/**
 * Signal Generation Service
 * Converts ML model predictions into actionable trading signals
 */

import { v4 as uuidv4 } from 'uuid';
import mlModelClient, { MarketData, PredictionResponse } from '../../clients/mlModelClient';
import {
  TradingSignal,
  SignalType,
  SignalDirection,
  SignalStrength,
  SignalTimeframe,
} from '../../types/signals';
import { createLogger } from '../../utils/logger';

// Create logger
const logger = createLogger('SignalGenerationService');

/**
 * Maps a confidence score (0-100) to a SignalStrength enum.
 * @param score - The confidence score.
 * @returns The corresponding signal strength.
 */
function getStrengthFromConfidence(score: number): SignalStrength {
  if (score >= 80) return SignalStrength.VERY_STRONG;
  if (score >= 65) return SignalStrength.STRONG;
  if (score >= 50) return SignalStrength.MODERATE;
  if (score >= 35) return SignalStrength.WEAK;
  return SignalStrength.VERY_WEAK;
}

/**
 * Signal Generation Service class
 * Provides methods to generate and manage trading signals from the core ML model.
 */
export class SignalGenerationService {
  /**
   * Generate a trading signal for a specific symbol using the core strategy.
   * @param symbol - Trading pair symbol (e.g., 'BTC-USD').
   * @param marketData - The market data required by the ML model.
   * @returns A trading signal object or null if the signal is 'HOLD'.
   */
  async generateSignal(
    symbol: string,
    marketData: MarketData,
  ): Promise<TradingSignal | null> {
    try {
      logger.info(`Generating signal for ${symbol}`);

      // 1. Get prediction from the ML service
      const prediction = await mlModelClient.getPrediction(marketData);
      logger.debug(`Received prediction for ${symbol}`, { prediction });

      // 2. If the signal is HOLD, do nothing.
      if (prediction.signal === 'HOLD') {
        logger.info(`HOLD signal received for ${symbol}. No action taken.`, { reason: prediction.reason });
        return null;
      }

      // 3. If we have a BUY or SELL signal, construct the full TradingSignal object.
      const currentPrice = marketData.m15.close[marketData.m15.close.length - 1];
      if (!currentPrice) {
        throw new Error('Could not determine current price from m15 market data.');
      }
      if (!prediction.stop_loss || !prediction.target_price) {
          throw new Error('Signal response missing stop_loss or target_price.');
      }

      const direction = prediction.signal === 'BUY' ? SignalDirection.LONG : SignalDirection.SHORT;
      
      const expectedReturn = Math.abs((prediction.target_price - currentPrice) / currentPrice) * 100;
      const expectedRisk = Math.abs((prediction.stop_loss - currentPrice) / currentPrice) * 100;
      const riskRewardRatio = expectedRisk > 0 ? expectedReturn / expectedRisk : 0;

      const signal: TradingSignal = {
        id: uuidv4(),
        symbol: symbol,
        type: SignalType.ENTRY,
        direction: direction,
        strength: getStrengthFromConfidence(prediction.confidence),
        timeframe: SignalTimeframe.SHORT, // Our strategy is short-term focused
        price: currentPrice,
        targetPrice: prediction.target_price,
        stopLoss: prediction.stop_loss,
        confidenceScore: prediction.confidence,
        expectedReturn: expectedReturn,
        expectedRisk: expectedRisk,
        riskRewardRatio: riskRewardRatio,
        generatedAt: new Date().toISOString(),
        source: 'CoreTradingStrategy_v1',
        metadata: { reason: prediction.reason },
        predictionValues: [], // Raw values no longer provided
      };

      logger.info(`Generated ${signal.direction} signal for ${symbol}`, { signalId: signal.id });

      // Note: Storing the signal in the database is now handled by a separate service/process.
      // This service is only responsible for generation.

      return signal;

    } catch (error) {
      const logData = {
        symbol,
        error: error instanceof Error ? error.message : String(error),
      };
      logger.error(`Error generating signal for ${symbol}`, logData);
      // Re-throw the error to be handled by the caller
      throw error;
    }
  }
}