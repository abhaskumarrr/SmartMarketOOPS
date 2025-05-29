/**
 * Trading Signal Types
 * Definitions for the signal generation system
 */

import { Timestamp } from './common';

/**
 * Signal types represent different kinds of trading signals
 */
export enum SignalType {
  ENTRY = 'ENTRY',       // Signal to enter a position
  EXIT = 'EXIT',         // Signal to exit a position
  INCREASE = 'INCREASE', // Signal to increase position size
  DECREASE = 'DECREASE', // Signal to decrease position size
  HOLD = 'HOLD'          // Signal to maintain current position
}

/**
 * Signal direction represents the market direction
 */
export enum SignalDirection {
  LONG = 'LONG',   // Bullish signal
  SHORT = 'SHORT', // Bearish signal
  NEUTRAL = 'NEUTRAL' // Neutral signal
}

/**
 * Signal strength represents the confidence level
 */
export enum SignalStrength {
  VERY_WEAK = 'VERY_WEAK',     // 0-20% confidence
  WEAK = 'WEAK',               // 20-40% confidence
  MODERATE = 'MODERATE',       // 40-60% confidence
  STRONG = 'STRONG',           // 60-80% confidence
  VERY_STRONG = 'VERY_STRONG'  // 80-100% confidence
}

/**
 * Signal timeframe represents the expected duration
 */
export enum SignalTimeframe {
  VERY_SHORT = 'VERY_SHORT', // Hours (intraday)
  SHORT = 'SHORT',           // 1 day
  MEDIUM = 'MEDIUM',         // Days (2-3)
  LONG = 'LONG',             // Week
  VERY_LONG = 'VERY_LONG'    // Weeks or longer
}

/**
 * Model prediction from ML system
 */
export interface ModelPrediction {
  symbol: string;
  predictions: number[];
  model_version: string;
  timestamp: string;
  features_used?: string[];
}

/**
 * Trading signal generated from model prediction
 */
export interface TradingSignal {
  id: string;
  symbol: string;
  type: SignalType;
  direction: SignalDirection;
  strength: SignalStrength;
  timeframe: SignalTimeframe;
  price: number;
  targetPrice?: number;
  stopLoss?: number;
  confidenceScore: number; // 0-100
  expectedReturn: number;
  expectedRisk: number;
  riskRewardRatio: number;
  generatedAt: string;
  expiresAt?: string;
  source: string; // Source of the signal (model name/version)
  metadata: Record<string, any>; // Additional metadata
  predictionValues: number[]; // Raw prediction values
  
  // Validation fields
  validatedAt?: string;
  validationStatus?: boolean;
  validationReason?: string;
}

/**
 * Options for signal generation
 */
export interface SignalGenerationOptions {
  validateSignals: boolean;
  useHistoricalData: boolean;
  lookbackPeriod: number;
  minConfidenceThreshold: number;
  maxSignalsPerSymbol: number;
  filterWeakSignals: boolean;
}

/**
 * Criteria for filtering signals
 */
export interface SignalFilterCriteria {
  symbol?: string;
  types?: SignalType[];
  directions?: SignalDirection[];
  minStrength?: SignalStrength;
  timeframes?: SignalTimeframe[];
  minConfidenceScore?: number;
  fromTimestamp?: string;
  toTimestamp?: string;
  status?: 'active' | 'expired' | 'validated' | 'invalidated' | 'all';
}

/**
 * Signal event for broadcasting
 */
export interface SignalEvent {
  eventType: 'new' | 'update' | 'expired' | 'validated' | 'invalidated';
  signal: TradingSignal;
  timestamp: Timestamp;
} 