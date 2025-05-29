/**
 * Trading Strategy Types
 * Definitions for the strategy execution system
 */

import { Timestamp } from './common';
import { TradingSignal, SignalDirection, SignalType } from './signals';

/**
 * Strategy types represent different trading approaches
 */
export enum StrategyType {
  TREND_FOLLOWING = 'TREND_FOLLOWING',    // Follows market trends
  MEAN_REVERSION = 'MEAN_REVERSION',      // Reverts to the mean
  BREAKOUT = 'BREAKOUT',                  // Trades breakouts
  MOMENTUM = 'MOMENTUM',                  // Trades momentum
  ARBITRAGE = 'ARBITRAGE',                // Exploits price differences
  GRID = 'GRID',                          // Grid trading
  MARTINGALE = 'MARTINGALE',              // Martingale approach
  ML_PREDICTION = 'ML_PREDICTION',        // Based on ML predictions
  CUSTOM = 'CUSTOM'                       // Custom strategy
}

/**
 * Strategy time horizons
 */
export enum StrategyTimeHorizon {
  SCALPING = 'SCALPING',           // Minutes to hours
  INTRADAY = 'INTRADAY',           // Within a day
  SWING = 'SWING',                 // Days to weeks
  POSITION = 'POSITION',           // Weeks to months
  LONG_TERM = 'LONG_TERM'          // Months to years
}

/**
 * Strategy execution status
 */
export enum StrategyExecutionStatus {
  ACTIVE = 'ACTIVE',               // Strategy is actively running
  PAUSED = 'PAUSED',               // Strategy is temporarily paused
  STOPPED = 'STOPPED',             // Strategy is stopped
  BACKTEST = 'BACKTEST',           // Strategy is running in backtest mode
  SIMULATION = 'SIMULATION',       // Strategy is running in simulation mode
  ERROR = 'ERROR'                  // Strategy encountered an error
}

/**
 * Entry rule types
 */
export enum EntryRuleType {
  SIGNAL_BASED = 'SIGNAL_BASED',         // Based on trading signals
  PRICE_BREAKOUT = 'PRICE_BREAKOUT',     // Price breaks above/below level
  INDICATOR_CROSS = 'INDICATOR_CROSS',   // Indicator crosses another
  PATTERN_MATCH = 'PATTERN_MATCH',       // Chart pattern match
  TIME_BASED = 'TIME_BASED',             // Time-based entry
  VOLUME_SPIKE = 'VOLUME_SPIKE',         // Volume spike
  ML_PREDICTION = 'ML_PREDICTION',       // ML prediction based
  CUSTOM = 'CUSTOM'                      // Custom rule
}

/**
 * Exit rule types
 */
export enum ExitRuleType {
  SIGNAL_BASED = 'SIGNAL_BASED',         // Based on trading signals
  STOP_LOSS = 'STOP_LOSS',               // Stop loss hit
  TAKE_PROFIT = 'TAKE_PROFIT',           // Take profit hit
  TRAILING_STOP = 'TRAILING_STOP',       // Trailing stop hit
  TIME_BASED = 'TIME_BASED',             // Time-based exit
  INDICATOR_BASED = 'INDICATOR_BASED',   // Based on indicator
  ML_PREDICTION = 'ML_PREDICTION',       // ML prediction based
  CUSTOM = 'CUSTOM'                      // Custom rule
}

/**
 * Strategy rule configuration
 */
export interface StrategyRule {
  id: string;
  name: string;
  type: EntryRuleType | ExitRuleType;
  direction?: SignalDirection;          // For entry rules
  parameters: Record<string, any>;      // Rule-specific parameters
  priority: number;                     // Execution priority (lower = higher)
  isRequired: boolean;                  // Must be satisfied
  description?: string;
}

/**
 * Strategy configuration
 */
export interface StrategyConfig {
  id: string;
  name: string;
  description?: string;
  type: StrategyType;
  timeHorizon: StrategyTimeHorizon;
  symbols: string[];                    // Trading symbols
  entryRules: StrategyRule[];           // Entry rules
  exitRules: StrategyRule[];            // Exit rules
  positionSizing: {                     // Position sizing configuration
    method: string;                     // Position sizing method
    parameters: Record<string, any>;    // Method-specific parameters
  };
  riskManagement: {                     // Risk management configuration
    maxPositionSize: number;            // Maximum position size
    maxDrawdown: number;                // Maximum allowed drawdown
    maxOpenPositions: number;           // Maximum open positions
    useCircuitBreakers: boolean;        // Use circuit breakers
    targetRiskRewardRatio?: number;     // Target risk-reward ratio
  };
  indicators: {                         // Technical indicators
    name: string;
    parameters: Record<string, any>;
  }[];
  isActive: boolean;                    // Is strategy active
  userId: string;                       // User who owns the strategy
  isPublic: boolean;                    // Whether strategy is publicly visible
  createdAt: Timestamp;
  updatedAt: Timestamp;
}

/**
 * Strategy execution instance
 */
export interface StrategyExecution {
  id: string;
  strategyId: string;
  userId: string;
  botId?: string;                       // Optional bot ID
  status: StrategyExecutionStatus;
  lastExecutedAt?: Timestamp;
  startedAt: Timestamp;
  stoppedAt?: Timestamp;
  currentPositions: string[];           // Position IDs
  historicalPositions: string[];        // Historical position IDs
  performance: {                        // Performance metrics
    totalPnL: number;
    winRate: number;
    totalTrades: number;
    successfulTrades: number;
    failedTrades: number;
    averageHoldingTime: number;
    maxDrawdown: number;
  };
  logs: {                               // Execution logs
    timestamp: Timestamp;
    message: string;
    level: 'info' | 'warning' | 'error';
    data?: any;
  }[];
  errors: {                             // Error logs
    timestamp: Timestamp;
    message: string;
    stackTrace?: string;
  }[];
}

/**
 * Strategy execution result for a single signal
 */
export interface StrategyExecutionResult {
  executionId: string;
  signal: TradingSignal;
  entryRuleResults: {
    ruleId: string;
    satisfied: boolean;
    details: any;
  }[];
  exitRuleResults: {
    ruleId: string;
    satisfied: boolean;
    details: any;
  }[];
  action: 'ENTRY' | 'EXIT' | 'INCREASE' | 'DECREASE' | 'HOLD' | 'NONE';
  positionSize?: number;
  entryPrice?: number;
  stopLossPrice?: number;
  takeProfitPrice?: number;
  confidence: number;                   // 0-100
  notes: string;
  timestamp: Timestamp;
}

/**
 * Strategy validation result
 */
export interface StrategyValidationResult {
  isValid: boolean;
  errors: string[];
  warnings: string[];
  suggestions: string[];
}

/**
 * Strategy backtest configuration
 */
export interface StrategyBacktestConfig {
  strategyId: string;
  startDate: Timestamp;
  endDate: Timestamp;
  initialCapital: number;
  symbols: string[];
  includeFees: boolean;
  feePercentage: number;
  includeSlippage: boolean;
  slippagePercentage: number;
}

/**
 * Strategy backtest result
 */
export interface StrategyBacktestResult {
  id: string;
  strategyId: string;
  config: StrategyBacktestConfig;
  performance: {
    totalPnL: number;
    totalPnLPercentage: number;
    winRate: number;
    totalTrades: number;
    successfulTrades: number;
    failedTrades: number;
    averageHoldingTime: number;
    maxDrawdown: number;
    sharpeRatio: number;
    sortinoRatio: number;
  };
  trades: {
    symbol: string;
    direction: SignalDirection;
    entryPrice: number;
    exitPrice: number;
    quantity: number;
    entryDate: Timestamp;
    exitDate: Timestamp;
    pnl: number;
    pnlPercentage: number;
    fees: number;
    slippage: number;
  }[];
  equityCurve: {
    timestamp: Timestamp;
    equity: number;
  }[];
  createdAt: Timestamp;
} 