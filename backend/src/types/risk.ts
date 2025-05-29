/**
 * Risk Management Types
 * Definitions for the risk management system
 */

import { Timestamp } from './common';

/**
 * Risk level enum for portfolio and positions
 */
export enum RiskLevel {
  VERY_LOW = 'VERY_LOW',     // 0-20% of max allowed risk
  LOW = 'LOW',               // 20-40% of max allowed risk
  MODERATE = 'MODERATE',     // 40-60% of max allowed risk
  HIGH = 'HIGH',             // 60-80% of max allowed risk
  VERY_HIGH = 'VERY_HIGH'    // 80-100% of max allowed risk
}

/**
 * Position sizing methods
 */
export enum PositionSizingMethod {
  FIXED_FRACTIONAL = 'FIXED_FRACTIONAL',  // Risk a fixed percentage of account on each trade
  KELLY_CRITERION = 'KELLY_CRITERION',    // Size based on win rate and win/loss ratio
  FIXED_RATIO = 'FIXED_RATIO',            // Increase position size based on profit milestones
  FIXED_AMOUNT = 'FIXED_AMOUNT',          // Use same position size for all trades
  VOLATILITY_BASED = 'VOLATILITY_BASED',  // Size based on market volatility
  CUSTOM = 'CUSTOM'                       // Custom sizing algorithm
}

/**
 * Position sizing parameters
 */
export interface PositionSizingParams {
  method: PositionSizingMethod;
  riskPercentage: number; // % of account to risk per trade (0-100)
  maxPositionSize: number; // Maximum position size in base currency
  kellyFraction?: number; // Fraction of Kelly criterion to use (0-1)
  winRate?: number; // Historical win rate for Kelly (0-1)
  customParams?: Record<string, any>; // Additional params for custom methods
}

/**
 * Stop loss types
 */
export enum StopLossType {
  FIXED = 'FIXED',                // Fixed price
  PERCENTAGE = 'PERCENTAGE',      // Percentage from entry
  ATR_MULTIPLE = 'ATR_MULTIPLE',  // Multiple of ATR
  SUPPORT_RESISTANCE = 'SUPPORT_RESISTANCE', // Based on support/resistance
  VOLATILITY_BASED = 'VOLATILITY_BASED',     // Based on volatility
  TRAILING = 'TRAILING',          // Trailing stop
  TIME_BASED = 'TIME_BASED',      // Time-based stop
  MARTINGALE = 'MARTINGALE',      // Increase position on loss
  PYRAMID = 'PYRAMID'             // Decrease position on loss
}

/**
 * Take profit types
 */
export enum TakeProfitType {
  FIXED = 'FIXED',                // Fixed price
  PERCENTAGE = 'PERCENTAGE',      // Percentage from entry
  RISK_REWARD = 'RISK_REWARD',    // Risk-reward ratio
  TRAILING = 'TRAILING',          // Trailing take profit
  PARTIAL = 'PARTIAL',            // Take partial profits at levels
  RESISTANCE_LEVEL = 'RESISTANCE_LEVEL', // At resistance level
  PARABOLIC = 'PARABOLIC',        // Parabolic SAR
  VOLATILITY_BASED = 'VOLATILITY_BASED', // Based on volatility
  PROFIT_TARGET = 'PROFIT_TARGET' // Fixed profit target
}

/**
 * Stop loss configuration
 */
export interface StopLossConfig {
  type: StopLossType;
  value: number; // Interpretation depends on type (price, percentage, ATR multiple, etc.)
  trailingCallback?: number; // For trailing stops, distance to start trailing
  trailingStep?: number; // For trailing stops, the minimum price movement to adjust stop
  timeLimit?: number; // For time-based stops, duration in seconds
  levels?: Array<{price: number, percentage: number}>; // For partial stops
}

/**
 * Take profit configuration
 */
export interface TakeProfitConfig {
  type: TakeProfitType;
  value: number; // Interpretation depends on type
  trailingActivation?: number; // For trailing take profits, price movement to activate
  trailingStep?: number; // For trailing take profits, step size
  levels?: Array<{price: number, percentage: number}>; // For partial take profits
}

/**
 * Risk limit configuration
 */
export interface RiskLimitConfig {
  maxRiskPerTrade: number; // Maximum % of account to risk on a single trade
  maxRiskPerSymbol: number; // Maximum % of account to risk on a single symbol
  maxRiskPerDirection: number; // Maximum % of account to risk in same direction (long/short)
  maxTotalRisk: number; // Maximum % of account to risk across all positions
  maxDrawdown: number; // Maximum allowed drawdown % before halting trading
  maxPositions: number; // Maximum number of open positions
  maxDailyLoss: number; // Maximum daily loss % before halting trading
  cooldownPeriod: number; // Cooldown period in seconds after hitting limits
}

/**
 * Risk management settings for a user or bot
 */
export interface RiskManagementSettings {
  id: string;
  userId: string;
  botId?: string; // Optional, if settings are for a specific bot
  name: string;
  description?: string;
  positionSizing: PositionSizingParams;
  stopLoss: StopLossConfig;
  takeProfit: TakeProfitConfig;
  riskLimits: RiskLimitConfig;
  volatilityLookback: number; // Periods to consider for volatility calculations
  isActive: boolean;
  createdAt: Timestamp;
  updatedAt: Timestamp;
}

/**
 * Risk report for a user's portfolio
 */
export interface RiskReport {
  userId: string;
  timestamp: Timestamp;
  totalBalance: number;
  totalEquity: number;
  totalMargin: number;
  freeMargin: number;
  marginLevel: number; // equity / used margin * 100%
  openPositions: number;
  openPositionsRisk: number; // % of account at risk from open positions
  maxDrawdown: number; // Maximum historical drawdown
  currentDrawdown: number; // Current drawdown from equity peak
  dailyPnL: number; // Daily profit/loss amount
  dailyPnLPercentage: number; // Daily profit/loss as % of balance
  exposureBySymbol: Record<string, number>; // Exposure by symbol (% of account)
  exposureByDirection: Record<string, number>; // Exposure by direction (% of account)
  riskLevel: RiskLevel;
  alerts: RiskAlert[];
}

/**
 * Risk alert types
 */
export enum RiskAlertType {
  MARGIN_CALL = 'MARGIN_CALL',             // Low margin level
  HIGH_EXPOSURE = 'HIGH_EXPOSURE',         // High exposure to an asset
  DRAWDOWN_WARNING = 'DRAWDOWN_WARNING',   // Significant drawdown
  CONCENTRATION_RISK = 'CONCENTRATION_RISK', // Portfolio concentration
  VOLATILITY_SPIKE = 'VOLATILITY_SPIKE',   // Abnormal volatility
  CIRCUIT_BREAKER = 'CIRCUIT_BREAKER',     // Trading halted
  CORRELATION_RISK = 'CORRELATION_RISK',   // High correlation in portfolio
  DAILY_LOSS_WARNING = 'DAILY_LOSS_WARNING', // Significant daily loss
  POSITION_SIZE_WARNING = 'POSITION_SIZE_WARNING', // Large position
  TRADE_FREQUENCY_WARNING = 'TRADE_FREQUENCY_WARNING', // Overtrading
  STOP_DISTANCE_WARNING = 'STOP_DISTANCE_WARNING', // Stop too close/far
  WEEKEND_RISK = 'WEEKEND_RISK',           // Open positions over weekend
  API_CONNECTION_WARNING = 'API_CONNECTION_WARNING', // API issues
  EXTERNAL_EVENT_RISK = 'EXTERNAL_EVENT_RISK', // News, earnings, etc.
  LIQUIDITY_RISK = 'LIQUIDITY_RISK'        // Low liquidity warning
}

/**
 * Risk alert
 */
export interface RiskAlert {
  id: string;
  userId: string;
  type: RiskAlertType;
  level: 'info' | 'warning' | 'critical';
  message: string;
  details: Record<string, any>;
  timestamp: Timestamp;
  acknowledged: boolean;
  resolvedAt?: Timestamp;
}

/**
 * Position sizing calculation request
 */
export interface PositionSizingRequest {
  userId: string;
  botId?: string;
  symbol: string;
  direction: 'long' | 'short';
  entryPrice: number;
  stopLossPrice?: number;
  stopLossPercentage?: number;
  riskAmount?: number; // Optional specific risk amount override
  confidence?: number; // Signal confidence score (0-100)
  volatility?: number; // Optional market volatility metric
}

/**
 * Position sizing calculation result
 */
export interface PositionSizingResult {
  positionSize: number; // Size in base currency units
  maxPositionSize: number; // Maximum allowed position size
  riskAmount: number; // Amount at risk in account currency
  riskPercentage: number; // Percentage of account at risk
  leverage: number; // Recommended leverage
  margin: number; // Required margin
  potentialLoss: number; // Potential loss amount
  potentialProfit: number; // Potential profit amount (if take profit set)
  riskRewardRatio: number; // Risk/reward ratio
  adjustedForRisk: boolean; // Whether position was adjusted for risk limits
  warnings: string[]; // Any warnings about the position size
}

/**
 * Trade risk analysis
 */
export interface TradeRiskAnalysis {
  tradeId: string;
  symbol: string;
  direction: 'long' | 'short';
  entryPrice: number;
  currentPrice: number;
  positionSize: number;
  exposure: number; // % of account
  riskAmount: number; // Amount at risk
  riskPercentage: number; // % of account at risk
  stopLossPrice: number;
  takeProfitPrice?: number;
  unrealizedPnL: number;
  unrealizedPnLPercentage: number;
  distanceToStopLoss: number; // % to stop loss
  distanceToTakeProfit?: number; // % to take profit
  breakEvenPrice: number;
  riskRewardRatio: number;
  timeInTrade: number; // seconds
  remainingTimeLimit?: number; // seconds (if time-based exit)
  margin: number;
  liquidationPrice?: number;
  riskScore: number; // Normalized risk score (0-100)
  riskLevel: RiskLevel;
}

/**
 * Portfolio risk metrics
 */
export interface PortfolioRiskMetrics {
  userId: string;
  timestamp: Timestamp;
  totalEquity: number;
  peakEquity: number;
  drawdown: number; // Current drawdown percentage
  maxDrawdown: number; // Maximum historical drawdown
  sharpeRatio: number;
  sortinoRatio: number;
  calmarRatio: number;
  valueAtRisk: number; // VaR (95%)
  expectedShortfall: number; // CVaR/Expected Shortfall
  dailyVolatility: number; // Standard deviation of daily returns
  betaToMarket: number; // Portfolio beta
  correlationMatrix: Record<string, Record<string, number>>; // Correlation between positions
  stressTestResults: Record<string, number>; // Results from stress test scenarios
  concentrationRisk: number; // Measure of portfolio concentration
  riskContributionByPosition: Record<string, number>; // Risk contribution % by position
}

/**
 * Circuit breaker conditions
 */
export interface CircuitBreakerConfig {
  enabled: boolean;
  maxDailyLoss: number; // % of account
  maxDrawdown: number; // % from peak
  volatilityMultiplier: number; // Trigger on x times normal volatility
  consecutiveLosses: number; // Number of consecutive losses
  tradingPause: number; // Pause duration in seconds
  marketWideEnabled: boolean; // React to market-wide circuit breakers
  enableManualOverride: boolean; // Allow manual reset
}

/**
 * Risk settings
 */
export interface RiskSettings {
  id: string;                 // Settings ID
  userId: string;             // User ID
  botId?: string;             // Optional Bot ID
  name: string;               // Settings name
  description?: string;       // Settings description
  isActive: boolean;          // Whether settings are active
  
  // Position sizing
  positionSizingMethod: PositionSizingMethod; // Position sizing method
  riskPercentage: number;     // Risk percentage per trade
  maxPositionSize: number;    // Maximum position size
  kellyFraction?: number;     // Kelly criterion fraction
  winRate?: number;           // Historical win rate
  customSizingParams?: Record<string, any>; // Custom sizing parameters
  
  // Stop loss
  stopLossType: StopLossType; // Stop loss type
  stopLossValue: number;      // Stop loss value
  trailingCallback?: number;  // Trailing stop callback
  trailingStep?: number;      // Trailing stop step
  timeLimit?: number;         // Time limit for time-based stops
  stopLossLevels?: any[];     // Stop loss levels for partial stops
  
  // Take profit
  takeProfitType: TakeProfitType; // Take profit type
  takeProfitValue: number;    // Take profit value
  trailingActivation?: number; // Trailing take profit activation
  takeProfitLevels?: any[];   // Take profit levels for partial profits
  
  // Risk limits
  maxRiskPerTrade: number;    // Maximum risk per trade
  maxRiskPerSymbol: number;   // Maximum risk per symbol
  maxRiskPerDirection: number; // Maximum risk per direction
  maxTotalRisk: number;       // Maximum total risk
  maxDrawdown: number;        // Maximum drawdown
  maxPositions: number;       // Maximum open positions
  maxDailyLoss: number;       // Maximum daily loss
  cooldownPeriod: number;     // Cooldown period after loss
  
  // Volatility settings
  volatilityLookback: number; // Volatility lookback periods
  
  // Circuit breaker
  circuitBreakerEnabled: boolean; // Circuit breaker enabled
  maxDailyLossBreaker: number; // Max daily loss for circuit breaker
  maxDrawdownBreaker: number;  // Max drawdown for circuit breaker
  volatilityMultiplier: number; // Volatility multiplier for circuit breaker
  consecutiveLossesBreaker: number; // Consecutive losses for circuit breaker
  tradingPause: number;       // Trading pause duration in seconds
  marketWideEnabled: boolean; // Market-wide circuit breaker enabled
  enableManualOverride: boolean; // Manual override enabled
  
  // Timestamps
  createdAt: string;          // Creation timestamp
  updatedAt: string;          // Update timestamp
} 