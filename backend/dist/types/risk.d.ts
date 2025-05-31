/**
 * Risk Management Types
 * Definitions for the risk management system
 */
import { Timestamp } from './common';
/**
 * Risk level enum for portfolio and positions
 */
export declare enum RiskLevel {
    VERY_LOW = "VERY_LOW",// 0-20% of max allowed risk
    LOW = "LOW",// 20-40% of max allowed risk
    MODERATE = "MODERATE",// 40-60% of max allowed risk
    HIGH = "HIGH",// 60-80% of max allowed risk
    VERY_HIGH = "VERY_HIGH"
}
/**
 * Position sizing methods
 */
export declare enum PositionSizingMethod {
    FIXED_FRACTIONAL = "FIXED_FRACTIONAL",// Risk a fixed percentage of account on each trade
    KELLY_CRITERION = "KELLY_CRITERION",// Size based on win rate and win/loss ratio
    FIXED_RATIO = "FIXED_RATIO",// Increase position size based on profit milestones
    FIXED_AMOUNT = "FIXED_AMOUNT",// Use same position size for all trades
    VOLATILITY_BASED = "VOLATILITY_BASED",// Size based on market volatility
    CUSTOM = "CUSTOM"
}
/**
 * Position sizing parameters
 */
export interface PositionSizingParams {
    method: PositionSizingMethod;
    riskPercentage: number;
    maxPositionSize: number;
    kellyFraction?: number;
    winRate?: number;
    customParams?: Record<string, any>;
}
/**
 * Stop loss types
 */
export declare enum StopLossType {
    FIXED = "FIXED",// Fixed price
    PERCENTAGE = "PERCENTAGE",// Percentage from entry
    ATR_MULTIPLE = "ATR_MULTIPLE",// Multiple of ATR
    SUPPORT_RESISTANCE = "SUPPORT_RESISTANCE",// Based on support/resistance
    VOLATILITY_BASED = "VOLATILITY_BASED",// Based on volatility
    TRAILING = "TRAILING",// Trailing stop
    TIME_BASED = "TIME_BASED",// Time-based stop
    MARTINGALE = "MARTINGALE",// Increase position on loss
    PYRAMID = "PYRAMID"
}
/**
 * Take profit types
 */
export declare enum TakeProfitType {
    FIXED = "FIXED",// Fixed price
    PERCENTAGE = "PERCENTAGE",// Percentage from entry
    RISK_REWARD = "RISK_REWARD",// Risk-reward ratio
    TRAILING = "TRAILING",// Trailing take profit
    PARTIAL = "PARTIAL",// Take partial profits at levels
    RESISTANCE_LEVEL = "RESISTANCE_LEVEL",// At resistance level
    PARABOLIC = "PARABOLIC",// Parabolic SAR
    VOLATILITY_BASED = "VOLATILITY_BASED",// Based on volatility
    PROFIT_TARGET = "PROFIT_TARGET"
}
/**
 * Stop loss configuration
 */
export interface StopLossConfig {
    type: StopLossType;
    value: number;
    trailingCallback?: number;
    trailingStep?: number;
    timeLimit?: number;
    levels?: Array<{
        price: number;
        percentage: number;
    }>;
}
/**
 * Take profit configuration
 */
export interface TakeProfitConfig {
    type: TakeProfitType;
    value: number;
    trailingActivation?: number;
    trailingStep?: number;
    levels?: Array<{
        price: number;
        percentage: number;
    }>;
}
/**
 * Risk limit configuration
 */
export interface RiskLimitConfig {
    maxRiskPerTrade: number;
    maxRiskPerSymbol: number;
    maxRiskPerDirection: number;
    maxTotalRisk: number;
    maxDrawdown: number;
    maxPositions: number;
    maxDailyLoss: number;
    cooldownPeriod: number;
}
/**
 * Risk management settings for a user or bot
 */
export interface RiskManagementSettings {
    id: string;
    userId: string;
    botId?: string;
    name: string;
    description?: string;
    positionSizing: PositionSizingParams;
    stopLoss: StopLossConfig;
    takeProfit: TakeProfitConfig;
    riskLimits: RiskLimitConfig;
    volatilityLookback: number;
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
    marginLevel: number;
    openPositions: number;
    openPositionsRisk: number;
    maxDrawdown: number;
    currentDrawdown: number;
    dailyPnL: number;
    dailyPnLPercentage: number;
    exposureBySymbol: Record<string, number>;
    exposureByDirection: Record<string, number>;
    riskLevel: RiskLevel;
    alerts: RiskAlert[];
}
/**
 * Risk alert types
 */
export declare enum RiskAlertType {
    MARGIN_CALL = "MARGIN_CALL",// Low margin level
    HIGH_EXPOSURE = "HIGH_EXPOSURE",// High exposure to an asset
    DRAWDOWN_WARNING = "DRAWDOWN_WARNING",// Significant drawdown
    CONCENTRATION_RISK = "CONCENTRATION_RISK",// Portfolio concentration
    VOLATILITY_SPIKE = "VOLATILITY_SPIKE",// Abnormal volatility
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER",// Trading halted
    CORRELATION_RISK = "CORRELATION_RISK",// High correlation in portfolio
    DAILY_LOSS_WARNING = "DAILY_LOSS_WARNING",// Significant daily loss
    POSITION_SIZE_WARNING = "POSITION_SIZE_WARNING",// Large position
    TRADE_FREQUENCY_WARNING = "TRADE_FREQUENCY_WARNING",// Overtrading
    STOP_DISTANCE_WARNING = "STOP_DISTANCE_WARNING",// Stop too close/far
    WEEKEND_RISK = "WEEKEND_RISK",// Open positions over weekend
    API_CONNECTION_WARNING = "API_CONNECTION_WARNING",// API issues
    EXTERNAL_EVENT_RISK = "EXTERNAL_EVENT_RISK",// News, earnings, etc.
    LIQUIDITY_RISK = "LIQUIDITY_RISK"
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
    riskAmount?: number;
    confidence?: number;
    volatility?: number;
}
/**
 * Position sizing calculation result
 */
export interface PositionSizingResult {
    positionSize: number;
    maxPositionSize: number;
    riskAmount: number;
    riskPercentage: number;
    leverage: number;
    margin: number;
    potentialLoss: number;
    potentialProfit: number;
    riskRewardRatio: number;
    adjustedForRisk: boolean;
    warnings: string[];
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
    exposure: number;
    riskAmount: number;
    riskPercentage: number;
    stopLossPrice: number;
    takeProfitPrice?: number;
    unrealizedPnL: number;
    unrealizedPnLPercentage: number;
    distanceToStopLoss: number;
    distanceToTakeProfit?: number;
    breakEvenPrice: number;
    riskRewardRatio: number;
    timeInTrade: number;
    remainingTimeLimit?: number;
    margin: number;
    liquidationPrice?: number;
    riskScore: number;
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
    drawdown: number;
    maxDrawdown: number;
    sharpeRatio: number;
    sortinoRatio: number;
    calmarRatio: number;
    valueAtRisk: number;
    expectedShortfall: number;
    dailyVolatility: number;
    betaToMarket: number;
    correlationMatrix: Record<string, Record<string, number>>;
    stressTestResults: Record<string, number>;
    concentrationRisk: number;
    riskContributionByPosition: Record<string, number>;
}
/**
 * Circuit breaker conditions
 */
export interface CircuitBreakerConfig {
    enabled: boolean;
    maxDailyLoss: number;
    maxDrawdown: number;
    volatilityMultiplier: number;
    consecutiveLosses: number;
    tradingPause: number;
    marketWideEnabled: boolean;
    enableManualOverride: boolean;
}
/**
 * Risk settings
 */
export interface RiskSettings {
    id: string;
    userId: string;
    botId?: string;
    name: string;
    description?: string;
    isActive: boolean;
    positionSizingMethod: PositionSizingMethod;
    riskPercentage: number;
    maxPositionSize: number;
    kellyFraction?: number;
    winRate?: number;
    customSizingParams?: Record<string, any>;
    stopLossType: StopLossType;
    stopLossValue: number;
    trailingCallback?: number;
    trailingStep?: number;
    timeLimit?: number;
    stopLossLevels?: any[];
    takeProfitType: TakeProfitType;
    takeProfitValue: number;
    trailingActivation?: number;
    takeProfitLevels?: any[];
    maxRiskPerTrade: number;
    maxRiskPerSymbol: number;
    maxRiskPerDirection: number;
    maxTotalRisk: number;
    maxDrawdown: number;
    maxPositions: number;
    maxDailyLoss: number;
    cooldownPeriod: number;
    volatilityLookback: number;
    circuitBreakerEnabled: boolean;
    maxDailyLossBreaker: number;
    maxDrawdownBreaker: number;
    volatilityMultiplier: number;
    consecutiveLossesBreaker: number;
    tradingPause: number;
    marketWideEnabled: boolean;
    enableManualOverride: boolean;
    createdAt: string;
    updatedAt: string;
}
