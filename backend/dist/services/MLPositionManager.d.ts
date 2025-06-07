/**
 * ML-Powered Position Management System
 * Advanced position management using ML models for dynamic stop/take profit optimization,
 * position sizing, and exit timing prediction
 */
import { TradingDecision } from './EnhancedTradingDecisionEngine';
export interface Position {
    id: string;
    symbol: string;
    side: 'long' | 'short';
    entryPrice: number;
    currentPrice: number;
    quantity: number;
    leverage: number;
    stopLoss: number;
    takeProfit: number;
    trailingStop?: number;
    exitProbability: number;
    optimalExitPrice: number;
    riskScore: number;
    unrealizedPnL: number;
    maxDrawdown: number;
    maxProfit: number;
    holdingTime: number;
    entryTimestamp: number;
    lastUpdate: number;
    decisionId: string;
}
export interface PositionManagerConfig {
    exitPredictionThreshold: number;
    riskAdjustmentFactor: number;
    trailingStopEnabled: boolean;
    trailingStopDistance: number;
    maxStopLossAdjustment: number;
    dynamicTakeProfitEnabled: boolean;
    profitLockingThreshold: number;
    maxTakeProfitExtension: number;
    maxPositionAdjustment: number;
    riskBasedSizing: boolean;
    holdTimeOptimization: boolean;
    maxHoldTime: number;
    minHoldTime: number;
}
export interface PositionTrainingData {
    features: number[];
    exitPrice: number;
    exitTime: number;
    profitLoss: number;
    wasOptimal: boolean;
}
export declare class MLPositionManager {
    private decisionEngine;
    private dataIntegration;
    private tradingBot;
    private redis;
    private activePositions;
    private positionHistory;
    private trainingData;
    private config;
    private performanceMetrics;
    constructor();
    /**
     * Initialize ML Position Manager
     */
    initialize(): Promise<void>;
    /**
     * Create new position from trading decision
     */
    createPosition(decision: TradingDecision, currentPrice: number): Promise<Position | null>;
    /**
     * Update position with current market data and ML predictions
     */
    updatePosition(positionId: string, currentPrice: number): Promise<Position | null>;
    /**
     * Check if position should be closed based on ML predictions
     */
    shouldClosePosition(positionId: string): Promise<{
        shouldClose: boolean;
        reason: string;
        urgency: 'low' | 'medium' | 'high';
    }>;
    /**
     * Close position and record training data
     */
    closePosition(positionId: string, exitPrice: number, reason: string): Promise<boolean>;
    /**
     * Get all active positions
     */
    getActivePositions(): Position[];
    /**
     * Get position by ID
     */
    getPosition(positionId: string): Position | null;
    /**
     * Get performance metrics
     */
    getPerformanceMetrics(): any;
    /**
     * Update configuration
     */
    updateConfiguration(newConfig: Partial<PositionManagerConfig>): void;
    /**
     * Cleanup resources
     */
    cleanup(): Promise<void>;
    /**
     * Calculate position quantity based on decision and current price
     */
    private calculatePositionQuantity;
    /**
     * Calculate unrealized P&L for position
     */
    private calculateUnrealizedPnL;
    /**
     * Calculate realized P&L at exit
     */
    private calculateRealizedPnL;
    /**
     * Extract ML features for position management (45 features)
     */
    private extractPositionFeatures;
    /**
     * Update ML predictions for position
     */
    private updateMLPredictions;
    /**
     * Apply dynamic position management based on ML insights
     */
    private applyDynamicManagement;
    /**
     * Update trailing stop based on current price movement
     */
    private updateTrailingStop;
    /**
     * Update dynamic take profit based on ML predictions
     */
    private updateDynamicTakeProfit;
    /**
     * Update stop loss based on risk assessment
     */
    private updateRiskBasedStopLoss;
    /**
     * Check if stop loss is hit
     */
    private isStopLossHit;
    /**
     * Check if take profit is hit
     */
    private isTakeProfitHit;
    /**
     * Predict exit probability using ML features
     */
    private predictExitProbability;
    /**
     * Predict optimal exit price using ML features
     */
    private predictOptimalExitPrice;
    /**
     * Predict risk score using ML features
     */
    private predictRiskScore;
    /**
     * Record training data for ML model improvement
     */
    private recordTrainingData;
    /**
     * Determine if exit was optimal
     */
    private wasExitOptimal;
    /**
     * Update performance metrics
     */
    private updatePerformanceMetrics;
    /**
     * Save position to Redis
     */
    private savePositionToRedis;
    /**
     * Load active positions from Redis
     */
    private loadActivePositions;
    /**
     * Load training data from Redis
     */
    private loadTrainingData;
    /**
     * Save training data to Redis
     */
    private saveTrainingData;
    /**
     * Get time of day feature (0-1)
     */
    private getTimeOfDayFeature;
    /**
     * Get market session feature
     */
    private getMarketSessionFeature;
}
