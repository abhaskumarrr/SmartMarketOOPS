/**
 * AI Position Manager
 * Manages Delta Exchange positions using AI-powered dynamic take profit system
 */
import { DeltaExchangeUnified } from './DeltaExchangeUnified';
import DeltaExchangeAPI from './deltaApiService';
export interface ManagedPosition {
    id: string;
    symbol: string;
    side: 'LONG' | 'SHORT';
    size: number;
    entryPrice: number;
    currentPrice: number;
    unrealizedPnl: number;
    takeProfitLevels: any[];
    stopLoss: number;
    partialExits: PartialExit[];
    status: 'ACTIVE' | 'MANAGING' | 'CLOSED';
    lastUpdate: number;
    aiRecommendations: string[];
}
export interface PartialExit {
    level: number;
    percentage: number;
    targetPrice: number;
    executed: boolean;
    executedAt?: number;
    orderId?: string;
    pnl?: number;
}
export interface AIAnalysis {
    action: 'HOLD' | 'PARTIAL_EXIT' | 'FULL_EXIT' | 'ADJUST_STOP' | 'TRAIL_STOP';
    confidence: number;
    reasoning: string;
    targetPrice?: number;
    percentage?: number;
    newStopLoss?: number;
}
export interface EnhancedPositionHealth {
    score: number;
    trend_alignment: number;
    momentum_score: number;
    risk_adjusted_return: number;
    volatility_factor: number;
    regime_compatibility: number;
    ml_prediction: {
        outcome_probability: number;
        expected_return: number;
        time_to_target: number;
        confidence: number;
    };
    factors: {
        multi_timeframe_alignment: number;
        position_age_factor: number;
        pnl_momentum: number;
        market_regime_score: number;
        volume_confirmation: number;
    };
    recommendations: {
        action: 'HOLD' | 'SCALE_IN' | 'SCALE_OUT' | 'CLOSE' | 'TRAIL_STOP';
        urgency: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
        reasoning: string[];
        optimal_exit_price?: number;
        risk_level: 'LOW' | 'MEDIUM' | 'HIGH';
    };
}
export declare class AIPositionManager {
    private deltaApi;
    private deltaUnified;
    private takeProfitManager;
    private mtfAnalyzer;
    private regimeDetector;
    private managedPositions;
    private isRunning;
    private updateInterval;
    constructor(deltaApi: DeltaExchangeAPI, deltaUnified: DeltaExchangeUnified);
    /**
     * Start AI position management
     */
    startManagement(): Promise<void>;
    /**
     * Stop AI position management
     */
    stopManagement(): void;
    /**
     * Scan and manage all positions
     */
    private scanAndManagePositions;
    /**
     * Manage individual position with AI
     */
    private managePosition;
    /**
     * Initialize AI management for new position
     */
    private initializePositionManagement;
    /**
     * Get current price for symbol
     */
    private getCurrentPrice;
    /**
     * Get enhanced position health analysis
     */
    private getEnhancedPositionHealth;
    /**
     * Get AI analysis for position (enhanced version)
     */
    private getAIAnalysis;
    /**
     * Original AI analysis method (fallback)
     */
    private getBasicAIAnalysis;
    /**
     * Execute AI recommendation
     */
    private executeAIRecommendation;
    /**
     * Execute partial exit
     */
    private executePartialExit;
    /**
     * Execute full exit
     */
    private executeFullExit;
    /**
     * Update stop loss
     */
    private updateStopLoss;
    /**
     * Calculate partial P&L
     */
    private calculatePartialPnl;
    /**
     * Display management summary
     */
    private displayManagementSummary;
    /**
     * Get managed positions
     */
    getManagedPositions(): ManagedPosition[];
    /**
     * Calculate trend alignment across timeframes
     */
    private calculateTrendAlignment;
    /**
     * Calculate momentum score
     */
    private calculateMomentumScore;
    /**
     * Calculate risk-adjusted return
     */
    private calculateRiskAdjustedReturn;
    /**
     * Calculate regime compatibility
     */
    private calculateRegimeCompatibility;
    /**
     * Generate ML prediction
     */
    private generateMLPrediction;
    /**
     * Calculate age factor
     */
    private calculateAgeFactor;
    /**
     * Calculate P&L momentum
     */
    private calculatePnLMomentum;
    /**
     * Calculate volume confirmation
     */
    private calculateVolumeConfirmation;
    /**
     * Calculate enhanced health score
     */
    private calculateEnhancedHealthScore;
    /**
     * Generate enhanced recommendations
     */
    private generateEnhancedRecommendations;
    /**
     * Calculate optimal exit price
     */
    private calculateOptimalExitPrice;
    /**
     * Get exit percentage based on action and score
     */
    private getExitPercentage;
    /**
     * Calculate dynamic stop loss
     */
    private calculateDynamicStopLoss;
    /**
     * Fallback basic position health
     */
    private getBasicPositionHealth;
}
