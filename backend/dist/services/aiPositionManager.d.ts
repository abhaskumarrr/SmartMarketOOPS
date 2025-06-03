/**
 * AI Position Manager
 * Manages Delta Exchange positions using AI-powered dynamic take profit system
 */
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
export declare class AIPositionManager {
    private deltaApi;
    private takeProfitManager;
    private managedPositions;
    private isRunning;
    private updateInterval;
    constructor(deltaApi: DeltaExchangeAPI);
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
     * Get AI analysis for position
     */
    private getAIAnalysis;
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
}
