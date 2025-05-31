/**
 * Bridge Service
 * Integrates ML and Trading systems
 */
import { BacktestRequest, BridgeHealth, TrainingRequest, ModelStatus } from '../../types/bridge';
import { TradingSignal } from '../../types/signals';
/**
 * Bridge Service class
 * Connects ML and Trading systems
 */
declare class BridgeService {
    private healthStatus;
    private latencyHistory;
    /**
     * Creates a new Bridge Service instance
     */
    constructor();
    /**
     * Get model prediction and generate trading signal
     * @param symbol - Trading symbol
     * @param timeframe - Timeframe
     * @param options - Additional options
     * @returns Generated trading signal
     */
    getPredictionAndGenerateSignal(symbol: string, timeframe: string, options?: {
        modelVersion?: string;
        confidenceThreshold?: number;
        signalExpiry?: number;
    }): Promise<TradingSignal>;
    /**
     * Convert ML prediction to trading signal
     * @param prediction - ML prediction
     * @param options - Additional options
     * @returns Trading signal
     */
    private convertPredictionToSignal;
    /**
     * Temporary method to get current price
     * This should be replaced with a proper market data service
     * @param symbol - Symbol
     * @returns Current price
     */
    private getCurrentPrice;
    /**
     * Calculate stop loss based on current price and direction
     * @param currentPrice - Current price
     * @param direction - Signal direction
     * @returns Stop loss price
     */
    private calculateStopLoss;
    /**
     * Run backtest using ML predictions
     * @param request - Backtest request
     * @returns Backtest result
     */
    runBacktest(request: BacktestRequest): Promise<any>;
    /**
     * Get system health status
     * @returns Health status
     */
    getHealth(): Promise<BridgeHealth>;
    /**
     * Start model training
     * @param request - Training request
     * @returns Training status
     */
    startModelTraining(request: TrainingRequest): Promise<any>;
    /**
     * Get available models
     * @returns Array of model status
     */
    getAvailableModels(): Promise<ModelStatus[]>;
    /**
     * Start health monitoring
     * @private
     */
    private startHealthMonitoring;
    /**
     * Update latency averages
     * @private
     */
    private updateLatencyAverages;
    /**
     * Calculate average of an array
     * @private
     * @param arr - Array of numbers
     * @returns Average
     */
    private calculateAverage;
    /**
     * Record error in health status
     * @private
     * @param component - Component name
     * @param message - Error message
     */
    private recordError;
}
declare const bridgeService: BridgeService;
export default bridgeService;
