/**
 * Signal Generation Service
 * Converts ML model predictions into actionable trading signals
 */
import { TradingSignal, SignalGenerationOptions, SignalFilterCriteria } from '../../types/signals';
/**
 * Signal Generation Service class
 * Provides methods to generate and manage trading signals
 */
export declare class SignalGenerationService {
    private options;
    /**
     * Creates a new Signal Generation Service instance
     * @param options - Signal generation options
     */
    constructor(options?: Partial<SignalGenerationOptions>);
    /**
     * Generate trading signals for a specific symbol
     * @param symbol - Trading pair symbol
     * @param features - Current market features
     * @param options - Optional signal generation options to override defaults
     * @returns Generated trading signals
     */
    generateSignals(symbol: string, features: Record<string, number>, options?: Partial<SignalGenerationOptions>): Promise<TradingSignal[]>;
    /**
     * Process enhanced prediction into trading signals
     * @private
     * @param prediction - Enhanced ML model prediction
     * @param features - Current market features
     * @returns Processed trading signals
     */
    private _processEnhancedPrediction;
    /**
     * Get regime-based multiplier for target calculation
     * @private
     * @param regime - Market regime
     * @returns Multiplier value
     */
    private _getRegimeMultiplier;
    /**
     * Get timeframe based on market regime
     * @private
     * @param regime - Market regime
     * @returns Signal timeframe
     */
    private _getTimeframeFromRegime;
    /**
     * Process model prediction into trading signals
     * @private
     * @param prediction - ML model prediction
     * @param features - Current market features
     * @returns Processed trading signals
     */
    private _processModelPrediction;
    /**
     * Calculate signal expiry time based on timeframe
     * @private
     * @param timeframe - Signal timeframe
     * @returns Expiry timestamp
     */
    private _calculateExpiryTime;
    /**
     * Validate signals using additional techniques
     * @private
     * @param signals - Signals to validate
     * @returns Validated signals
     */
    private _validateSignals;
    /**
     * Filter signals based on confidence threshold and other criteria
     * @private
     * @param signals - Signals to filter
     * @param options - Signal generation options
     * @returns Filtered signals
     */
    private _filterSignals;
    /**
     * Store signals in the database
     * @private
     * @param signals - Signals to store
     */
    private _storeSignals;
    /**
     * Get signals based on filter criteria
     * @param criteria - Filter criteria
     * @returns Filtered signals
     */
    getSignals(criteria?: SignalFilterCriteria): Promise<TradingSignal[]>;
    /**
     * Get the latest signal for a symbol
     * @param symbol - Trading pair symbol
     * @returns Latest signal or null if none found
     */
    getLatestSignal(symbol: string): Promise<TradingSignal | null>;
}
declare const signalGenerationService: SignalGenerationService;
export default signalGenerationService;
