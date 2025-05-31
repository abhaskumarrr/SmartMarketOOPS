/**
 * Circuit Breaker Service
 * Implements circuit breaker patterns to halt trading in abnormal conditions
 */
/**
 * Circuit breaker status
 */
interface CircuitBreakerStatus {
    userId: string;
    botId?: string;
    isTripped: boolean;
    reason?: string;
    trippedAt?: Date;
    cooldownUntil?: Date;
    resetable: boolean;
}
/**
 * Circuit Breaker Service class
 * Handles trading halt mechanisms during abnormal conditions
 */
export declare class CircuitBreakerService {
    private circuitBreakerStatus;
    private consecutiveLosses;
    private marketVolatility;
    /**
     * Creates a new Circuit Breaker Service instance
     */
    constructor();
    /**
     * Check if trading is allowed
     * @param userId - User ID
     * @param botId - Optional Bot ID
     * @param symbol - Trading symbol
     * @returns Object with allowed status and reason if not allowed
     */
    isTradingAllowed(userId: string, botId?: string, symbol?: string): Promise<{
        allowed: boolean;
        reason?: string;
    }>;
    /**
     * Record a trade result (win/loss) for consecutive loss tracking
     * @param userId - User ID
     * @param isWin - Whether the trade was a win
     * @param botId - Optional Bot ID
     */
    recordTradeResult(userId: string, isWin: boolean, botId?: string): void;
    /**
     * Update market volatility for a symbol
     * @param symbol - Trading symbol
     * @param volatilityValue - Volatility percentage
     */
    updateMarketVolatility(symbol: string, volatilityValue: number): void;
    /**
     * Manually trip circuit breaker
     * @param userId - User ID
     * @param reason - Reason for tripping
     * @param botId - Optional Bot ID
     * @returns Trip success status
     */
    manuallyTripCircuitBreaker(userId: string, reason?: string, botId?: string): Promise<boolean>;
    /**
     * Reset circuit breaker
     * @param userId - User ID
     * @param botId - Optional Bot ID
     * @returns Reset success status
     */
    resetCircuitBreaker(userId: string, botId?: string): Promise<boolean>;
    /**
     * Get circuit breaker status
     * @param userId - User ID
     * @param botId - Optional Bot ID
     * @returns Circuit breaker status
     */
    getCircuitBreakerStatus(userId: string, botId?: string): CircuitBreakerStatus | null;
    /**
     * Trip circuit breaker
     * @private
     * @param userId - User ID
     * @param reason - Reason for tripping
     * @param botId - Optional Bot ID
     */
    private _tripCircuitBreaker;
    /**
     * Check if market-wide circuit breaker is in effect
     * @private
     * @param symbol - Trading symbol
     * @returns Whether market-wide circuit breaker is in effect
     */
    private _checkMarketWideCircuitBreaker;
    /**
     * Get risk settings
     * @private
     * @param userId - User ID
     * @param botId - Optional Bot ID
     * @returns Risk settings
     */
    private _getRiskSettings;
    /**
     * Get normal volatility for a symbol
     * @private
     * @param symbol - Trading symbol
     * @returns Normal volatility percentage
     */
    private _getNormalVolatility;
    /**
     * Get circuit breaker ID
     * @private
     * @param userId - User ID
     * @param botId - Optional Bot ID
     * @returns Circuit breaker ID
     */
    private _getCircuitBreakerId;
    /**
     * Clean up expired circuit breakers
     * @private
     */
    private _cleanupExpiredCircuitBreakers;
}
declare const circuitBreakerService: CircuitBreakerService;
export default circuitBreakerService;
