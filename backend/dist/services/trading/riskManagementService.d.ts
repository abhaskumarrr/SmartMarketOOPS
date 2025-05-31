/**
 * Risk Management Service
 * Handles position sizing, risk assessment, and protection mechanisms
 */
import { PositionSizingRequest, PositionSizingResult } from '../../types/risk';
/**
 * Risk Management Service class
 * Provides methods for managing trading risk
 */
export declare class RiskManagementService {
    /**
     * Creates a new Risk Management Service instance
     */
    constructor();
    /**
     * Get risk settings for a user or bot
     * @param userId - User ID
     * @param botId - Optional Bot ID
     * @returns Risk settings or default settings if none found
     */
    getRiskSettings(userId: string, botId?: string): Promise<any>;
    /**
     * Create default risk settings for a user or bot
     * @param userId - User ID
     * @param botId - Optional Bot ID
     * @returns Default risk settings
     */
    getDefaultRiskSettings(userId: string, botId?: string): Promise<any>;
    /**
     * Create or update risk settings
     * @param settings - Risk settings to create or update
     * @returns Created or updated risk settings
     */
    saveRiskSettings(settings: any): Promise<any>;
    /**
     * Calculate position size based on risk parameters
     * @param request - Position sizing request
     * @returns Position sizing result
     */
    calculatePositionSize(request: PositionSizingRequest): Promise<PositionSizingResult>;
    /**
     * Get account balance for a user
     * @private
     * @param userId - User ID
     * @returns Account balance
     */
    private _getAccountBalance;
    /**
     * Get open positions for a user
     * @private
     * @param userId - User ID
     * @returns Open positions
     */
    private _getOpenPositions;
    /**
     * Calculate existing risk exposure
     * @private
     * @param positions - Open positions
     * @param accountBalance - Account balance
     * @returns Risk exposure details
     */
    private _calculateExistingRisk;
    /**
     * Calculate risk for a single position
     * @private
     * @param position - Position
     * @param accountBalance - Account balance
     * @returns Risk percentage
     */
    private _calculatePositionRisk;
}
declare const riskManagementService: RiskManagementService;
export default riskManagementService;
