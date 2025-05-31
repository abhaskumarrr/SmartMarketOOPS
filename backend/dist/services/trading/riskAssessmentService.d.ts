/**
 * Risk Assessment Service
 * Handles portfolio risk analysis, monitoring, and alerts
 */
import { RiskAlertType, TradeRiskAnalysis, RiskReport, RiskAlert } from '../../types/risk';
/**
 * Risk Assessment Service class
 * Provides methods for analyzing and monitoring trading risk
 */
export declare class RiskAssessmentService {
    private equityHighWatermark;
    /**
     * Creates a new Risk Assessment Service instance
     */
    constructor();
    /**
     * Generate a comprehensive risk report for a user
     * @param userId - User ID
     * @returns Risk report
     */
    generateRiskReport(userId: string): Promise<RiskReport>;
    /**
     * Analyze risk for a specific trade/position
     * @param userId - User ID
     * @param positionId - Position ID
     * @returns Trade risk analysis
     */
    analyzeTradeRisk(userId: string, positionId: string): Promise<TradeRiskAnalysis>;
    /**
     * Create a risk alert
     * @param userId - User ID
     * @param type - Alert type
     * @param level - Alert level
     * @param message - Alert message
     * @param details - Alert details
     * @returns Created alert
     */
    createAlert(userId: string, type: RiskAlertType, level: 'info' | 'warning' | 'critical', message: string, details: Record<string, any>): Promise<RiskAlert>;
    /**
     * Get account balance for a user
     * @private
     * @param userId - User ID
     * @returns Account balance
     */
    private _getAccountBalance;
    /**
     * Get account equity for a user
     * @private
     * @param userId - User ID
     * @returns Account equity
     */
    private _getAccountEquity;
    /**
     * Get open positions for a user
     * @private
     * @param userId - User ID
     * @returns Open positions
     */
    private _getOpenPositions;
    /**
     * Calculate total margin used
     * @private
     * @param positions - Open positions
     * @returns Total margin
     */
    private _calculateTotalMargin;
    /**
     * Calculate exposure by symbol and direction
     * @private
     * @param positions - Open positions
     * @param accountBalance - Account balance
     * @returns Exposure details
     */
    private _calculateExposure;
    /**
     * Calculate drawdown
     * @private
     * @param userId - User ID
     * @param currentEquity - Current equity
     * @returns Drawdown details
     */
    private _calculateDrawdown;
    /**
     * Calculate daily P&L
     * @private
     * @param userId - User ID
     * @param accountBalance - Account balance
     * @returns Daily P&L details
     */
    private _calculateDailyPnL;
    /**
     * Generate risk alerts
     * @private
     * @param userId - User ID
     * @param metrics - Risk metrics
     * @returns Risk alerts
     */
    private _generateAlerts;
    /**
     * Determine overall risk level
     * @private
     * @param metrics - Risk metrics
     * @returns Risk level
     */
    private _determineRiskLevel;
    /**
     * Determine position risk level
     * @private
     * @param riskScore - Risk score
     * @returns Risk level
     */
    private _determinePositionRiskLevel;
}
declare const riskAssessmentService: RiskAssessmentService;
export default riskAssessmentService;
