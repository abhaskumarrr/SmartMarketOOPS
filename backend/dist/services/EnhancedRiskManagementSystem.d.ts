/**
 * Enhanced Risk Management System
 * Advanced risk controls with dynamic position sizing, circuit breakers, and failsafe mechanisms
 * Optimized for high-leverage trading with extreme market condition protection
 */
import { TradingDecision } from './EnhancedTradingDecisionEngine';
export interface RiskMetrics {
    totalExposure: number;
    leverageRatio: number;
    marginUtilization: number;
    portfolioVaR: number;
    expectedShortfall: number;
    volatilityIndex: number;
    currentDrawdown: number;
    maxDrawdown: number;
    drawdownDuration: number;
    sharpeRatio: number;
    sortinoRatio: number;
    winRate: number;
    overallRiskScore: number;
    marketRegimeRisk: number;
    concentrationRisk: number;
}
export interface CircuitBreakerConfig {
    maxVolatilityThreshold: number;
    volatilityLookbackPeriod: number;
    maxDrawdownThreshold: number;
    dailyLossLimit: number;
    maxPositionSize: number;
    maxTotalExposure: number;
    maxLeverageRatio: number;
    minLiquidityThreshold: number;
    maxSpreadThreshold: number;
    emergencyStopEnabled: boolean;
    forceCloseThreshold: number;
}
export interface FailsafeMechanism {
    id: string;
    name: string;
    type: 'circuit_breaker' | 'position_limit' | 'volatility_control' | 'emergency_stop';
    isActive: boolean;
    threshold: number;
    currentValue: number;
    triggeredAt?: number;
    description: string;
}
export interface RiskAssessment {
    isAcceptable: boolean;
    riskScore: number;
    riskFactors: string[];
    recommendations: string[];
    maxPositionSize: number;
    maxLeverage: number;
}
export declare class EnhancedRiskManagementSystem {
    private positionManager;
    private decisionEngine;
    private dataIntegration;
    private tradingBot;
    private redis;
    private riskMetrics;
    private historicalReturns;
    private portfolioValues;
    private circuitBreakerConfig;
    private failsafeMechanisms;
    private riskEvents;
    constructor();
    /**
     * Initialize Enhanced Risk Management System
     */
    initialize(): Promise<void>;
    /**
     * Assess risk for a trading decision
     */
    assessTradingRisk(decision: TradingDecision, currentPrice: number): Promise<RiskAssessment>;
    /**
     * Check circuit breakers and failsafe mechanisms
     */
    checkCircuitBreakers(): Promise<{
        triggered: boolean;
        mechanisms: FailsafeMechanism[];
    }>;
    /**
     * Calculate dynamic position sizing based on risk
     */
    calculateDynamicPositionSize(baseSize: number, confidence: number, riskScore: number, marketVolatility: number): number;
    /**
     * Get current risk metrics
     */
    getRiskMetrics(): RiskMetrics;
    /**
     * Get active failsafe mechanisms
     */
    getFailsafeMechanisms(): FailsafeMechanism[];
    /**
     * Get risk events history
     */
    getRiskEvents(limit?: number): any[];
    /**
     * Update circuit breaker configuration
     */
    updateCircuitBreakerConfig(newConfig: Partial<CircuitBreakerConfig>): void;
    /**
     * Enable/disable specific failsafe mechanism
     */
    toggleFailsafeMechanism(mechanismId: string, enabled: boolean): boolean;
    /**
     * Cleanup resources
     */
    cleanup(): Promise<void>;
    /**
     * Initialize risk metrics
     */
    private initializeRiskMetrics;
    /**
     * Initialize failsafe mechanisms
     */
    private initializeFailsafeMechanisms;
    /**
     * Update risk metrics with current portfolio state
     */
    private updateRiskMetrics;
    /**
     * Calculate position-specific risk
     */
    private calculatePositionRisk;
    /**
     * Calculate portfolio impact of new position
     */
    private calculatePortfolioImpact;
    /**
     * Assess current market conditions
     */
    private assessMarketConditions;
    /**
     * Calculate overall risk score from components
     */
    private calculateOverallRiskScore;
    /**
     * Identify risk factors from risk components
     */
    private identifyRiskFactors;
    /**
     * Generate risk mitigation recommendations
     */
    private generateRiskRecommendations;
    /**
     * Calculate maximum recommended position size based on risk
     */
    private calculateMaxPositionSize;
    /**
     * Calculate maximum recommended leverage based on risk
     */
    private calculateMaxLeverage;
    /**
     * Load historical data for risk calculations
     */
    private loadHistoricalData;
    /**
     * Start risk monitoring
     */
    private startRiskMonitoring;
    /**
     * Calculate total portfolio exposure
     */
    private calculateTotalExposure;
    /**
     * Calculate portfolio leverage ratio
     */
    private calculateLeverageRatio;
    /**
     * Calculate volatility index using historical returns
     */
    private calculateVolatilityIndex;
    /**
     * Calculate Value at Risk and Expected Shortfall
     */
    private calculateRiskMeasures;
    /**
     * Update drawdown metrics
     */
    private updateDrawdownMetrics;
    /**
     * Calculate Sharpe ratio
     */
    private calculateSharpeRatio;
    /**
     * Calculate Sortino ratio
     */
    private calculateSortinoRatio;
    /**
     * Calculate market regime risk
     */
    private calculateMarketRegimeRisk;
    /**
     * Calculate concentration risk
     */
    private calculateConcentrationRisk;
    /**
     * Calculate timing risk based on market session and volatility
     */
    private calculateTimingRisk;
    /**
     * Calculate correlation risk (simplified)
     */
    private calculateCorrelationRisk;
    /**
     * Calculate portfolio heat (risk concentration)
     */
    private calculatePortfolioHeat;
    /**
     * Update failsafe mechanism current values
     */
    private updateFailsafeMechanismValues;
    /**
     * Check if a failsafe mechanism is triggered
     */
    private isMechanismTriggered;
    /**
     * Execute emergency actions when circuit breakers are triggered
     */
    private executeEmergencyActions;
    /**
     * Execute emergency stop - close all positions
     */
    private executeEmergencyStop;
    /**
     * Execute circuit breaker actions
     */
    private executeCircuitBreakerActions;
    /**
     * Execute volatility controls
     */
    private executeVolatilityControls;
    /**
     * Execute position limit actions
     */
    private executePositionLimitActions;
    /**
     * Reduce position sizes by a factor
     */
    private reducePositionSizes;
    /**
     * Tighten stop losses by a factor
     */
    private tightenStopLosses;
    /**
     * Suspend trading for a specific reason
     */
    private suspendTrading;
    /**
     * Calculate daily P&L
     */
    private calculateDailyPnL;
    /**
     * Record risk event for monitoring and analysis
     */
    private recordRiskEvent;
    /**
     * Save risk data to Redis
     */
    private saveRiskData;
}
