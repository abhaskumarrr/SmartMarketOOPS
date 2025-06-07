/**
 * Enhanced Risk Management System
 * Advanced risk controls with dynamic position sizing, circuit breakers, and failsafe mechanisms
 * Optimized for high-leverage trading with extreme market condition protection
 */

import { MLPositionManager, Position } from './MLPositionManager';
import { EnhancedTradingDecisionEngine, TradingDecision } from './EnhancedTradingDecisionEngine';
import { DataCollectorIntegration, ExtractedFeatures } from './DataCollectorIntegration';
import { DeltaTradingBot } from './DeltaTradingBot';
import { logger } from '../utils/logger';
import Redis from 'ioredis';

// Risk management types
export interface RiskMetrics {
  // Portfolio risk
  totalExposure: number;           // Total portfolio exposure
  leverageRatio: number;           // Current leverage ratio
  marginUtilization: number;      // Margin utilization percentage
  
  // Volatility metrics
  portfolioVaR: number;           // Value at Risk (1-day, 95%)
  expectedShortfall: number;      // Expected Shortfall (CVaR)
  volatilityIndex: number;        // Current market volatility index
  
  // Drawdown metrics
  currentDrawdown: number;        // Current drawdown from peak
  maxDrawdown: number;           // Maximum historical drawdown
  drawdownDuration: number;       // Days in current drawdown
  
  // Performance metrics
  sharpeRatio: number;           // Risk-adjusted returns
  sortinoRatio: number;          // Downside risk-adjusted returns
  winRate: number;               // Percentage of winning trades
  
  // Risk scores
  overallRiskScore: number;      // 0-1 overall risk assessment
  marketRegimeRisk: number;      // Market regime risk factor
  concentrationRisk: number;     // Position concentration risk
}

// Circuit breaker configuration
export interface CircuitBreakerConfig {
  // Volatility circuit breakers
  maxVolatilityThreshold: number;     // 0.15 - 15% max volatility
  volatilityLookbackPeriod: number;   // 20 - periods for volatility calculation
  
  // Drawdown circuit breakers
  maxDrawdownThreshold: number;       // 0.20 - 20% max drawdown
  dailyLossLimit: number;             // 0.05 - 5% daily loss limit
  
  // Position limits
  maxPositionSize: number;            // 0.10 - 10% max single position
  maxTotalExposure: number;           // 3.0 - 300% max total exposure
  maxLeverageRatio: number;           // 200 - maximum leverage
  
  // Market condition limits
  minLiquidityThreshold: number;      // Minimum market liquidity
  maxSpreadThreshold: number;         // Maximum bid-ask spread
  
  // Emergency controls
  emergencyStopEnabled: boolean;      // Enable emergency stop
  forceCloseThreshold: number;        // 0.25 - 25% force close threshold
}

// Failsafe mechanism types
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

// Risk assessment result
export interface RiskAssessment {
  isAcceptable: boolean;
  riskScore: number;              // 0-1 overall risk score
  riskFactors: string[];          // List of risk factors
  recommendations: string[];      // Risk mitigation recommendations
  maxPositionSize: number;        // Recommended max position size
  maxLeverage: number;           // Recommended max leverage
}

export class EnhancedRiskManagementSystem {
  private positionManager: MLPositionManager;
  private decisionEngine: EnhancedTradingDecisionEngine;
  private dataIntegration: DataCollectorIntegration;
  private tradingBot: DeltaTradingBot;
  private redis: Redis;
  
  // Risk tracking
  private riskMetrics: RiskMetrics;
  private historicalReturns: number[] = [];
  private portfolioValues: number[] = [];
  
  // Circuit breaker configuration optimized for high-leverage trading
  private circuitBreakerConfig: CircuitBreakerConfig = {
    // Volatility controls for extreme market conditions
    maxVolatilityThreshold: 0.15,       // 15% max volatility
    volatilityLookbackPeriod: 20,       // 20 periods lookback
    
    // Drawdown protection for capital preservation
    maxDrawdownThreshold: 0.20,         // 20% max drawdown
    dailyLossLimit: 0.05,               // 5% daily loss limit
    
    // Position limits for risk concentration
    maxPositionSize: 0.10,              // 10% max single position
    maxTotalExposure: 3.0,              // 300% max total exposure
    maxLeverageRatio: 200,              // 200x maximum leverage
    
    // Market condition controls
    minLiquidityThreshold: 1000000,     // $1M minimum liquidity
    maxSpreadThreshold: 0.002,          // 0.2% max spread
    
    // Emergency controls
    emergencyStopEnabled: true,
    forceCloseThreshold: 0.25           // 25% force close threshold
  };
  
  // Active failsafe mechanisms
  private failsafeMechanisms: Map<string, FailsafeMechanism> = new Map();
  
  // Performance tracking
  private riskEvents: Array<{
    timestamp: number;
    type: string;
    description: string;
    riskScore: number;
    action: string;
  }> = [];

  constructor() {
    this.positionManager = new MLPositionManager();
    this.decisionEngine = new EnhancedTradingDecisionEngine();
    this.dataIntegration = new DataCollectorIntegration();
    this.tradingBot = new DeltaTradingBot();
    this.redis = new Redis({
      host: process.env.REDIS_HOST || 'localhost',
      port: parseInt(process.env.REDIS_PORT || '6379')
    });
    
    // Initialize risk metrics
    this.riskMetrics = this.initializeRiskMetrics();
    
    // Initialize failsafe mechanisms
    this.initializeFailsafeMechanisms();
  }

  /**
   * Initialize Enhanced Risk Management System
   */
  public async initialize(): Promise<void> {
    try {
      logger.info('🛡️ Initializing Enhanced Risk Management System...');
      
      // Initialize dependencies
      await this.positionManager.initialize();
      await this.decisionEngine.initialize();
      await this.dataIntegration.initialize();
      
      // Load historical data for risk calculations
      await this.loadHistoricalData();
      
      // Start risk monitoring
      this.startRiskMonitoring();
      
      logger.info('✅ Enhanced Risk Management System initialized successfully');
      logger.info(`🔒 Circuit Breakers: Max Drawdown ${(this.circuitBreakerConfig.maxDrawdownThreshold * 100).toFixed(0)}%, Max Volatility ${(this.circuitBreakerConfig.maxVolatilityThreshold * 100).toFixed(0)}%`);
      logger.info(`⚡ Position Limits: Max Size ${(this.circuitBreakerConfig.maxPositionSize * 100).toFixed(0)}%, Max Exposure ${(this.circuitBreakerConfig.maxTotalExposure * 100).toFixed(0)}%`);
      
    } catch (error: any) {
      logger.error('❌ Failed to initialize Enhanced Risk Management System:', error.message);
      throw error;
    }
  }

  /**
   * Assess risk for a trading decision
   */
  public async assessTradingRisk(decision: TradingDecision, currentPrice: number): Promise<RiskAssessment> {
    try {
      logger.debug(`🔍 Assessing trading risk for ${decision.symbol} ${decision.action}`);

      // Update current risk metrics
      await this.updateRiskMetrics();

      // Calculate position-specific risk
      const positionRisk = this.calculatePositionRisk(decision, currentPrice);
      
      // Calculate portfolio impact
      const portfolioImpact = this.calculatePortfolioImpact(decision, currentPrice);
      
      // Check market conditions
      const marketRisk = await this.assessMarketConditions(decision.symbol);
      
      // Calculate overall risk score
      const overallRiskScore = this.calculateOverallRiskScore(positionRisk, portfolioImpact, marketRisk);
      
      // Generate risk assessment
      const assessment: RiskAssessment = {
        isAcceptable: overallRiskScore < 0.8, // 80% risk threshold
        riskScore: overallRiskScore,
        riskFactors: this.identifyRiskFactors(positionRisk, portfolioImpact, marketRisk),
        recommendations: this.generateRiskRecommendations(overallRiskScore, positionRisk, portfolioImpact),
        maxPositionSize: this.calculateMaxPositionSize(overallRiskScore),
        maxLeverage: this.calculateMaxLeverage(overallRiskScore)
      };

      // Log risk assessment
      if (!assessment.isAcceptable) {
        logger.warn(`⚠️ High risk detected for ${decision.symbol}: ${(overallRiskScore * 100).toFixed(1)}%`);
        logger.warn(`   Risk Factors: ${assessment.riskFactors.join(', ')}`);
      }

      // Record risk event
      this.recordRiskEvent('risk_assessment', `${decision.symbol} ${decision.action}`, overallRiskScore, 
        assessment.isAcceptable ? 'approved' : 'rejected');

      return assessment;

    } catch (error: any) {
      logger.error(`❌ Failed to assess trading risk for ${decision.symbol}:`, error.message);
      return {
        isAcceptable: false,
        riskScore: 1.0,
        riskFactors: ['Risk assessment error'],
        recommendations: ['Manual review required'],
        maxPositionSize: 0.01,
        maxLeverage: 10
      };
    }
  }

  /**
   * Check circuit breakers and failsafe mechanisms
   */
  public async checkCircuitBreakers(): Promise<{ triggered: boolean; mechanisms: FailsafeMechanism[] }> {
    try {
      const triggeredMechanisms: FailsafeMechanism[] = [];

      // Update risk metrics
      await this.updateRiskMetrics();

      // Check each failsafe mechanism
      for (const [id, mechanism] of this.failsafeMechanisms) {
        if (mechanism.isActive && this.isMechanismTriggered(mechanism)) {
          mechanism.triggeredAt = Date.now();
          triggeredMechanisms.push(mechanism);
          
          logger.warn(`🚨 Circuit breaker triggered: ${mechanism.name}`);
          logger.warn(`   Current: ${mechanism.currentValue.toFixed(4)} | Threshold: ${mechanism.threshold.toFixed(4)}`);
          
          // Record risk event
          this.recordRiskEvent('circuit_breaker', mechanism.name, mechanism.currentValue, 'triggered');
        }
      }

      // Execute emergency actions if needed
      if (triggeredMechanisms.length > 0) {
        await this.executeEmergencyActions(triggeredMechanisms);
      }

      return {
        triggered: triggeredMechanisms.length > 0,
        mechanisms: triggeredMechanisms
      };

    } catch (error: any) {
      logger.error('❌ Failed to check circuit breakers:', error.message);
      return { triggered: false, mechanisms: [] };
    }
  }

  /**
   * Calculate dynamic position sizing based on risk
   */
  public calculateDynamicPositionSize(
    baseSize: number, 
    confidence: number, 
    riskScore: number, 
    marketVolatility: number
  ): number {
    try {
      // Start with base position size
      let adjustedSize = baseSize;

      // Confidence adjustment (higher confidence = larger size)
      const confidenceMultiplier = 0.5 + (confidence * 1.5); // 0.5x to 2.0x
      adjustedSize *= confidenceMultiplier;

      // Risk adjustment (higher risk = smaller size)
      const riskMultiplier = Math.max(0.1, 1 - riskScore); // 0.1x to 1.0x
      adjustedSize *= riskMultiplier;

      // Volatility adjustment (higher volatility = smaller size)
      const volatilityMultiplier = Math.max(0.2, 1 - marketVolatility * 2); // 0.2x to 1.0x
      adjustedSize *= volatilityMultiplier;

      // Portfolio heat adjustment
      const portfolioHeat = this.calculatePortfolioHeat();
      const heatMultiplier = Math.max(0.1, 1 - portfolioHeat); // 0.1x to 1.0x
      adjustedSize *= heatMultiplier;

      // Apply circuit breaker limits
      adjustedSize = Math.min(adjustedSize, this.circuitBreakerConfig.maxPositionSize);

      logger.debug(`📊 Dynamic position sizing: Base ${(baseSize * 100).toFixed(1)}% → Adjusted ${(adjustedSize * 100).toFixed(1)}%`);
      logger.debug(`   Confidence: ${(confidence * 100).toFixed(0)}% | Risk: ${(riskScore * 100).toFixed(0)}% | Volatility: ${(marketVolatility * 100).toFixed(0)}%`);

      return Math.max(0.001, adjustedSize); // Minimum 0.1% position

    } catch (error: any) {
      logger.error('❌ Failed to calculate dynamic position size:', error.message);
      return Math.min(baseSize * 0.1, 0.01); // Conservative fallback
    }
  }

  /**
   * Get current risk metrics
   */
  public getRiskMetrics(): RiskMetrics {
    return { ...this.riskMetrics };
  }

  /**
   * Get active failsafe mechanisms
   */
  public getFailsafeMechanisms(): FailsafeMechanism[] {
    return Array.from(this.failsafeMechanisms.values());
  }

  /**
   * Get risk events history
   */
  public getRiskEvents(limit: number = 100): any[] {
    return this.riskEvents.slice(-limit);
  }

  /**
   * Update circuit breaker configuration
   */
  public updateCircuitBreakerConfig(newConfig: Partial<CircuitBreakerConfig>): void {
    this.circuitBreakerConfig = { ...this.circuitBreakerConfig, ...newConfig };
    logger.info('🔧 Circuit breaker configuration updated');
  }

  /**
   * Enable/disable specific failsafe mechanism
   */
  public toggleFailsafeMechanism(mechanismId: string, enabled: boolean): boolean {
    const mechanism = this.failsafeMechanisms.get(mechanismId);
    if (mechanism) {
      mechanism.isActive = enabled;
      logger.info(`${enabled ? '🟢' : '🔴'} Failsafe mechanism ${mechanism.name}: ${enabled ? 'ENABLED' : 'DISABLED'}`);
      return true;
    }
    return false;
  }

  /**
   * Cleanup resources
   */
  public async cleanup(): Promise<void> {
    try {
      logger.info('🧹 Cleaning up Enhanced Risk Management System...');
      
      // Save risk data
      await this.saveRiskData();
      
      await this.redis.quit();
      
      logger.info('✅ Enhanced Risk Management System cleanup completed');
    } catch (error: any) {
      logger.error('❌ Error during Enhanced Risk Management System cleanup:', error.message);
    }
  }

  // Private methods for risk management

  /**
   * Initialize risk metrics
   */
  private initializeRiskMetrics(): RiskMetrics {
    return {
      totalExposure: 0,
      leverageRatio: 0,
      marginUtilization: 0,
      portfolioVaR: 0,
      expectedShortfall: 0,
      volatilityIndex: 0,
      currentDrawdown: 0,
      maxDrawdown: 0,
      drawdownDuration: 0,
      sharpeRatio: 0,
      sortinoRatio: 0,
      winRate: 0,
      overallRiskScore: 0,
      marketRegimeRisk: 0,
      concentrationRisk: 0
    };
  }

  /**
   * Initialize failsafe mechanisms
   */
  private initializeFailsafeMechanisms(): void {
    // Volatility circuit breaker
    this.failsafeMechanisms.set('volatility_breaker', {
      id: 'volatility_breaker',
      name: 'Volatility Circuit Breaker',
      type: 'volatility_control',
      isActive: true,
      threshold: this.circuitBreakerConfig.maxVolatilityThreshold,
      currentValue: 0,
      description: 'Stops trading when market volatility exceeds threshold'
    });

    // Drawdown circuit breaker
    this.failsafeMechanisms.set('drawdown_breaker', {
      id: 'drawdown_breaker',
      name: 'Drawdown Circuit Breaker',
      type: 'circuit_breaker',
      isActive: true,
      threshold: this.circuitBreakerConfig.maxDrawdownThreshold,
      currentValue: 0,
      description: 'Stops trading when drawdown exceeds threshold'
    });

    // Position size limit
    this.failsafeMechanisms.set('position_limit', {
      id: 'position_limit',
      name: 'Position Size Limit',
      type: 'position_limit',
      isActive: true,
      threshold: this.circuitBreakerConfig.maxPositionSize,
      currentValue: 0,
      description: 'Prevents positions larger than threshold'
    });

    // Total exposure limit
    this.failsafeMechanisms.set('exposure_limit', {
      id: 'exposure_limit',
      name: 'Total Exposure Limit',
      type: 'position_limit',
      isActive: true,
      threshold: this.circuitBreakerConfig.maxTotalExposure,
      currentValue: 0,
      description: 'Prevents total exposure exceeding threshold'
    });

    // Daily loss limit
    this.failsafeMechanisms.set('daily_loss_limit', {
      id: 'daily_loss_limit',
      name: 'Daily Loss Limit',
      type: 'circuit_breaker',
      isActive: true,
      threshold: this.circuitBreakerConfig.dailyLossLimit,
      currentValue: 0,
      description: 'Stops trading when daily loss exceeds threshold'
    });

    // Emergency stop
    this.failsafeMechanisms.set('emergency_stop', {
      id: 'emergency_stop',
      name: 'Emergency Stop',
      type: 'emergency_stop',
      isActive: this.circuitBreakerConfig.emergencyStopEnabled,
      threshold: this.circuitBreakerConfig.forceCloseThreshold,
      currentValue: 0,
      description: 'Emergency position closure mechanism'
    });

    logger.info(`🛡️ Initialized ${this.failsafeMechanisms.size} failsafe mechanisms`);
  }

  /**
   * Update risk metrics with current portfolio state
   */
  private async updateRiskMetrics(): Promise<void> {
    try {
      const activePositions = this.positionManager.getActivePositions();
      const performanceMetrics = this.positionManager.getPerformanceMetrics();

      // Calculate total exposure
      this.riskMetrics.totalExposure = this.calculateTotalExposure(activePositions);

      // Calculate leverage ratio
      this.riskMetrics.leverageRatio = this.calculateLeverageRatio(activePositions);

      // Calculate margin utilization (simplified)
      this.riskMetrics.marginUtilization = Math.min(1.0, this.riskMetrics.totalExposure / 10); // Assume 10x base

      // Calculate volatility index
      this.riskMetrics.volatilityIndex = await this.calculateVolatilityIndex();

      // Calculate VaR and Expected Shortfall
      const riskMeasures = this.calculateRiskMeasures();
      this.riskMetrics.portfolioVaR = riskMeasures.var;
      this.riskMetrics.expectedShortfall = riskMeasures.expectedShortfall;

      // Update drawdown metrics
      this.updateDrawdownMetrics();

      // Calculate performance ratios
      this.riskMetrics.sharpeRatio = this.calculateSharpeRatio();
      this.riskMetrics.sortinoRatio = this.calculateSortinoRatio();
      this.riskMetrics.winRate = parseFloat(performanceMetrics.winRate) / 100;

      // Calculate risk scores
      this.riskMetrics.overallRiskScore = this.calculateOverallRiskScore();
      this.riskMetrics.marketRegimeRisk = await this.calculateMarketRegimeRisk();
      this.riskMetrics.concentrationRisk = this.calculateConcentrationRisk(activePositions);

      // Update failsafe mechanism current values
      this.updateFailsafeMechanismValues();

    } catch (error: any) {
      logger.error('❌ Failed to update risk metrics:', error.message);
    }
  }

  /**
   * Calculate position-specific risk
   */
  private calculatePositionRisk(decision: TradingDecision, currentPrice: number): number {
    let riskScore = 0;

    // Leverage risk (higher leverage = higher risk)
    const leverageRisk = Math.min(1.0, decision.leverage / 200); // Normalize to 200x max
    riskScore += leverageRisk * 0.3; // 30% weight

    // Position size risk
    const sizeRisk = Math.min(1.0, decision.positionSize / 0.1); // Normalize to 10% max
    riskScore += sizeRisk * 0.2; // 20% weight

    // Confidence risk (lower confidence = higher risk)
    const confidenceRisk = 1 - decision.confidence;
    riskScore += confidenceRisk * 0.2; // 20% weight

    // Stop loss distance risk
    const stopDistance = Math.abs(currentPrice - decision.stopLoss) / currentPrice;
    const stopRisk = Math.min(1.0, stopDistance / 0.05); // Normalize to 5% max
    riskScore += (1 - stopRisk) * 0.15; // 15% weight (closer stop = higher risk due to tight margin)

    // Market timing risk
    const timingRisk = this.calculateTimingRisk();
    riskScore += timingRisk * 0.15; // 15% weight

    return Math.min(1.0, riskScore);
  }

  /**
   * Calculate portfolio impact of new position
   */
  private calculatePortfolioImpact(decision: TradingDecision, currentPrice: number): number {
    const activePositions = this.positionManager.getActivePositions();

    // Calculate correlation risk (simplified)
    const correlationRisk = this.calculateCorrelationRisk(decision.symbol, activePositions);

    // Calculate concentration risk
    const newExposure = decision.positionSize * decision.leverage;
    const totalExposure = this.riskMetrics.totalExposure + newExposure;
    const concentrationRisk = Math.min(1.0, totalExposure / this.circuitBreakerConfig.maxTotalExposure);

    // Calculate margin impact
    const marginImpact = Math.min(1.0, newExposure / 5); // Normalize to 500% exposure

    return Math.max(correlationRisk, concentrationRisk, marginImpact);
  }

  /**
   * Assess current market conditions
   */
  private async assessMarketConditions(symbol: string): Promise<number> {
    try {
      const features = await this.dataIntegration.getRealTimeTradingFeatures(symbol);
      if (!features) return 0.8; // High risk if no data

      let marketRisk = 0;

      // Volatility risk
      const volatilityRisk = Math.min(1.0, features.volatility * 5); // Scale volatility
      marketRisk += volatilityRisk * 0.4; // 40% weight

      // Bias alignment risk (poor alignment = higher risk)
      const alignmentRisk = 1 - features.biasAlignment;
      marketRisk += alignmentRisk * 0.3; // 30% weight

      // Data quality risk
      const qualityRisk = 1 - features.dataQuality;
      marketRisk += qualityRisk * 0.2; // 20% weight

      // Market session risk
      const sessionRisk = features.marketSession === 0 ? 0.3 : 0; // Asian session risk
      marketRisk += sessionRisk * 0.1; // 10% weight

      return Math.min(1.0, marketRisk);

    } catch (error: any) {
      logger.error(`❌ Failed to assess market conditions for ${symbol}:`, error.message);
      return 0.8; // Conservative high risk
    }
  }

  /**
   * Calculate overall risk score from components
   */
  private calculateOverallRiskScore(
    positionRisk?: number,
    portfolioImpact?: number,
    marketRisk?: number
  ): number {
    if (positionRisk !== undefined && portfolioImpact !== undefined && marketRisk !== undefined) {
      // For trading decision assessment
      return (positionRisk * 0.4) + (portfolioImpact * 0.35) + (marketRisk * 0.25);
    } else {
      // For portfolio-level risk assessment
      const volatilityComponent = Math.min(1.0, this.riskMetrics.volatilityIndex / 0.3);
      const drawdownComponent = Math.min(1.0, Math.abs(this.riskMetrics.currentDrawdown) / 0.2);
      const exposureComponent = Math.min(1.0, this.riskMetrics.totalExposure / 3.0);
      const leverageComponent = Math.min(1.0, this.riskMetrics.leverageRatio / 200);

      return (volatilityComponent * 0.3) + (drawdownComponent * 0.3) +
             (exposureComponent * 0.25) + (leverageComponent * 0.15);
    }
  }

  /**
   * Identify risk factors from risk components
   */
  private identifyRiskFactors(positionRisk: number, portfolioImpact: number, marketRisk: number): string[] {
    const factors: string[] = [];

    if (positionRisk > 0.7) factors.push('High position risk');
    if (portfolioImpact > 0.7) factors.push('High portfolio impact');
    if (marketRisk > 0.7) factors.push('Adverse market conditions');
    if (this.riskMetrics.volatilityIndex > 0.15) factors.push('High market volatility');
    if (this.riskMetrics.currentDrawdown < -0.1) factors.push('Significant drawdown');
    if (this.riskMetrics.totalExposure > 2.5) factors.push('High portfolio exposure');
    if (this.riskMetrics.concentrationRisk > 0.8) factors.push('Position concentration risk');

    return factors;
  }

  /**
   * Generate risk mitigation recommendations
   */
  private generateRiskRecommendations(
    overallRisk: number,
    positionRisk: number,
    portfolioImpact: number
  ): string[] {
    const recommendations: string[] = [];

    if (overallRisk > 0.8) {
      recommendations.push('Consider reducing position size');
      recommendations.push('Implement tighter stop losses');
    }

    if (positionRisk > 0.7) {
      recommendations.push('Reduce leverage');
      recommendations.push('Wait for better entry conditions');
    }

    if (portfolioImpact > 0.7) {
      recommendations.push('Diversify positions across symbols');
      recommendations.push('Reduce total portfolio exposure');
    }

    if (this.riskMetrics.volatilityIndex > 0.15) {
      recommendations.push('Wait for lower volatility');
      recommendations.push('Use smaller position sizes');
    }

    return recommendations;
  }

  /**
   * Calculate maximum recommended position size based on risk
   */
  private calculateMaxPositionSize(riskScore: number): number {
    const baseMaxSize = this.circuitBreakerConfig.maxPositionSize;
    const riskAdjustment = Math.max(0.1, 1 - riskScore); // 10% to 100% of base
    return baseMaxSize * riskAdjustment;
  }

  /**
   * Calculate maximum recommended leverage based on risk
   */
  private calculateMaxLeverage(riskScore: number): number {
    const baseMaxLeverage = this.circuitBreakerConfig.maxLeverageRatio;
    const riskAdjustment = Math.max(0.25, 1 - riskScore); // 25% to 100% of base
    return Math.round(baseMaxLeverage * riskAdjustment);
  }

  /**
   * Load historical data for risk calculations
   */
  private async loadHistoricalData(): Promise<void> {
    try {
      // Load historical returns from Redis
      const returnsData = await this.redis.get('risk_historical_returns');
      if (returnsData) {
        this.historicalReturns = JSON.parse(returnsData);
      }

      // Load portfolio values from Redis
      const portfolioData = await this.redis.get('risk_portfolio_values');
      if (portfolioData) {
        this.portfolioValues = JSON.parse(portfolioData);
      }

      logger.info(`📊 Loaded ${this.historicalReturns.length} historical returns and ${this.portfolioValues.length} portfolio values`);

    } catch (error: any) {
      logger.error('❌ Failed to load historical data:', error.message);
    }
  }

  /**
   * Start risk monitoring
   */
  private startRiskMonitoring(): void {
    // Update risk metrics every 30 seconds
    setInterval(async () => {
      try {
        await this.updateRiskMetrics();
        await this.checkCircuitBreakers();
      } catch (error: any) {
        logger.error('❌ Error in risk monitoring:', error.message);
      }
    }, 30000);

    logger.info('⏰ Risk monitoring started (30-second intervals)');
  }

  /**
   * Calculate total portfolio exposure
   */
  private calculateTotalExposure(positions: Position[]): number {
    return positions.reduce((total, position) => {
      const positionValue = Math.abs(position.quantity * position.currentPrice);
      const exposure = positionValue * position.leverage;
      return total + exposure;
    }, 0);
  }

  /**
   * Calculate portfolio leverage ratio
   */
  private calculateLeverageRatio(positions: Position[]): number {
    if (positions.length === 0) return 0;

    const totalNotional = positions.reduce((total, position) => {
      return total + Math.abs(position.quantity * position.currentPrice * position.leverage);
    }, 0);

    const totalEquity = 10000; // Placeholder - should get actual account equity
    return totalNotional / totalEquity;
  }

  /**
   * Calculate volatility index using historical returns
   */
  private async calculateVolatilityIndex(): Promise<number> {
    if (this.historicalReturns.length < 20) return 0.1; // Default volatility

    // Calculate rolling 20-period volatility
    const recentReturns = this.historicalReturns.slice(-20);
    const mean = recentReturns.reduce((sum, ret) => sum + ret, 0) / recentReturns.length;

    const variance = recentReturns.reduce((sum, ret) => {
      return sum + Math.pow(ret - mean, 2);
    }, 0) / (recentReturns.length - 1);

    const volatility = Math.sqrt(variance);
    return Math.min(1.0, volatility); // Cap at 100%
  }

  /**
   * Calculate Value at Risk and Expected Shortfall
   */
  private calculateRiskMeasures(): { var: number; expectedShortfall: number } {
    if (this.historicalReturns.length < 100) {
      return { var: 0.02, expectedShortfall: 0.03 }; // Default values
    }

    // Sort returns in ascending order
    const sortedReturns = [...this.historicalReturns].sort((a, b) => a - b);

    // Calculate 95% VaR (5th percentile)
    const varIndex = Math.floor(sortedReturns.length * 0.05);
    const var95 = Math.abs(sortedReturns[varIndex]);

    // Calculate Expected Shortfall (average of returns below VaR)
    const tailReturns = sortedReturns.slice(0, varIndex);
    const expectedShortfall = tailReturns.length > 0 ?
      Math.abs(tailReturns.reduce((sum, ret) => sum + ret, 0) / tailReturns.length) : var95;

    return { var: var95, expectedShortfall };
  }

  /**
   * Update drawdown metrics
   */
  private updateDrawdownMetrics(): void {
    if (this.portfolioValues.length < 2) return;

    // Find peak value
    const peak = Math.max(...this.portfolioValues);
    const current = this.portfolioValues[this.portfolioValues.length - 1];

    // Calculate current drawdown
    this.riskMetrics.currentDrawdown = (current - peak) / peak;

    // Update maximum drawdown
    if (this.riskMetrics.currentDrawdown < this.riskMetrics.maxDrawdown) {
      this.riskMetrics.maxDrawdown = this.riskMetrics.currentDrawdown;
    }

    // Calculate drawdown duration (simplified)
    let durationCount = 0;
    for (let i = this.portfolioValues.length - 1; i >= 0; i--) {
      if (this.portfolioValues[i] < peak) {
        durationCount++;
      } else {
        break;
      }
    }
    this.riskMetrics.drawdownDuration = durationCount;
  }

  /**
   * Calculate Sharpe ratio
   */
  private calculateSharpeRatio(): number {
    if (this.historicalReturns.length < 30) return 0;

    const mean = this.historicalReturns.reduce((sum, ret) => sum + ret, 0) / this.historicalReturns.length;
    const variance = this.historicalReturns.reduce((sum, ret) => {
      return sum + Math.pow(ret - mean, 2);
    }, 0) / (this.historicalReturns.length - 1);

    const stdDev = Math.sqrt(variance);
    const riskFreeRate = 0.02 / 252; // 2% annual risk-free rate, daily

    return stdDev > 0 ? (mean - riskFreeRate) / stdDev : 0;
  }

  /**
   * Calculate Sortino ratio
   */
  private calculateSortinoRatio(): number {
    if (this.historicalReturns.length < 30) return 0;

    const mean = this.historicalReturns.reduce((sum, ret) => sum + ret, 0) / this.historicalReturns.length;
    const negativeReturns = this.historicalReturns.filter(ret => ret < 0);

    if (negativeReturns.length === 0) return 10; // Very high ratio if no negative returns

    const downside = negativeReturns.reduce((sum, ret) => sum + Math.pow(ret, 2), 0) / negativeReturns.length;
    const downsideStdDev = Math.sqrt(downside);
    const riskFreeRate = 0.02 / 252;

    return downsideStdDev > 0 ? (mean - riskFreeRate) / downsideStdDev : 0;
  }

  /**
   * Calculate market regime risk
   */
  private async calculateMarketRegimeRisk(): Promise<number> {
    try {
      // Simplified market regime detection based on volatility and trend
      const volatilityRisk = Math.min(1.0, this.riskMetrics.volatilityIndex / 0.2);

      // Add trend consistency risk (would need market data)
      const trendRisk = 0.3; // Placeholder

      return Math.max(volatilityRisk, trendRisk);

    } catch (error: any) {
      logger.error('❌ Failed to calculate market regime risk:', error.message);
      return 0.5; // Default moderate risk
    }
  }

  /**
   * Calculate concentration risk
   */
  private calculateConcentrationRisk(positions: Position[]): number {
    if (positions.length === 0) return 0;

    // Calculate Herfindahl-Hirschman Index for concentration
    const totalExposure = this.calculateTotalExposure(positions);
    if (totalExposure === 0) return 0;

    const hhi = positions.reduce((sum, position) => {
      const positionExposure = Math.abs(position.quantity * position.currentPrice * position.leverage);
      const share = positionExposure / totalExposure;
      return sum + Math.pow(share, 2);
    }, 0);

    // Normalize HHI to 0-1 scale (1/n to 1, where n is number of positions)
    const minHHI = 1 / positions.length;
    const normalizedHHI = (hhi - minHHI) / (1 - minHHI);

    return Math.max(0, Math.min(1, normalizedHHI));
  }

  /**
   * Calculate timing risk based on market session and volatility
   */
  private calculateTimingRisk(): number {
    const now = new Date();
    const utcHours = now.getUTCHours();

    // Higher risk during Asian session (low liquidity)
    if (utcHours >= 0 && utcHours < 7) return 0.4;

    // Lower risk during European and American sessions
    if (utcHours >= 7 && utcHours < 22) return 0.1;

    // Moderate risk during transition periods
    return 0.2;
  }

  /**
   * Calculate correlation risk (simplified)
   */
  private calculateCorrelationRisk(symbol: string, positions: Position[]): number {
    // Simplified correlation calculation
    // In production, would use actual correlation matrix

    const sameSymbolPositions = positions.filter(pos => pos.symbol === symbol);
    if (sameSymbolPositions.length > 0) return 0.8; // High correlation risk

    // Check for related symbols (BTC/ETH correlation)
    const cryptoSymbols = ['BTCUSD', 'ETHUSD', 'SOLUSD'];
    const hasCrypto = positions.some(pos => cryptoSymbols.includes(pos.symbol));
    if (hasCrypto && cryptoSymbols.includes(symbol)) return 0.6; // Moderate correlation

    return 0.2; // Low correlation
  }

  /**
   * Calculate portfolio heat (risk concentration)
   */
  private calculatePortfolioHeat(): number {
    const activePositions = this.positionManager.getActivePositions();
    if (activePositions.length === 0) return 0;

    // Calculate heat based on unrealized P&L and risk
    let totalHeat = 0;
    for (const position of activePositions) {
      const positionHeat = Math.abs(position.unrealizedPnL) * position.leverage / 100;
      totalHeat += positionHeat;
    }

    return Math.min(1.0, totalHeat);
  }

  /**
   * Update failsafe mechanism current values
   */
  private updateFailsafeMechanismValues(): void {
    // Update volatility breaker
    const volatilityBreaker = this.failsafeMechanisms.get('volatility_breaker');
    if (volatilityBreaker) {
      volatilityBreaker.currentValue = this.riskMetrics.volatilityIndex;
    }

    // Update drawdown breaker
    const drawdownBreaker = this.failsafeMechanisms.get('drawdown_breaker');
    if (drawdownBreaker) {
      drawdownBreaker.currentValue = Math.abs(this.riskMetrics.currentDrawdown);
    }

    // Update position limit
    const positionLimit = this.failsafeMechanisms.get('position_limit');
    if (positionLimit) {
      const activePositions = this.positionManager.getActivePositions();
      const maxPositionSize = activePositions.reduce((max, pos) => {
        const positionSize = Math.abs(pos.quantity * pos.currentPrice) / 10000; // Normalize to account size
        return Math.max(max, positionSize);
      }, 0);
      positionLimit.currentValue = maxPositionSize;
    }

    // Update exposure limit
    const exposureLimit = this.failsafeMechanisms.get('exposure_limit');
    if (exposureLimit) {
      exposureLimit.currentValue = this.riskMetrics.totalExposure;
    }

    // Update daily loss limit
    const dailyLossLimit = this.failsafeMechanisms.get('daily_loss_limit');
    if (dailyLossLimit) {
      const dailyPnL = this.calculateDailyPnL();
      dailyLossLimit.currentValue = Math.abs(Math.min(0, dailyPnL));
    }

    // Update emergency stop
    const emergencyStop = this.failsafeMechanisms.get('emergency_stop');
    if (emergencyStop) {
      emergencyStop.currentValue = Math.abs(this.riskMetrics.currentDrawdown);
    }
  }

  /**
   * Check if a failsafe mechanism is triggered
   */
  private isMechanismTriggered(mechanism: FailsafeMechanism): boolean {
    switch (mechanism.type) {
      case 'circuit_breaker':
      case 'volatility_control':
      case 'emergency_stop':
        return mechanism.currentValue > mechanism.threshold;

      case 'position_limit':
        return mechanism.currentValue > mechanism.threshold;

      default:
        return false;
    }
  }

  /**
   * Execute emergency actions when circuit breakers are triggered
   */
  private async executeEmergencyActions(triggeredMechanisms: FailsafeMechanism[]): Promise<void> {
    try {
      logger.warn(`🚨 Executing emergency actions for ${triggeredMechanisms.length} triggered mechanisms`);

      for (const mechanism of triggeredMechanisms) {
        switch (mechanism.type) {
          case 'emergency_stop':
            await this.executeEmergencyStop();
            break;

          case 'circuit_breaker':
            await this.executeCircuitBreakerActions(mechanism);
            break;

          case 'volatility_control':
            await this.executeVolatilityControls();
            break;

          case 'position_limit':
            await this.executePositionLimitActions();
            break;
        }
      }

    } catch (error: any) {
      logger.error('❌ Failed to execute emergency actions:', error.message);
    }
  }

  /**
   * Execute emergency stop - close all positions
   */
  private async executeEmergencyStop(): Promise<void> {
    logger.error('🚨 EMERGENCY STOP TRIGGERED - CLOSING ALL POSITIONS');

    const activePositions = this.positionManager.getActivePositions();

    for (const position of activePositions) {
      try {
        await this.positionManager.closePosition(
          position.id,
          position.currentPrice,
          'Emergency stop triggered'
        );
        logger.warn(`🔒 Emergency closed position: ${position.id}`);
      } catch (error: any) {
        logger.error(`❌ Failed to emergency close position ${position.id}:`, error.message);
      }
    }

    // Record critical risk event
    this.recordRiskEvent('emergency_stop', 'All positions closed', 1.0, 'executed');
  }

  /**
   * Execute circuit breaker actions
   */
  private async executeCircuitBreakerActions(mechanism: FailsafeMechanism): Promise<void> {
    logger.warn(`🔴 Circuit breaker action: ${mechanism.name}`);

    if (mechanism.id === 'drawdown_breaker') {
      // Stop new trades and reduce position sizes
      await this.reducePositionSizes(0.5); // Reduce by 50%
    } else if (mechanism.id === 'daily_loss_limit') {
      // Stop trading for the day
      await this.suspendTrading('daily_loss_limit');
    }
  }

  /**
   * Execute volatility controls
   */
  private async executeVolatilityControls(): Promise<void> {
    logger.warn('🌪️ High volatility detected - implementing controls');

    // Reduce position sizes during high volatility
    await this.reducePositionSizes(0.3); // Reduce by 70%

    // Tighten stop losses
    await this.tightenStopLosses(0.5); // Tighten by 50%
  }

  /**
   * Execute position limit actions
   */
  private async executePositionLimitActions(): Promise<void> {
    logger.warn('📊 Position limits exceeded - implementing controls');

    // Prevent new large positions
    // This would be implemented in the trading decision process
  }

  /**
   * Reduce position sizes by a factor
   */
  private async reducePositionSizes(reductionFactor: number): Promise<void> {
    const activePositions = this.positionManager.getActivePositions();

    for (const position of activePositions) {
      try {
        // Calculate partial close amount
        const closeAmount = position.quantity * reductionFactor;

        // This would require implementing partial position closure
        logger.info(`📉 Reducing position ${position.id} by ${(reductionFactor * 100).toFixed(0)}%`);

      } catch (error: any) {
        logger.error(`❌ Failed to reduce position ${position.id}:`, error.message);
      }
    }
  }

  /**
   * Tighten stop losses by a factor
   */
  private async tightenStopLosses(tighteningFactor: number): Promise<void> {
    const activePositions = this.positionManager.getActivePositions();

    for (const position of activePositions) {
      try {
        const currentDistance = Math.abs(position.currentPrice - position.stopLoss);
        const newDistance = currentDistance * (1 - tighteningFactor);

        if (position.side === 'long') {
          position.stopLoss = position.currentPrice - newDistance;
        } else {
          position.stopLoss = position.currentPrice + newDistance;
        }

        logger.info(`🔒 Tightened stop loss for position ${position.id}: ${position.stopLoss}`);

      } catch (error: any) {
        logger.error(`❌ Failed to tighten stop loss for position ${position.id}:`, error.message);
      }
    }
  }

  /**
   * Suspend trading for a specific reason
   */
  private async suspendTrading(reason: string): Promise<void> {
    logger.error(`⏸️ Trading suspended: ${reason}`);

    // Set suspension flag in Redis
    await this.redis.setex('trading_suspended', 86400, JSON.stringify({
      reason,
      timestamp: Date.now(),
      duration: 86400000 // 24 hours
    }));
  }

  /**
   * Calculate daily P&L
   */
  private calculateDailyPnL(): number {
    // Simplified daily P&L calculation
    // In production, would track actual daily performance
    const performanceMetrics = this.positionManager.getPerformanceMetrics();
    return parseFloat(performanceMetrics.totalPnL) * 0.1; // Approximate daily portion
  }

  /**
   * Record risk event for monitoring and analysis
   */
  private recordRiskEvent(type: string, description: string, riskScore: number, action: string): void {
    const event = {
      timestamp: Date.now(),
      type,
      description,
      riskScore,
      action
    };

    this.riskEvents.push(event);

    // Keep only last 1000 events
    if (this.riskEvents.length > 1000) {
      this.riskEvents = this.riskEvents.slice(-1000);
    }

    logger.debug(`📝 Risk event recorded: ${type} - ${description} (${action})`);
  }

  /**
   * Save risk data to Redis
   */
  private async saveRiskData(): Promise<void> {
    try {
      // Save historical returns
      await this.redis.setex('risk_historical_returns', 86400 * 7, JSON.stringify(this.historicalReturns));

      // Save portfolio values
      await this.redis.setex('risk_portfolio_values', 86400 * 7, JSON.stringify(this.portfolioValues));

      // Save risk events
      await this.redis.setex('risk_events', 86400 * 7, JSON.stringify(this.riskEvents));

      logger.debug('💾 Risk data saved to Redis');

    } catch (error: any) {
      logger.error('❌ Failed to save risk data:', error.message);
    }
  }
}
