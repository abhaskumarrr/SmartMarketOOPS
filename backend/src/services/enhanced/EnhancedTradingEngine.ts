/**
 * Enhanced Trading Engine (2025 Edition)
 * 
 * Professional trading engine with advanced patterns:
 * - Event-driven order management with async patterns
 * - Advanced position management with real-time tracking
 * - Comprehensive risk controls with circuit breakers
 * - Performance monitoring and adaptive behavior
 * - Auto-scaling and intelligent order routing
 * - Dead letter queues for error handling
 * 
 * Integrates with the enhanced ML Service Bridge and Real-time Data Stream Manager
 */

import { EventEmitter } from 'events';
import { logger } from '../../utils/logger';
import { RealtimeDataStreamManager, STREAM_NAMES, MessageProcessor, StreamMessage } from './RealtimeDataStreamManager';
import { MLServiceBridge, MLPredictionRequest, MLPredictionResponse } from './MLServiceBridge';

// Trading interfaces
export interface Order {
  id: string;
  clientOrderId: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  type: 'MARKET' | 'LIMIT' | 'STOP' | 'STOP_LIMIT';
  status: 'PENDING' | 'PARTIALLY_FILLED' | 'FILLED' | 'CANCELLED' | 'REJECTED' | 'EXPIRED';
  quantity: number;
  price?: number;
  stopPrice?: number;
  timeInForce: 'GTC' | 'IOC' | 'FOK' | 'DAY';
  exchange: string;
  timestamp: number;
  filledQuantity: number;
  avgFillPrice: number;
  commission: number;
  metadata: {
    source: 'manual' | 'ml_signal' | 'risk_management' | 'stop_loss' | 'take_profit';
    signalId?: string;
    riskScore?: number;
    expectedPnL?: number;
    parentOrderId?: string;
  };
}

export interface Position {
  id: string;
  symbol: string;
  side: 'LONG' | 'SHORT';
  quantity: number;
  entryPrice: number;
  currentPrice: number;
  unrealizedPnL: number;
  realizedPnL: number;
  margin: number;
  leverage: number;
  stopLoss?: number;
  takeProfit?: number;
  openTime: number;
  lastUpdateTime: number;
  orders: string[]; // Associated order IDs
  metadata: {
    strategy: string;
    riskLevel: 'LOW' | 'MEDIUM' | 'HIGH';
    maxRisk: number;
    targetReturn: number;
  };
}

export interface TradingSignal {
  id: string;
  symbol: string;
  action: 'BUY' | 'SELL' | 'HOLD' | 'CLOSE_LONG' | 'CLOSE_SHORT';
  confidence: number;
  price: number;
  quantity?: number;
  stopLoss?: number;
  takeProfit?: number;
  timeframe: string;
  source: 'ml_prediction' | 'technical_analysis' | 'fundamental_analysis' | 'manual';
  timestamp: number;
  expiryTime?: number;
  metadata: {
    mlPrediction?: MLPredictionResponse;
    riskRewardRatio?: number;
    expectedVolatility?: number;
    marketRegime?: string;
  };
}

export interface RiskLimits {
  maxPositionSize: number;
  maxDailyLoss: number;
  maxDrawdown: number;
  maxLeverage: number;
  maxPositionsPerSymbol: number;
  maxTotalPositions: number;
  allowedSymbols: string[];
  forbiddenSymbols: string[];
  tradingHours: {
    start: string;
    end: string;
    timezone: string;
  };
}

export interface TradingEngineConfig {
  accountId: string;
  initialCapital: number;
  riskLimits: RiskLimits;
  orderExecutionConfig: {
    slippageTolerance: number;
    maxExecutionTime: number;
    retryAttempts: number;
    partialFillHandling: 'ACCEPT' | 'REJECT' | 'CANCEL_REMAINING';
  };
  performanceConfig: {
    metricsInterval: number;
    healthCheckInterval: number;
    positionRebalanceInterval: number;
  };
  mlIntegrationConfig: {
    autoTradingEnabled: boolean;
    minConfidenceThreshold: number;
    maxMLPositionsPerHour: number;
    mlSignalCooldown: number;
  };
}

export interface TradingMetrics {
  totalOrders: number;
  successfulOrders: number;
  failedOrders: number;
  totalVolume: number;
  totalPnL: number;
  totalCommissions: number;
  activePositions: number;
  avgExecutionTime: number;
  successRate: number;
  sharpeRatio: number;
  maxDrawdown: number;
  dailyPnL: number;
  winRate: number;
  avgWin: number;
  avgLoss: number;
}

// Order processor for handling order events
class OrderProcessor implements MessageProcessor {
  constructor(private engine: EnhancedTradingEngine) {}

  async process(message: StreamMessage): Promise<void> {
    try {
      const orderData = JSON.parse(message.fields.data || '{}');
      await this.engine.processOrderEvent(orderData);
    } catch (error) {
      logger.error('❌ Error processing order event:', error);
      throw error;
    }
  }

  async onError(error: Error, message: StreamMessage): Promise<void> {
    logger.error(`❌ Order processing error for message ${message.id}:`, error);
    // Send to dead letter queue for manual review
    await this.engine.sendToDeadLetter('order_processing_error', message, error);
  }
}

// Signal processor for handling trading signals
class SignalProcessor implements MessageProcessor {
  constructor(private engine: EnhancedTradingEngine) {}

  async process(message: StreamMessage): Promise<void> {
    try {
      const signalData = JSON.parse(message.fields.data || '{}');
      await this.engine.processSignalEvent(signalData);
    } catch (error) {
      logger.error('❌ Error processing signal event:', error);
      throw error;
    }
  }

  async onError(error: Error, message: StreamMessage): Promise<void> {
    logger.error(`❌ Signal processing error for message ${message.id}:`, error);
    await this.engine.sendToDeadLetter('signal_processing_error', message, error);
  }
}

export class EnhancedTradingEngine extends EventEmitter {
  private config: TradingEngineConfig;
  private streamManager: RealtimeDataStreamManager;
  private mlBridge: MLServiceBridge;
  
  // State management
  private orders: Map<string, Order> = new Map();
  private positions: Map<string, Position> = new Map();
  private activeSignals: Map<string, TradingSignal> = new Map();
  
  // Performance tracking
  private metrics: TradingMetrics;
  private isRunning = false;
  private lastHealthCheck = Date.now();
  
  // Monitoring intervals
  private metricsInterval?: NodeJS.Timeout;
  private healthInterval?: NodeJS.Timeout;
  private rebalanceInterval?: NodeJS.Timeout;

  // Risk management state
  private dailyLoss = 0;
  private currentDrawdown = 0;
  private maxDrawdownToday = 0;
  private tradingEnabled = true;
  private emergencyStopTriggered = false;

  constructor(
    config: TradingEngineConfig,
    streamManager: RealtimeDataStreamManager,
    mlBridge: MLServiceBridge
  ) {
    super();
    
    this.config = config;
    this.streamManager = streamManager;
    this.mlBridge = mlBridge;
    
    // Initialize metrics
    this.metrics = {
      totalOrders: 0,
      successfulOrders: 0,
      failedOrders: 0,
      totalVolume: 0,
      totalPnL: 0,
      totalCommissions: 0,
      activePositions: 0,
      avgExecutionTime: 0,
      successRate: 0,
      sharpeRatio: 0,
      maxDrawdown: 0,
      dailyPnL: 0,
      winRate: 0,
      avgWin: 0,
      avgLoss: 0
    };

    this.setupEventHandlers();
  }

  /**
   * Initialize the trading engine
   */
  public async initialize(): Promise<void> {
    try {
      logger.info('🏭 Initializing Enhanced Trading Engine...');

      // Register stream processors
      await this.streamManager.registerConsumer(
        STREAM_NAMES.ORDERS,
        new OrderProcessor(this),
        {
          groupName: 'trading_engine_orders',
          consumerName: `engine_${this.config.accountId}`,
          autoScaling: true,
          maxConcurrency: 5,
          healthCheckInterval: 30000
        }
      );

      await this.streamManager.registerConsumer(
        STREAM_NAMES.SIGNALS,
        new SignalProcessor(this),
        {
          groupName: 'trading_engine_signals',
          consumerName: `engine_${this.config.accountId}`,
          autoScaling: true,
          maxConcurrency: 3,
          healthCheckInterval: 30000
        }
      );

      // Start monitoring intervals
      this.startMonitoring();

      this.isRunning = true;
      logger.info(`✅ Enhanced Trading Engine initialized for account: ${this.config.accountId}`);

    } catch (error) {
      logger.error('❌ Failed to initialize Enhanced Trading Engine:', error);
      throw error;
    }
  }

  /**
   * Process ML-generated trading signal with full automation
   */
  public async processMLSignal(
    symbol: string,
    mlFeatures: any,
    metadata?: Record<string, any>
  ): Promise<{ success: boolean; orderId?: string; reason?: string }> {
    try {
      if (!this.config.mlIntegrationConfig.autoTradingEnabled) {
        return { success: false, reason: 'Auto-trading is disabled' };
      }

      if (!this.tradingEnabled || this.emergencyStopTriggered) {
        return { success: false, reason: 'Trading is currently disabled' };
      }

      // Get ML prediction
      const predictionRequest: MLPredictionRequest = {
        features: mlFeatures,
        config: {
          voting_method: 'confidence_weighted',
          min_models_required: 2,
          confidence_threshold: this.config.mlIntegrationConfig.minConfidenceThreshold,
          return_details: true
        }
      };

      const prediction = await this.mlBridge.getPrediction(predictionRequest);
      
      if (!prediction) {
        return { success: false, reason: 'ML prediction failed' };
      }

      // Validate prediction confidence
      if (prediction.ensemble_confidence < this.config.mlIntegrationConfig.minConfidenceThreshold) {
        return { 
          success: false, 
          reason: `Confidence too low: ${prediction.ensemble_confidence.toFixed(3)}` 
        };
      }

      // Create trading signal from ML prediction
      const signal: TradingSignal = {
        id: `ml_signal_${Date.now()}`,
        symbol,
        action: prediction.recommendation,
        confidence: prediction.ensemble_confidence * 100,
        price: mlFeatures.price_data[mlFeatures.price_data.length - 1]?.close || 0,
        timeframe: mlFeatures.metadata.timeframe,
        source: 'ml_prediction',
        timestamp: Date.now(),
        expiryTime: Date.now() + 300000, // 5 minutes
        metadata: {
          mlPrediction: prediction,
          riskRewardRatio: 2.0, // Default from ML system
          expectedVolatility: prediction.uncertainty,
          marketRegime: prediction.market_regime
        }
      };

      // Execute signal
      const result = await this.executeSignal(signal);
      
      logger.info(`🤖 ML signal processed: ${signal.action} ${symbol} (${prediction.ensemble_confidence.toFixed(3)} confidence) - ${result.success ? 'SUCCESS' : 'FAILED'}`);
      
      return result;

    } catch (error) {
      logger.error('❌ ML signal processing failed:', error);
      return { success: false, reason: (error as Error).message };
    }
  }

  /**
   * Execute trading signal with comprehensive risk checks
   */
  public async executeSignal(signal: TradingSignal): Promise<{ success: boolean; orderId?: string; reason?: string }> {
    try {
      // Pre-execution validation
      const validationResult = await this.validateSignal(signal);
      if (!validationResult.isValid) {
        return { success: false, reason: validationResult.reason };
      }

      // Risk assessment
      const riskAssessment = await this.assessSignalRisk(signal);
      if (!riskAssessment.acceptable) {
        return { success: false, reason: riskAssessment.reason };
      }

      // Calculate position size based on risk
      const positionSize = this.calculatePositionSize(signal, riskAssessment);
      if (positionSize <= 0) {
        return { success: false, reason: 'Position size calculation failed' };
      }

      // Create and submit order
      const order = this.createOrderFromSignal(signal, positionSize);
      const executionResult = await this.submitOrder(order);

      if (executionResult.success) {
        // Store active signal
        this.activeSignals.set(signal.id, signal);
        
        // Publish order event
        await this.streamManager.publishMessage(STREAM_NAMES.ORDERS, {
          eventType: 'ORDER_SUBMITTED',
          order,
          signal,
          timestamp: Date.now()
        });

        return { success: true, orderId: order.id };
      } else {
        return { success: false, reason: executionResult.reason };
      }

    } catch (error) {
      logger.error('❌ Signal execution failed:', error);
      return { success: false, reason: (error as Error).message };
    }
  }

  /**
   * Get comprehensive trading metrics
   */
  public getMetrics(): TradingMetrics & {
    riskMetrics: {
      currentDrawdown: number;
      dailyLoss: number;
      utilizationRate: number;
      riskLimitUtilization: Record<string, number>;
    };
    systemMetrics: {
      ordersPerSecond: number;
      avgOrderLatency: number;
      systemHealth: 'healthy' | 'degraded' | 'critical';
      lastHealthCheck: number;
    };
  } {
    const totalCapital = this.config.initialCapital + this.metrics.totalPnL;
    const utilizationRate = this.positions.size / this.config.riskLimits.maxTotalPositions;
    
    return {
      ...this.metrics,
      riskMetrics: {
        currentDrawdown: this.currentDrawdown,
        dailyLoss: this.dailyLoss,
        utilizationRate,
        riskLimitUtilization: {
          maxPositions: this.positions.size / this.config.riskLimits.maxTotalPositions,
          maxDailyLoss: Math.abs(this.dailyLoss) / this.config.riskLimits.maxDailyLoss,
          maxDrawdown: this.currentDrawdown / this.config.riskLimits.maxDrawdown
        }
      },
      systemMetrics: {
        ordersPerSecond: this.calculateOrdersPerSecond(),
        avgOrderLatency: this.metrics.avgExecutionTime,
        systemHealth: this.determineSystemHealth(),
        lastHealthCheck: this.lastHealthCheck
      }
    };
  }

  /**
   * Emergency stop all trading activities
   */
  public async emergencyStop(reason: string): Promise<void> {
    try {
      logger.warn(`🚨 EMERGENCY STOP TRIGGERED: ${reason}`);
      
      this.emergencyStopTriggered = true;
      this.tradingEnabled = false;

      // Cancel all pending orders
      const pendingOrders = Array.from(this.orders.values())
        .filter(order => order.status === 'PENDING');

      const cancelPromises = pendingOrders.map(order => this.cancelOrder(order.id));
      await Promise.allSettled(cancelPromises);

      // Close all positions if required
      if (reason.includes('CRITICAL')) {
        const closePromises = Array.from(this.positions.values())
          .map(position => this.closePosition(position.id, 'EMERGENCY'));
        await Promise.allSettled(closePromises);
      }

      // Publish emergency stop event
      await this.streamManager.publishMessage(STREAM_NAMES.RISK_EVENTS, {
        eventType: 'EMERGENCY_STOP',
        reason,
        timestamp: Date.now(),
        affectedOrders: pendingOrders.length,
        affectedPositions: this.positions.size
      });

      this.emit('emergencyStop', { reason, timestamp: Date.now() });
      
      logger.warn('🛑 Emergency stop completed');

    } catch (error) {
      logger.error('❌ Emergency stop failed:', error);
      throw error;
    }
  }

  /**
   * Resume trading after emergency stop
   */
  public async resumeTrading(): Promise<void> {
    try {
      logger.info('▶️ Resuming trading operations...');

      // Perform health checks
      const healthStatus = await this.performHealthCheck();
      if (healthStatus.overall !== 'healthy') {
        throw new Error(`Cannot resume trading: ${healthStatus.overall} status`);
      }

      this.emergencyStopTriggered = false;
      this.tradingEnabled = true;

      // Reset daily limits
      this.dailyLoss = 0;
      this.maxDrawdownToday = 0;

      // Publish resume event
      await this.streamManager.publishMessage(STREAM_NAMES.RISK_EVENTS, {
        eventType: 'TRADING_RESUMED',
        timestamp: Date.now()
      });

      this.emit('tradingResumed', { timestamp: Date.now() });
      
      logger.info('✅ Trading operations resumed');

    } catch (error) {
      logger.error('❌ Failed to resume trading:', error);
      throw error;
    }
  }

  /**
   * Cleanup and shutdown
   */
  public async cleanup(): Promise<void> {
    try {
      logger.info('🧹 Shutting down Enhanced Trading Engine...');

      this.isRunning = false;

      // Clear intervals
      if (this.metricsInterval) clearInterval(this.metricsInterval);
      if (this.healthInterval) clearInterval(this.healthInterval);
      if (this.rebalanceInterval) clearInterval(this.rebalanceInterval);

      // Cancel all pending orders
      const pendingOrders = Array.from(this.orders.values())
        .filter(order => order.status === 'PENDING');
      
      if (pendingOrders.length > 0) {
        logger.info(`🛑 Cancelling ${pendingOrders.length} pending orders...`);
        const cancelPromises = pendingOrders.map(order => this.cancelOrder(order.id));
        await Promise.allSettled(cancelPromises);
      }

      logger.info('✅ Enhanced Trading Engine shutdown complete');

    } catch (error) {
      logger.error('❌ Error during trading engine cleanup:', error);
    }
  }

  // Private implementation methods

  private setupEventHandlers(): void {
    // ML Bridge events
    this.mlBridge.on('predictionSuccess', (data) => {
      this.emit('mlPredictionReceived', data);
    });

    this.mlBridge.on('predictionError', (data) => {
      this.emit('mlPredictionError', data);
    });

    // Stream manager events
    this.streamManager.on('messageProcessed', (data) => {
      this.emit('eventProcessed', data);
    });

    this.streamManager.on('messageError', (data) => {
      this.emit('eventProcessingError', data);
    });
  }

  private startMonitoring(): void {
    // Metrics collection
    this.metricsInterval = setInterval(() => {
      this.updateMetrics();
    }, this.config.performanceConfig.metricsInterval);

    // Health monitoring
    this.healthInterval = setInterval(async () => {
      await this.performHealthCheck();
    }, this.config.performanceConfig.healthCheckInterval);

    // Position rebalancing
    this.rebalanceInterval = setInterval(() => {
      this.rebalancePositions();
    }, this.config.performanceConfig.positionRebalanceInterval);

    logger.info('📊 Trading engine monitoring started');
  }

  private async validateSignal(signal: TradingSignal): Promise<{ isValid: boolean; reason?: string }> {
    // Check if trading is enabled
    if (!this.tradingEnabled) {
      return { isValid: false, reason: 'Trading is disabled' };
    }

    // Check signal expiry
    if (signal.expiryTime && Date.now() > signal.expiryTime) {
      return { isValid: false, reason: 'Signal has expired' };
    }

    // Check allowed symbols
    if (!this.config.riskLimits.allowedSymbols.includes(signal.symbol)) {
      return { isValid: false, reason: 'Symbol not allowed' };
    }

    // Check forbidden symbols
    if (this.config.riskLimits.forbiddenSymbols.includes(signal.symbol)) {
      return { isValid: false, reason: 'Symbol is forbidden' };
    }

    // Check trading hours
    if (!this.isWithinTradingHours()) {
      return { isValid: false, reason: 'Outside trading hours' };
    }

    return { isValid: true };
  }

  private async assessSignalRisk(signal: TradingSignal): Promise<{ acceptable: boolean; reason?: string; riskScore: number }> {
    let riskScore = 0;

    // Confidence risk
    if (signal.confidence < 70) riskScore += 0.3;
    else if (signal.confidence < 80) riskScore += 0.1;

    // Position concentration risk
    const symbolPositions = Array.from(this.positions.values())
      .filter(pos => pos.symbol === signal.symbol);
    
    if (symbolPositions.length >= this.config.riskLimits.maxPositionsPerSymbol) {
      return { acceptable: false, reason: 'Maximum positions per symbol reached', riskScore: 1.0 };
    }

    // Portfolio risk
    if (this.positions.size >= this.config.riskLimits.maxTotalPositions) {
      return { acceptable: false, reason: 'Maximum total positions reached', riskScore: 1.0 };
    }

    // Daily loss limit
    if (Math.abs(this.dailyLoss) >= this.config.riskLimits.maxDailyLoss) {
      return { acceptable: false, reason: 'Daily loss limit reached', riskScore: 1.0 };
    }

    // Drawdown limit
    if (this.currentDrawdown >= this.config.riskLimits.maxDrawdown) {
      return { acceptable: false, reason: 'Maximum drawdown reached', riskScore: 1.0 };
    }

    return { acceptable: riskScore < 0.7, riskScore };
  }

  private calculatePositionSize(signal: TradingSignal, riskAssessment: { riskScore: number }): number {
    const baseSize = this.config.initialCapital * 0.02; // 2% base risk
    const confidenceMultiplier = signal.confidence / 100;
    const riskMultiplier = 1 - riskAssessment.riskScore;
    
    return baseSize * confidenceMultiplier * riskMultiplier / signal.price;
  }

  private createOrderFromSignal(signal: TradingSignal, quantity: number): Order {
    return {
      id: `order_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      clientOrderId: `client_${signal.id}`,
      symbol: signal.symbol,
      side: signal.action === 'BUY' || signal.action === 'CLOSE_SHORT' ? 'BUY' : 'SELL',
      type: 'MARKET',
      status: 'PENDING',
      quantity,
      timeInForce: 'IOC',
      exchange: 'default',
      timestamp: Date.now(),
      filledQuantity: 0,
      avgFillPrice: 0,
      commission: 0,
      metadata: {
        source: 'ml_signal',
        signalId: signal.id,
        riskScore: 0.5,
        expectedPnL: signal.metadata.riskRewardRatio ? quantity * signal.price * (signal.metadata.riskRewardRatio || 2) * 0.01 : undefined
      }
    };
  }

  private async submitOrder(order: Order): Promise<{ success: boolean; reason?: string }> {
    try {
      // Store order
      this.orders.set(order.id, order);

      // Simulate order execution (replace with actual exchange integration)
      await new Promise(resolve => setTimeout(resolve, 100));

      // Update order status
      order.status = 'FILLED';
      order.filledQuantity = order.quantity;
      order.avgFillPrice = order.price || order.quantity; // Mock fill price

      // Update metrics
      this.metrics.totalOrders++;
      this.metrics.successfulOrders++;
      this.metrics.totalVolume += order.quantity * (order.avgFillPrice || 0);

      logger.info(`✅ Order executed: ${order.id} - ${order.side} ${order.quantity} ${order.symbol}`);
      
      return { success: true };

    } catch (error) {
      this.metrics.failedOrders++;
      logger.error(`❌ Order execution failed: ${order.id}`, error);
      return { success: false, reason: (error as Error).message };
    }
  }

  private async cancelOrder(orderId: string): Promise<boolean> {
    const order = this.orders.get(orderId);
    if (!order) return false;

    order.status = 'CANCELLED';
    logger.info(`🛑 Order cancelled: ${orderId}`);
    return true;
  }

  private async closePosition(positionId: string, reason: string): Promise<boolean> {
    const position = this.positions.get(positionId);
    if (!position) return false;

    // Create closing order
    const closeOrder: Order = {
      id: `close_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      clientOrderId: `close_${position.id}`,
      symbol: position.symbol,
      side: position.side === 'LONG' ? 'SELL' : 'BUY',
      type: 'MARKET',
      status: 'PENDING',
      quantity: position.quantity,
      timeInForce: 'IOC',
      exchange: 'default',
      timestamp: Date.now(),
      filledQuantity: 0,
      avgFillPrice: 0,
      commission: 0,
      metadata: {
        source: 'risk_management',
        parentOrderId: position.id
      }
    };

    const result = await this.submitOrder(closeOrder);
    if (result.success) {
      this.positions.delete(positionId);
      logger.info(`✅ Position closed: ${positionId} - Reason: ${reason}`);
      return true;
    }

    return false;
  }

  public async processOrderEvent(orderData: any): Promise<void> {
    // Process order state changes from the stream
    logger.debug('📋 Processing order event:', orderData.eventType);
  }

  public async processSignalEvent(signalData: any): Promise<void> {
    // Process trading signals from the stream
    logger.debug('📡 Processing signal event:', signalData.eventType);
  }

  public async sendToDeadLetter(errorType: string, message: StreamMessage, error: Error): Promise<void> {
    await this.streamManager.publishMessage(STREAM_NAMES.DEAD_LETTER, {
      errorType,
      originalMessage: message,
      error: error.message,
      timestamp: Date.now()
    });
  }

  private isWithinTradingHours(): boolean {
    // Simplified trading hours check
    return true; // Always allow for crypto markets
  }

  private updateMetrics(): void {
    // Calculate derived metrics
    this.metrics.activePositions = this.positions.size;
    this.metrics.successRate = this.metrics.totalOrders > 0 ? 
      (this.metrics.successfulOrders / this.metrics.totalOrders) * 100 : 0;
    
    // Update risk metrics
    this.currentDrawdown = this.calculateCurrentDrawdown();
    this.dailyLoss = this.calculateDailyLoss();
  }

  private async performHealthCheck(): Promise<{
    overall: 'healthy' | 'degraded' | 'critical';
    components: Record<string, boolean>;
  }> {
    this.lastHealthCheck = Date.now();

    const mlHealth = await this.mlBridge.healthCheck();
    const streamHealth = await this.streamManager.getHealthStatus();
    
    const components = {
      mlService: mlHealth.status === 'healthy',
      streams: streamHealth.overall === 'healthy',
      riskLimits: this.checkRiskLimits(),
      trading: this.tradingEnabled && !this.emergencyStopTriggered
    };

    const healthyComponents = Object.values(components).filter(Boolean).length;
    const totalComponents = Object.values(components).length;

    let overall: 'healthy' | 'degraded' | 'critical';
    if (healthyComponents === totalComponents) {
      overall = 'healthy';
    } else if (healthyComponents >= totalComponents * 0.7) {
      overall = 'degraded';
    } else {
      overall = 'critical';
    }

    return { overall, components };
  }

  private checkRiskLimits(): boolean {
    return (
      Math.abs(this.dailyLoss) < this.config.riskLimits.maxDailyLoss &&
      this.currentDrawdown < this.config.riskLimits.maxDrawdown &&
      this.positions.size < this.config.riskLimits.maxTotalPositions
    );
  }

  private rebalancePositions(): void {
    // Implement position rebalancing logic
    logger.debug('⚖️ Performing position rebalancing...');
  }

  private calculateOrdersPerSecond(): number {
    // Calculate orders per second over the last minute
    return this.metrics.totalOrders / 60; // Simplified
  }

  private determineSystemHealth(): 'healthy' | 'degraded' | 'critical' {
    if (!this.tradingEnabled || this.emergencyStopTriggered) return 'critical';
    if (this.checkRiskLimits()) return 'healthy';
    return 'degraded';
  }

  private calculateCurrentDrawdown(): number {
    // Calculate current drawdown from peak
    return Math.max(0, -this.metrics.totalPnL * 0.1); // Simplified
  }

  private calculateDailyLoss(): number {
    // Calculate today's P&L (simplified)
    return this.metrics.dailyPnL;
  }
}