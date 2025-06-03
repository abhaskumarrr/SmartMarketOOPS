/**
 * Event-Driven Trading System
 * Main orchestrator for event-driven trading architecture
 */

import { EventEmitter } from 'events';
import { redisStreamsService } from './redisStreamsService';
import { eventProcessingPipeline } from './eventProcessingPipeline';
import { MarketDataProcessor } from '../processors/marketDataProcessor';
import { SignalProcessor } from '../processors/signalProcessor';
import { logger } from '../utils/logger';
import {
  TradingEvent,
  MarketDataEvent,
  TradingSignalEvent,
  OrderEvent,
  SystemEvent,
  BotEvent,
  STREAM_NAMES,
  createEventId,
  createCorrelationId,
} from '../types/events';

export interface TradingSystemConfig {
  enableMarketDataProcessing: boolean;
  enableSignalProcessing: boolean;
  enableOrderProcessing: boolean;
  enableRiskManagement: boolean;
  enablePortfolioManagement: boolean;
  enableSystemMonitoring: boolean;
  marketDataSources: string[];
  signalGenerationModels: string[];
  riskManagementRules: string[];
}

export interface TradingSystemStats {
  uptime: number;
  eventsProcessed: number;
  eventsPerSecond: number;
  activeStreams: number;
  activeProcessors: number;
  systemHealth: 'HEALTHY' | 'DEGRADED' | 'UNHEALTHY';
  lastEventTime: number;
  errors: number;
}

export class EventDrivenTradingSystem extends EventEmitter {
  private static instance: EventDrivenTradingSystem;
  private isRunning: boolean = false;
  private startTime: number = 0;
  private config: TradingSystemConfig;
  private stats: TradingSystemStats;
  private processors: Map<string, any> = new Map();

  private constructor(config: Partial<TradingSystemConfig> = {}) {
    super();
    
    this.config = {
      enableMarketDataProcessing: true,
      enableSignalProcessing: true,
      enableOrderProcessing: true,
      enableRiskManagement: true,
      enablePortfolioManagement: true,
      enableSystemMonitoring: true,
      marketDataSources: ['delta-exchange', 'binance'],
      signalGenerationModels: ['transformer-v1', 'price-change-detector'],
      riskManagementRules: ['position-size', 'daily-loss', 'drawdown'],
      ...config,
    };

    this.stats = {
      uptime: 0,
      eventsProcessed: 0,
      eventsPerSecond: 0,
      activeStreams: 0,
      activeProcessors: 0,
      systemHealth: 'HEALTHY',
      lastEventTime: 0,
      errors: 0,
    };
  }

  public static getInstance(config?: Partial<TradingSystemConfig>): EventDrivenTradingSystem {
    if (!EventDrivenTradingSystem.instance) {
      EventDrivenTradingSystem.instance = new EventDrivenTradingSystem(config);
    }
    return EventDrivenTradingSystem.instance;
  }

  /**
   * Initialize and start the event-driven trading system
   */
  public async start(): Promise<void> {
    if (this.isRunning) {
      logger.warn('⚠️ Event-driven trading system is already running');
      return;
    }

    try {
      logger.info('🚀 Starting event-driven trading system...');
      
      // Initialize Redis Streams service
      await redisStreamsService.initialize();

      // Register event processors
      await this.registerProcessors();

      // Start event processing pipeline
      await eventProcessingPipeline.start();

      // Set up event listeners
      this.setupEventListeners();

      // Start system monitoring
      this.startSystemMonitoring();

      this.isRunning = true;
      this.startTime = Date.now();

      // Publish system started event
      await this.publishSystemEvent('SYSTEM_STARTED', {
        component: 'event-driven-trading-system',
        status: 'HEALTHY',
        message: 'Event-driven trading system started successfully',
        uptime: 0,
      });

      this.emit('started');
      logger.info('✅ Event-driven trading system started successfully');

    } catch (error) {
      logger.error('❌ Failed to start event-driven trading system:', error);
      this.isRunning = false;
      throw error;
    }
  }

  /**
   * Stop the event-driven trading system
   */
  public async stop(): Promise<void> {
    if (!this.isRunning) {
      logger.warn('⚠️ Event-driven trading system is not running');
      return;
    }

    try {
      logger.info('🛑 Stopping event-driven trading system...');

      // Publish system stopping event
      await this.publishSystemEvent('SYSTEM_STOPPED', {
        component: 'event-driven-trading-system',
        status: 'DOWN',
        message: 'Event-driven trading system stopping',
        uptime: Date.now() - this.startTime,
      });

      // Stop event processing pipeline
      await eventProcessingPipeline.stop();

      // Shutdown Redis Streams service
      await redisStreamsService.shutdown();

      this.isRunning = false;
      this.emit('stopped');
      
      logger.info('✅ Event-driven trading system stopped successfully');

    } catch (error) {
      logger.error('❌ Error stopping event-driven trading system:', error);
      throw error;
    }
  }

  /**
   * Register event processors
   */
  private async registerProcessors(): Promise<void> {
    try {
      // Market Data Processor
      if (this.config.enableMarketDataProcessing) {
        const marketDataProcessor = new MarketDataProcessor();
        eventProcessingPipeline.registerProcessor(marketDataProcessor);
        this.processors.set('marketData', marketDataProcessor);
        logger.info('📊 Registered market data processor');
      }

      // Signal Processor
      if (this.config.enableSignalProcessing) {
        const signalProcessor = new SignalProcessor();
        eventProcessingPipeline.registerProcessor(signalProcessor);
        this.processors.set('signal', signalProcessor);
        logger.info('🎯 Registered signal processor');
      }

      // TODO: Add more processors as needed
      // - Order Processor
      // - Risk Management Processor
      // - Portfolio Management Processor
      // - System Monitoring Processor

      this.stats.activeProcessors = this.processors.size;

    } catch (error) {
      logger.error('❌ Failed to register processors:', error);
      throw error;
    }
  }

  /**
   * Set up event listeners for monitoring
   */
  private setupEventListeners(): void {
    // Listen to pipeline events
    eventProcessingPipeline.on('eventProcessed', (data) => {
      this.stats.eventsProcessed++;
      this.stats.lastEventTime = Date.now();
      this.emit('eventProcessed', data);
    });

    eventProcessingPipeline.on('eventFailed', (data) => {
      this.stats.errors++;
      this.emit('eventFailed', data);
      logger.error(`❌ Event processing failed:`, data);
    });

    eventProcessingPipeline.on('circuitBreakerOpened', () => {
      this.stats.systemHealth = 'DEGRADED';
      this.emit('circuitBreakerOpened');
      logger.warn('🚨 Circuit breaker opened - system degraded');
    });

    eventProcessingPipeline.on('circuitBreakerClosed', () => {
      this.stats.systemHealth = 'HEALTHY';
      this.emit('circuitBreakerClosed');
      logger.info('✅ Circuit breaker closed - system healthy');
    });

    eventProcessingPipeline.on('statsUpdated', (pipelineStats) => {
      this.updateSystemStats(pipelineStats);
    });
  }

  /**
   * Start system monitoring
   */
  private startSystemMonitoring(): void {
    const monitoringInterval = setInterval(async () => {
      if (!this.isRunning) {
        clearInterval(monitoringInterval);
        return;
      }

      try {
        // Update uptime
        this.stats.uptime = Date.now() - this.startTime;

        // Calculate events per second
        this.stats.eventsPerSecond = this.stats.eventsProcessed / (this.stats.uptime / 1000);

        // Health check
        const isHealthy = await this.performHealthCheck();
        if (!isHealthy && this.stats.systemHealth === 'HEALTHY') {
          this.stats.systemHealth = 'UNHEALTHY';
          await this.publishSystemEvent('SYSTEM_ALERT', {
            component: 'event-driven-trading-system',
            status: 'UNHEALTHY',
            message: 'System health check failed',
            uptime: this.stats.uptime,
          });
        }

        // Emit stats update
        this.emit('statsUpdated', this.stats);

      } catch (error) {
        logger.error('❌ System monitoring error:', error);
      }
    }, 10000); // Every 10 seconds
  }

  /**
   * Perform system health check
   */
  private async performHealthCheck(): Promise<boolean> {
    try {
      // Check Redis Streams service
      const redisHealthy = await redisStreamsService.healthCheck();
      
      // Check event processing pipeline
      const pipelineHealthy = await eventProcessingPipeline.healthCheck();

      return redisHealthy && pipelineHealthy;
    } catch (error) {
      logger.error('❌ Health check failed:', error);
      return false;
    }
  }

  /**
   * Update system statistics
   */
  private updateSystemStats(pipelineStats: any): void {
    // Update stats based on pipeline statistics
    this.stats.eventsProcessed = pipelineStats.processedEvents;
    this.stats.errors = pipelineStats.failedEvents;
    
    // Determine system health
    if (pipelineStats.circuitBreakerOpen) {
      this.stats.systemHealth = 'DEGRADED';
    } else if (this.stats.errors > 100) {
      this.stats.systemHealth = 'UNHEALTHY';
    } else {
      this.stats.systemHealth = 'HEALTHY';
    }
  }

  // ============================================================================
  // EVENT PUBLISHING METHODS
  // ============================================================================

  /**
   * Publish market data event
   */
  public async publishMarketDataEvent(
    symbol: string,
    price: number,
    volume: number,
    exchange: string = 'default',
    additionalData: Partial<MarketDataEvent['data']> = {}
  ): Promise<string> {
    const event: MarketDataEvent = {
      id: createEventId(),
      type: 'MARKET_DATA_RECEIVED',
      timestamp: Date.now(),
      version: '1.0',
      source: 'trading-system',
      correlationId: createCorrelationId(),
      data: {
        symbol,
        exchange,
        price,
        volume,
        timestamp: Date.now(),
        ...additionalData,
      },
    };

    return await redisStreamsService.publishEvent(STREAM_NAMES.MARKET_DATA, event);
  }

  /**
   * Publish trading signal event
   */
  public async publishTradingSignalEvent(
    signalData: Partial<TradingSignalEvent['data']>,
    correlationId?: string
  ): Promise<string> {
    const event: TradingSignalEvent = {
      id: createEventId(),
      type: 'SIGNAL_GENERATED',
      timestamp: Date.now(),
      version: '1.0',
      source: 'trading-system',
      correlationId: correlationId || createCorrelationId(),
      data: {
        signalId: createEventId(),
        symbol: '',
        signalType: 'ENTRY',
        direction: 'LONG',
        strength: 'MODERATE',
        timeframe: '1m',
        price: 0,
        confidenceScore: 0,
        expectedReturn: 0,
        expectedRisk: 0,
        riskRewardRatio: 1,
        modelSource: 'unknown',
        ...signalData,
      } as TradingSignalEvent['data'],
    };

    return await redisStreamsService.publishEvent(STREAM_NAMES.TRADING_SIGNALS, event);
  }

  /**
   * Publish system event
   */
  public async publishSystemEvent(
    type: SystemEvent['type'],
    data: Partial<SystemEvent['data']>
  ): Promise<string> {
    const event: SystemEvent = {
      id: createEventId(),
      type,
      timestamp: Date.now(),
      version: '1.0',
      source: 'trading-system',
      data: {
        component: 'unknown',
        status: 'HEALTHY',
        message: '',
        ...data,
      } as SystemEvent['data'],
    };

    return await redisStreamsService.publishEvent(STREAM_NAMES.SYSTEM, event);
  }

  /**
   * Publish bot event
   */
  public async publishBotEvent(
    type: BotEvent['type'],
    botData: Partial<BotEvent['data']>,
    userId?: string
  ): Promise<string> {
    const event: BotEvent = {
      id: createEventId(),
      type,
      timestamp: Date.now(),
      version: '1.0',
      source: 'trading-system',
      userId,
      data: {
        botId: '',
        botName: '',
        status: 'STOPPED',
        symbol: '',
        strategy: '',
        timeframe: '',
        ...botData,
      } as BotEvent['data'],
    };

    return await redisStreamsService.publishEvent(STREAM_NAMES.BOTS, event);
  }

  // ============================================================================
  // UTILITY METHODS
  // ============================================================================

  /**
   * Get system statistics
   */
  public getStats(): TradingSystemStats {
    return { ...this.stats };
  }

  /**
   * Get system configuration
   */
  public getConfig(): TradingSystemConfig {
    return { ...this.config };
  }

  /**
   * Update system configuration
   */
  public updateConfig(newConfig: Partial<TradingSystemConfig>): void {
    this.config = { ...this.config, ...newConfig };
    logger.info('⚙️ System configuration updated');
  }

  /**
   * Get processor statistics
   */
  public getProcessorStats(): Record<string, any> {
    const stats: Record<string, any> = {};
    
    for (const [name, processor] of this.processors) {
      if (typeof processor.getStats === 'function') {
        stats[name] = processor.getStats();
      }
    }

    return stats;
  }

  /**
   * Check if system is running
   */
  public isSystemRunning(): boolean {
    return this.isRunning;
  }

  /**
   * Get system health status
   */
  public getHealthStatus(): 'HEALTHY' | 'DEGRADED' | 'UNHEALTHY' {
    return this.stats.systemHealth;
  }
}

// Export singleton instance
export const eventDrivenTradingSystem = EventDrivenTradingSystem.getInstance();
