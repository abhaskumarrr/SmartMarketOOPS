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

// Import required services
import prisma from '../utils/prismaClient';
import { orderExecutionService } from './trading/orderExecutionService';
import { riskAssessmentService } from './trading/riskAssessmentService';

// Processor class definitions
class OrderProcessor {
  async process(event: TradingEvent): Promise<TradingEvent | null> {
    if (event.type !== 'order') return event;
    
    logger.debug('Processing order event', { eventId: event.id, orderId: event.data?.orderId });
    
    try {
      const orderEvent = event as OrderEvent;
      
      // Validate order data
      if (!orderEvent.data || !orderEvent.data.symbol || !orderEvent.data.side || !orderEvent.data.amount) {
        logger.error('Invalid order event data', { eventId: event.id });
        return null;
      }
      
      // Process different order actions
      switch (orderEvent.data.action) {
        case 'create':
          await this.processOrderCreation(orderEvent);
          break;
        case 'update':
          await this.processOrderUpdate(orderEvent);
          break;
        case 'cancel':
          await this.processOrderCancellation(orderEvent);
          break;
        case 'fill':
          await this.processOrderFill(orderEvent);
          break;
        default:
          logger.warn('Unknown order action', { action: orderEvent.data.action, eventId: event.id });
      }
      
      return event;
    } catch (error) {
      logger.error('Error processing order event', { error, eventId: event.id });
      return null;
    }
  }
  
  private async processOrderCreation(event: OrderEvent) {
    try {
      // Create order record in database
      await prisma.order.create({
        data: {
          userId: event.data.userId,
          botId: event.data.botId,
          symbol: event.data.symbol,
          side: event.data.side,
          amount: event.data.amount,
          price: event.data.price,
          type: event.data.type || 'market',
          status: 'pending',
          metadata: {
            eventId: event.id,
            correlationId: event.correlationId
          }
        }
      });
      
      logger.info('Order created successfully', { 
        orderId: event.data.orderId,
        symbol: event.data.symbol,
        side: event.data.side,
        amount: event.data.amount
      });
    } catch (error) {
      logger.error('Error creating order', { error, eventId: event.id });
      throw error;
    }
  }
  
  private async processOrderUpdate(event: OrderEvent) {
    try {
      if (!event.data.orderId) {
        throw new Error('Order ID required for update');
      }
      
      await prisma.order.update({
        where: { id: event.data.orderId },
        data: {
          status: event.data.status,
          filledAmount: event.data.filledAmount,
          averagePrice: event.data.averagePrice,
          metadata: {
            lastUpdated: new Date().toISOString(),
            eventId: event.id
          }
        }
      });
      
      logger.info('Order updated successfully', { orderId: event.data.orderId });
    } catch (error) {
      logger.error('Error updating order', { error, eventId: event.id });
      throw error;
    }
  }
  
  private async processOrderCancellation(event: OrderEvent) {
    try {
      if (!event.data.orderId) {
        throw new Error('Order ID required for cancellation');
      }
      
      await prisma.order.update({
        where: { id: event.data.orderId },
        data: {
          status: 'cancelled',
          metadata: {
            cancelledAt: new Date().toISOString(),
            eventId: event.id
          }
        }
      });
      
      logger.info('Order cancelled successfully', { orderId: event.data.orderId });
    } catch (error) {
      logger.error('Error cancelling order', { error, eventId: event.id });
      throw error;
    }
  }
  
  private async processOrderFill(event: OrderEvent) {
    try {
      if (!event.data.orderId) {
        throw new Error('Order ID required for fill');
      }
      
      await prisma.order.update({
        where: { id: event.data.orderId },
        data: {
          status: 'filled',
          filledAmount: event.data.amount,
          averagePrice: event.data.price,
          executedAt: new Date(),
          metadata: {
            filledAt: new Date().toISOString(),
            eventId: event.id
          }
        }
      });
      
      // Create position if this is a new position
      if (event.data.createPosition) {
        await this.createPositionFromOrder(event);
      }
      
      logger.info('Order filled successfully', { 
        orderId: event.data.orderId,
        filledAmount: event.data.amount,
        averagePrice: event.data.price
      });
    } catch (error) {
      logger.error('Error processing order fill', { error, eventId: event.id });
      throw error;
    }
  }
  
  private async createPositionFromOrder(event: OrderEvent) {
    try {
      await prisma.position.create({
        data: {
          userId: event.data.userId,
          botId: event.data.botId,
          symbol: event.data.symbol,
          side: event.data.side === 'buy' ? 'Long' : 'Short',
          entryPrice: event.data.price,
          currentPrice: event.data.price,
          amount: event.data.amount,
          leverage: event.data.leverage || 1.0,
          status: 'Open',
          metadata: {
            orderId: event.data.orderId,
            eventId: event.id
          }
        }
      });
      
      logger.info('Position created from order', { 
        orderId: event.data.orderId,
        symbol: event.data.symbol,
        side: event.data.side
      });
    } catch (error) {
      logger.error('Error creating position from order', { error, eventId: event.id });
      throw error;
    }
  }
}

class RiskManagementProcessor {
  async process(event: TradingEvent): Promise<TradingEvent | null> {
    logger.debug('Processing event for risk management', { eventId: event.id, type: event.type });
    
    try {
      // Apply risk management based on event type
      switch (event.type) {
        case 'order':
          return await this.processOrderRisk(event as OrderEvent);
        case 'signal':
          return await this.processSignalRisk(event as TradingSignalEvent);
        case 'market_data':
          return await this.processMarketRisk(event as MarketDataEvent);
        default:
          return event;
      }
    } catch (error) {
      logger.error('Error in risk management processing', { error, eventId: event.id });
      return null;
    }
  }
  
  private async processOrderRisk(event: OrderEvent): Promise<TradingEvent | null> {
    try {
      // Get user's risk settings
      const riskSettings = await prisma.riskSettings.findFirst({
        where: {
          userId: event.data.userId,
          botId: event.data.botId,
          isActive: true
        }
      });
      
      if (!riskSettings) {
        logger.warn('No risk settings found for user/bot', { 
          userId: event.data.userId,
          botId: event.data.botId 
        });
        return event;
      }
      
      // Check position size limits
      const positionValue = event.data.amount * event.data.price;
      if (positionValue > riskSettings.maxPositionSize) {
        logger.warn('Order exceeds max position size', {
          positionValue,
          maxPositionSize: riskSettings.maxPositionSize,
          eventId: event.id
        });
        
        // Adjust order size
        event.data.amount = riskSettings.maxPositionSize / event.data.price;
        event.data.riskAdjusted = true;
      }
      
      // Check daily loss limits
      const todayLoss = await this.calculateDailyLoss(event.data.userId);
      if (todayLoss > riskSettings.maxDailyLoss) {
        logger.warn('Daily loss limit exceeded, blocking order', {
          todayLoss,
          maxDailyLoss: riskSettings.maxDailyLoss,
          eventId: event.id
        });
        return null; // Block the order
      }
      
      // Check maximum positions
      const openPositions = await prisma.position.count({
        where: {
          userId: event.data.userId,
          status: 'Open'
        }
      });
      
      if (openPositions >= riskSettings.maxPositions) {
        logger.warn('Maximum positions limit reached', {
          openPositions,
          maxPositions: riskSettings.maxPositions,
          eventId: event.id
        });
        return null; // Block the order
      }
      
      return event;
    } catch (error) {
      logger.error('Error processing order risk', { error, eventId: event.id });
      return event; // Allow order to proceed on error
    }
  }
  
  private async processSignalRisk(event: TradingSignalEvent): Promise<TradingEvent | null> {
    try {
      // Check signal confidence threshold
      if (event.data.confidenceScore < 60) {
        logger.debug('Signal confidence below threshold', {
          confidence: event.data.confidenceScore,
          eventId: event.id
        });
        return null; // Filter out low confidence signals
      }
      
      // Check if signal is for a volatile market
      const volatility = await this.getMarketVolatility(event.data.symbol);
      if (volatility > 0.05) { // 5% volatility threshold
        logger.warn('High market volatility detected', {
          symbol: event.data.symbol,
          volatility,
          eventId: event.id
        });
        
        // Reduce signal strength in volatile markets
        event.data.strength = 'WEAK';
        event.data.riskAdjusted = true;
      }
      
      return event;
    } catch (error) {
      logger.error('Error processing signal risk', { error, eventId: event.id });
      return event;
    }
  }
  
  private async processMarketRisk(event: MarketDataEvent): Promise<TradingEvent | null> {
    try {
      // Check for circuit breaker conditions
      const priceChange = this.calculatePriceChange(event.data);
      if (Math.abs(priceChange) > 0.1) { // 10% price change
        logger.warn('Large price movement detected', {
          symbol: event.data.symbol,
          priceChange,
          eventId: event.id
        });
        
        // Trigger circuit breaker event
        await this.triggerCircuitBreaker(event.data.symbol, priceChange);
      }
      
      return event;
    } catch (error) {
      logger.error('Error processing market risk', { error, eventId: event.id });
      return event;
    }
  }
  
  private async calculateDailyLoss(userId: string): Promise<number> {
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    
    const result = await prisma.order.aggregate({
      where: {
        userId,
        executedAt: {
          gte: today
        },
        status: 'filled'
      },
      _sum: {
        pnl: true
      }
    });
    
    return Math.abs(result._sum.pnl || 0);
  }
  
  private async getMarketVolatility(symbol: string): Promise<number> {
    // Simplified volatility calculation
    // In production, this would use historical price data
    return Math.random() * 0.1; // Mock volatility between 0-10%
  }
  
  private calculatePriceChange(marketData: any): number {
    if (!marketData.previousPrice || !marketData.currentPrice) {
      return 0;
    }
    return (marketData.currentPrice - marketData.previousPrice) / marketData.previousPrice;
  }
  
  private async triggerCircuitBreaker(symbol: string, priceChange: number) {
    try {
      await prisma.circuitBreaker.updateMany({
        where: {
          type: 'VOLATILITY',
          isActive: true
        },
        data: {
          status: 'TRIGGERED',
          lastTriggered: new Date(),
          metadata: {
            symbol,
            priceChange,
            triggeredAt: new Date().toISOString()
          }
        }
      });
      
      logger.warn('Circuit breaker triggered', { symbol, priceChange });
    } catch (error) {
      logger.error('Error triggering circuit breaker', { error, symbol });
    }
  }
}

class PortfolioManagementProcessor {
  async process(event: TradingEvent): Promise<TradingEvent | null> {
    logger.debug('Processing event for portfolio management', { eventId: event.id, type: event.type });
    
    try {
      switch (event.type) {
        case 'order':
          await this.updatePortfolioOnOrder(event as OrderEvent);
          break;
        case 'market_data':
          await this.updatePortfolioValuation(event as MarketDataEvent);
          break;
        default:
          break;
      }
      
      return event;
    } catch (error) {
      logger.error('Error in portfolio management processing', { error, eventId: event.id });
      return event;
    }
  }
  
  private async updatePortfolioOnOrder(event: OrderEvent) {
    try {
      if (event.data.status !== 'filled') {
        return;
      }
      
      // Update position if it exists
      const existingPosition = await prisma.position.findFirst({
        where: {
          userId: event.data.userId,
          symbol: event.data.symbol,
          status: 'Open'
        }
      });
      
      if (existingPosition) {
        // Update existing position
        const newAmount = event.data.side === 'buy' 
          ? existingPosition.amount + event.data.amount
          : existingPosition.amount - event.data.amount;
        
        if (newAmount <= 0) {
          // Close position
          await prisma.position.update({
            where: { id: existingPosition.id },
            data: {
              status: 'Closed',
              closedAt: new Date(),
              pnl: this.calculatePnL(existingPosition, event.data.price)
            }
          });
        } else {
          // Update position
          const newAvgPrice = this.calculateAveragePrice(
            existingPosition.entryPrice,
            existingPosition.amount,
            event.data.price,
            event.data.amount,
            event.data.side
          );
          
          await prisma.position.update({
            where: { id: existingPosition.id },
            data: {
              amount: newAmount,
              entryPrice: newAvgPrice,
              currentPrice: event.data.price
            }
          });
        }
      }
      
      logger.debug('Portfolio updated on order fill', {
        userId: event.data.userId,
        symbol: event.data.symbol,
        eventId: event.id
      });
    } catch (error) {
      logger.error('Error updating portfolio on order', { error, eventId: event.id });
    }
  }
  
  private async updatePortfolioValuation(event: MarketDataEvent) {
    try {
      // Update current prices for all open positions
      await prisma.position.updateMany({
        where: {
          symbol: event.data.symbol,
          status: 'Open'
        },
        data: {
          currentPrice: event.data.price
        }
      });
      
      logger.debug('Portfolio valuations updated', {
        symbol: event.data.symbol,
        price: event.data.price,
        eventId: event.id
      });
    } catch (error) {
      logger.error('Error updating portfolio valuation', { error, eventId: event.id });
    }
  }
  
  private calculatePnL(position: any, currentPrice: number): number {
    const priceDiff = currentPrice - position.entryPrice;
    const multiplier = position.side === 'Long' ? 1 : -1;
    return priceDiff * position.amount * multiplier;
  }
  
  private calculateAveragePrice(
    oldPrice: number,
    oldAmount: number,
    newPrice: number,
    newAmount: number,
    side: string
  ): number {
    if (side === 'buy') {
      return (oldPrice * oldAmount + newPrice * newAmount) / (oldAmount + newAmount);
    } else {
      // For sell orders, we don't change the average price
      return oldPrice;
    }
  }
}

class SystemMonitoringProcessor {
  private metrics = {
    eventsProcessed: 0,
    errorsCount: 0,
    lastEventTime: Date.now(),
    processingTimes: [] as number[]
  };
  
  async process(event: TradingEvent): Promise<TradingEvent | null> {
    const startTime = Date.now();
    
    try {
      // Update metrics
      this.metrics.eventsProcessed++;
      this.metrics.lastEventTime = Date.now();
      
      // Monitor system health
      await this.monitorSystemHealth(event);
      
      // Log performance metrics
      await this.logPerformanceMetrics(event);
      
      // Check for anomalies
      await this.detectAnomalies(event);
      
      const processingTime = Date.now() - startTime;
      this.metrics.processingTimes.push(processingTime);
      
      // Keep only last 1000 processing times
      if (this.metrics.processingTimes.length > 1000) {
        this.metrics.processingTimes.shift();
      }
      
      logger.debug('System monitoring completed', {
        eventId: event.id,
        processingTime,
        eventsProcessed: this.metrics.eventsProcessed
      });
      
      return event;
    } catch (error) {
      this.metrics.errorsCount++;
      logger.error('Error in system monitoring', { error, eventId: event.id });
      return event;
    }
  }
  
  private async monitorSystemHealth(event: TradingEvent) {
    try {
      // Check processing latency
      const eventAge = Date.now() - event.timestamp;
      if (eventAge > 5000) { // 5 seconds
        logger.warn('High event processing latency detected', {
          eventId: event.id,
          latency: eventAge
        });
      }
      
      // Check error rate
      const errorRate = this.metrics.errorsCount / this.metrics.eventsProcessed;
      if (errorRate > 0.05) { // 5% error rate
        logger.warn('High error rate detected', {
          errorRate,
          errorsCount: this.metrics.errorsCount,
          eventsProcessed: this.metrics.eventsProcessed
        });
      }
      
      // Record system metrics
      await prisma.performanceMetric.create({
        data: {
          system: 'TRADING',
          component: 'EventProcessor',
          metric: 'processing_latency',
          value: eventAge,
          unit: 'milliseconds',
          tags: {
            eventType: event.type,
            eventId: event.id
          }
        }
      });
    } catch (error) {
      logger.error('Error monitoring system health', { error, eventId: event.id });
    }
  }
  
  private async logPerformanceMetrics(event: TradingEvent) {
    try {
      // Calculate average processing time
      const avgProcessingTime = this.metrics.processingTimes.length > 0
        ? this.metrics.processingTimes.reduce((a, b) => a + b, 0) / this.metrics.processingTimes.length
        : 0;
      
      // Log metrics every 100 events
      if (this.metrics.eventsProcessed % 100 === 0) {
        logger.info('System performance metrics', {
          eventsProcessed: this.metrics.eventsProcessed,
          errorsCount: this.metrics.errorsCount,
          errorRate: this.metrics.errorsCount / this.metrics.eventsProcessed,
          avgProcessingTime,
          lastEventTime: this.metrics.lastEventTime
        });
      }
    } catch (error) {
      logger.error('Error logging performance metrics', { error, eventId: event.id });
    }
  }
  
  private async detectAnomalies(event: TradingEvent) {
    try {
      // Detect unusual event patterns
      if (event.type === 'market_data') {
        const marketEvent = event as MarketDataEvent;
        
        // Check for unusual price movements
        if (marketEvent.data.volume && marketEvent.data.volume > 1000000) {
          logger.warn('Unusual high volume detected', {
            symbol: marketEvent.data.symbol,
            volume: marketEvent.data.volume,
            eventId: event.id
          });
        }
        
        // Check for price gaps
        if (marketEvent.data.previousPrice && marketEvent.data.price) {
          const priceChange = Math.abs(
            (marketEvent.data.price - marketEvent.data.previousPrice) / marketEvent.data.previousPrice
          );
          
          if (priceChange > 0.05) { // 5% price gap
            logger.warn('Large price gap detected', {
              symbol: marketEvent.data.symbol,
              priceChange,
              eventId: event.id
            });
          }
        }
      }
    } catch (error) {
      logger.error('Error detecting anomalies', { error, eventId: event.id });
    }
  }
  
  public getMetrics() {
    return {
      ...this.metrics,
      avgProcessingTime: this.metrics.processingTimes.length > 0
        ? this.metrics.processingTimes.reduce((a, b) => a + b, 0) / this.metrics.processingTimes.length
        : 0,
      errorRate: this.metrics.errorsCount / this.metrics.eventsProcessed
    };
  }
}

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
      logger.warn('‚ö†Ô∏è Event-driven trading system is already running');
      return;
    }

    try {
      logger.info('üöÄ Starting event-driven trading system...');
      
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
      logger.info('‚úÖ Event-driven trading system started successfully');

    } catch (error) {
      logger.error('‚ùå Failed to start event-driven trading system:', error);
      this.isRunning = false;
      throw error;
    }
  }

  /**
   * Stop the event-driven trading system
   */
  public async stop(): Promise<void> {
    if (!this.isRunning) {
      logger.warn('‚ö†Ô∏è Event-driven trading system is not running');
      return;
    }

    try {
      logger.info('üõë Stopping event-driven trading system...');

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
      
      logger.info('‚úÖ Event-driven trading system stopped successfully');

    } catch (error) {
      logger.error('‚ùå Error stopping event-driven trading system:', error);
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
        logger.info('üìä Registered market data processor');
      }

      // Signal Processor
      if (this.config.enableSignalProcessing) {
        const signalProcessor = new SignalProcessor();
        eventProcessingPipeline.registerProcessor(signalProcessor);
        this.processors.set('signal', signalProcessor);
        logger.info('üéØ Registered signal processor');
      }

      // Register Order Processor
      if (!this.processors.has('order')) {
        const orderProcessor = new OrderProcessor();
        eventProcessingPipeline.registerProcessor(orderProcessor);
        this.processors.set('order', orderProcessor);
        logger.info('üéØ Registered order processor');
      }
      
      // Register Risk Management Processor
      if (!this.processors.has('risk')) {
        const riskProcessor = new RiskManagementProcessor();
        eventProcessingPipeline.registerProcessor(riskProcessor);
        this.processors.set('risk', riskProcessor);
        logger.info('üéØ Registered risk management processor');
      }
      
      // Register Portfolio Management Processor
      if (!this.processors.has('portfolio')) {
        const portfolioProcessor = new PortfolioManagementProcessor();
        eventProcessingPipeline.registerProcessor(portfolioProcessor);
        this.processors.set('portfolio', portfolioProcessor);
        logger.info('üéØ Registered portfolio management processor');
      }
      
      // Register System Monitoring Processor
      if (!this.processors.has('monitoring')) {
        const monitoringProcessor = new SystemMonitoringProcessor();
        eventProcessingPipeline.registerProcessor(monitoringProcessor);
        this.processors.set('monitoring', monitoringProcessor);
        logger.info('üéØ Registered system monitoring processor');
      }

      this.stats.activeProcessors = this.processors.size;

    } catch (error) {
      logger.error('‚ùå Failed to register processors:', error);
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
      logger.error(`‚ùå Event processing failed:`, data);
    });

    eventProcessingPipeline.on('circuitBreakerOpened', () => {
      this.stats.systemHealth = 'DEGRADED';
      this.emit('circuitBreakerOpened');
      logger.warn('üö® Circuit breaker opened - system degraded');
    });

    eventProcessingPipeline.on('circuitBreakerClosed', () => {
      this.stats.systemHealth = 'HEALTHY';
      this.emit('circuitBreakerClosed');
      logger.info('‚úÖ Circuit breaker closed - system healthy');
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
        logger.error('‚ùå System monitoring error:', error);
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
      logger.error('‚ùå Health check failed:', error);
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
    logger.info('‚öôÔ∏è System configuration updated');
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
