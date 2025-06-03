/**
 * Event Processing Pipeline
 * Core event-driven processing pipeline for trading system
 */

import { EventEmitter } from 'events';
import { redisStreamsService } from './redisStreamsService';
import { logger } from '../utils/logger';
import {
  TradingEvent,
  StreamName,
  ConsumerGroup,
  EventProcessingResult,
  ProcessingStatus,
  MarketDataEvent,
  TradingSignalEvent,
  OrderEvent,
  RiskEvent,
  SystemEvent,
  BotEvent,
  STREAM_NAMES,
  CONSUMER_GROUPS,
  createCorrelationId,
} from '../types/events';

export interface EventProcessor {
  process(event: TradingEvent): Promise<EventProcessingResult>;
  canProcess(event: TradingEvent): boolean;
  getName(): string;
}

export interface PipelineConfig {
  consumerName: string;
  batchSize: number;
  blockTime: number;
  retryAttempts: number;
  retryDelay: number;
  deadLetterThreshold: number;
  processingTimeout: number;
  enableCircuitBreaker: boolean;
  circuitBreakerThreshold: number;
  circuitBreakerTimeout: number;
}

export interface PipelineStats {
  processedEvents: number;
  failedEvents: number;
  retryEvents: number;
  deadLetterEvents: number;
  averageProcessingTime: number;
  throughput: number;
  circuitBreakerOpen: boolean;
  lastProcessedAt: number;
  uptime: number;
}

export class EventProcessingPipeline extends EventEmitter {
  private processors: Map<string, EventProcessor> = new Map();
  private isRunning: boolean = false;
  private config: PipelineConfig;
  private stats: PipelineStats;
  private startTime: number = 0;
  private processingTimes: number[] = [];
  private circuitBreakerOpen: boolean = false;
  private circuitBreakerOpenTime: number = 0;
  private consecutiveFailures: number = 0;

  constructor(config: Partial<PipelineConfig> = {}) {
    super();
    
    this.config = {
      consumerName: `pipeline-${process.pid}`,
      batchSize: 10,
      blockTime: 1000,
      retryAttempts: 3,
      retryDelay: 1000,
      deadLetterThreshold: 5,
      processingTimeout: 30000,
      enableCircuitBreaker: true,
      circuitBreakerThreshold: 10,
      circuitBreakerTimeout: 60000,
      ...config,
    };

    this.stats = {
      processedEvents: 0,
      failedEvents: 0,
      retryEvents: 0,
      deadLetterEvents: 0,
      averageProcessingTime: 0,
      throughput: 0,
      circuitBreakerOpen: false,
      lastProcessedAt: 0,
      uptime: 0,
    };
  }

  /**
   * Register an event processor
   */
  public registerProcessor(processor: EventProcessor): void {
    this.processors.set(processor.getName(), processor);
    logger.info(`📝 Registered event processor: ${processor.getName()}`);
  }

  /**
   * Start the event processing pipeline
   */
  public async start(): Promise<void> {
    if (this.isRunning) {
      logger.warn('⚠️ Event processing pipeline is already running');
      return;
    }

    try {
      await redisStreamsService.initialize();
      
      this.isRunning = true;
      this.startTime = Date.now();
      
      logger.info('🚀 Starting event processing pipeline...');
      
      // Start processing loops for each stream
      const processingPromises = [
        this.processStream(STREAM_NAMES.MARKET_DATA, CONSUMER_GROUPS.SIGNAL_PROCESSOR),
        this.processStream(STREAM_NAMES.TRADING_SIGNALS, CONSUMER_GROUPS.ORDER_EXECUTOR),
        this.processStream(STREAM_NAMES.ORDERS, CONSUMER_GROUPS.PORTFOLIO_MANAGER),
        this.processStream(STREAM_NAMES.RISK_MANAGEMENT, CONSUMER_GROUPS.RISK_MANAGER),
        this.processStream(STREAM_NAMES.SYSTEM, CONSUMER_GROUPS.MONITORING),
        this.processStream(STREAM_NAMES.BOTS, CONSUMER_GROUPS.MONITORING),
      ];

      // Start stats update loop
      this.startStatsUpdate();

      this.emit('started');
      logger.info('✅ Event processing pipeline started successfully');

      // Wait for all processing loops (they run indefinitely)
      await Promise.all(processingPromises);

    } catch (error) {
      logger.error('❌ Failed to start event processing pipeline:', error);
      this.isRunning = false;
      throw error;
    }
  }

  /**
   * Stop the event processing pipeline
   */
  public async stop(): Promise<void> {
    if (!this.isRunning) {
      logger.warn('⚠️ Event processing pipeline is not running');
      return;
    }

    logger.info('🛑 Stopping event processing pipeline...');
    
    this.isRunning = false;
    
    try {
      await redisStreamsService.shutdown();
      this.emit('stopped');
      logger.info('✅ Event processing pipeline stopped successfully');
    } catch (error) {
      logger.error('❌ Error stopping event processing pipeline:', error);
      throw error;
    }
  }

  /**
   * Process events from a specific stream
   */
  private async processStream(streamName: StreamName, groupName: ConsumerGroup): Promise<void> {
    logger.info(`🔄 Starting event processing for stream: ${streamName} (group: ${groupName})`);

    while (this.isRunning) {
      try {
        // Check circuit breaker
        if (this.isCircuitBreakerOpen()) {
          await this.sleep(this.config.circuitBreakerTimeout);
          continue;
        }

        // Read events from stream
        const events = await redisStreamsService.readEvents(streamName, {
          groupName,
          consumerName: this.config.consumerName,
          blockTime: this.config.blockTime,
          count: this.config.batchSize,
        });

        if (events.length === 0) {
          continue;
        }

        logger.debug(`📥 Received ${events.length} events from ${streamName}`);

        // Process events in parallel
        const processingPromises = events.map(event => this.processEvent(event, streamName, groupName));
        const results = await Promise.allSettled(processingPromises);

        // Handle results and acknowledgments
        const messageIds: string[] = [];
        let successCount = 0;
        let failureCount = 0;

        for (let i = 0; i < results.length; i++) {
          const result = results[i];
          const event = events[i];

          if (result.status === 'fulfilled' && result.value.status === ProcessingStatus.COMPLETED) {
            messageIds.push(event.id!);
            successCount++;
            this.consecutiveFailures = 0;
          } else {
            failureCount++;
            this.consecutiveFailures++;
            
            // Handle failed event
            await this.handleFailedEvent(event, streamName, groupName, 
              result.status === 'rejected' ? result.reason : 
              result.status === 'fulfilled' ? result.value.error : 'Unknown error'
            );
          }
        }

        // Acknowledge successfully processed events
        if (messageIds.length > 0) {
          await redisStreamsService.acknowledgeEvents(streamName, groupName, messageIds);
        }

        // Update stats
        this.stats.processedEvents += successCount;
        this.stats.failedEvents += failureCount;
        this.stats.lastProcessedAt = Date.now();

        // Check circuit breaker
        if (this.config.enableCircuitBreaker && 
            this.consecutiveFailures >= this.config.circuitBreakerThreshold) {
          this.openCircuitBreaker();
        }

        logger.debug(`✅ Processed ${successCount}/${events.length} events from ${streamName}`);

      } catch (error) {
        logger.error(`❌ Error processing stream ${streamName}:`, error);
        this.consecutiveFailures++;
        
        // Back off on error
        await this.sleep(this.config.retryDelay);
      }
    }

    logger.info(`🛑 Stopped processing stream: ${streamName}`);
  }

  /**
   * Process a single event
   */
  private async processEvent(
    event: TradingEvent,
    streamName: StreamName,
    groupName: ConsumerGroup
  ): Promise<EventProcessingResult> {
    const startTime = Date.now();
    
    try {
      // Find appropriate processor
      const processor = this.findProcessor(event);
      if (!processor) {
        throw new Error(`No processor found for event type: ${event.type}`);
      }

      // Process event with timeout
      const result = await Promise.race([
        processor.process(event),
        this.createTimeoutPromise(this.config.processingTimeout),
      ]);

      const processingTime = Date.now() - startTime;
      this.updateProcessingTime(processingTime);

      // Emit processing result
      this.emit('eventProcessed', {
        event,
        result,
        processingTime,
        processor: processor.getName(),
      });

      return {
        ...result,
        processingTime,
      };

    } catch (error) {
      const processingTime = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      logger.error(`❌ Failed to process event ${event.id}:`, error);

      this.emit('eventFailed', {
        event,
        error: errorMessage,
        processingTime,
      });

      return {
        eventId: event.id!,
        status: ProcessingStatus.FAILED,
        processingTime,
        error: errorMessage,
      };
    }
  }

  /**
   * Find appropriate processor for an event
   */
  private findProcessor(event: TradingEvent): EventProcessor | null {
    for (const processor of this.processors.values()) {
      if (processor.canProcess(event)) {
        return processor;
      }
    }
    return null;
  }

  /**
   * Handle failed event processing
   */
  private async handleFailedEvent(
    event: TradingEvent,
    streamName: StreamName,
    groupName: ConsumerGroup,
    error: string
  ): Promise<void> {
    try {
      // Get retry count from metadata
      const retryCount = (event.metadata?.retryCount || 0) + 1;

      if (retryCount <= this.config.retryAttempts) {
        // Retry the event
        event.metadata = {
          ...event.metadata,
          retryCount,
          errorMessage: error,
          lastRetryAt: Date.now(),
        };

        // Re-publish to stream for retry
        await redisStreamsService.publishEvent(streamName, event);
        this.stats.retryEvents++;

        logger.warn(`🔄 Retrying event ${event.id} (attempt ${retryCount}/${this.config.retryAttempts})`);
      } else {
        // Send to dead letter queue
        await this.sendToDeadLetterQueue(event, error);
        this.stats.deadLetterEvents++;

        logger.error(`💀 Sent event ${event.id} to dead letter queue after ${retryCount} attempts`);
      }
    } catch (retryError) {
      logger.error(`❌ Failed to handle failed event ${event.id}:`, retryError);
    }
  }

  /**
   * Send event to dead letter queue
   */
  private async sendToDeadLetterQueue(event: TradingEvent, error: string): Promise<void> {
    const deadLetterEvent: SystemEvent = {
      id: `dl-${event.id}`,
      type: 'SYSTEM_ERROR',
      timestamp: Date.now(),
      version: '1.0',
      source: 'event-processing-pipeline',
      correlationId: event.correlationId,
      causationId: event.id,
      data: {
        component: 'event-processing-pipeline',
        status: 'UNHEALTHY',
        message: `Dead letter: ${error}`,
        errorDetails: {
          originalEvent: event,
          error,
          deadLetterReason: 'Max retry attempts exceeded',
        },
      },
    };

    await redisStreamsService.publishEvent(STREAM_NAMES.SYSTEM, deadLetterEvent);
  }

  /**
   * Circuit breaker methods
   */
  private isCircuitBreakerOpen(): boolean {
    if (!this.circuitBreakerOpen) {
      return false;
    }

    // Check if circuit breaker timeout has passed
    if (Date.now() - this.circuitBreakerOpenTime >= this.config.circuitBreakerTimeout) {
      this.closeCircuitBreaker();
      return false;
    }

    return true;
  }

  private openCircuitBreaker(): void {
    this.circuitBreakerOpen = true;
    this.circuitBreakerOpenTime = Date.now();
    this.stats.circuitBreakerOpen = true;
    
    logger.warn('🚨 Circuit breaker opened due to consecutive failures');
    this.emit('circuitBreakerOpened');
  }

  private closeCircuitBreaker(): void {
    this.circuitBreakerOpen = false;
    this.circuitBreakerOpenTime = 0;
    this.consecutiveFailures = 0;
    this.stats.circuitBreakerOpen = false;
    
    logger.info('✅ Circuit breaker closed');
    this.emit('circuitBreakerClosed');
  }

  /**
   * Update processing time statistics
   */
  private updateProcessingTime(processingTime: number): void {
    this.processingTimes.push(processingTime);
    
    // Keep only last 1000 processing times
    if (this.processingTimes.length > 1000) {
      this.processingTimes = this.processingTimes.slice(-1000);
    }

    // Calculate average
    this.stats.averageProcessingTime = 
      this.processingTimes.reduce((sum, time) => sum + time, 0) / this.processingTimes.length;
  }

  /**
   * Start stats update loop
   */
  private startStatsUpdate(): void {
    const updateInterval = setInterval(() => {
      if (!this.isRunning) {
        clearInterval(updateInterval);
        return;
      }

      // Calculate throughput (events per second)
      const uptime = (Date.now() - this.startTime) / 1000;
      this.stats.uptime = uptime;
      this.stats.throughput = this.stats.processedEvents / uptime;

      this.emit('statsUpdated', this.stats);
    }, 5000); // Update every 5 seconds
  }

  /**
   * Utility methods
   */
  private async sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  private createTimeoutPromise(timeout: number): Promise<never> {
    return new Promise((_, reject) => {
      setTimeout(() => reject(new Error('Processing timeout')), timeout);
    });
  }

  /**
   * Get pipeline statistics
   */
  public getStats(): PipelineStats {
    return { ...this.stats };
  }

  /**
   * Get pipeline configuration
   */
  public getConfig(): PipelineConfig {
    return { ...this.config };
  }

  /**
   * Health check
   */
  public async healthCheck(): Promise<boolean> {
    try {
      return this.isRunning && await redisStreamsService.healthCheck();
    } catch (error) {
      return false;
    }
  }
}

// Export singleton instance
export const eventProcessingPipeline = new EventProcessingPipeline();
