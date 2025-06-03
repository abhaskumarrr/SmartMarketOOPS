/**
 * Trading Signal Event Processor
 * Processes trading signals and triggers order execution
 */

import { EventProcessor } from '../services/eventProcessingPipeline';
import { redisStreamsService } from '../services/redisStreamsService';
import { logger } from '../utils/logger';
import {
  TradingEvent,
  TradingSignalEvent,
  OrderEvent,
  RiskEvent,
  EventProcessingResult,
  ProcessingStatus,
  STREAM_NAMES,
  createEventId,
  createCorrelationId,
  isTradingSignalEvent,
} from '../types/events';

export interface SignalValidationRule {
  name: string;
  validate: (signal: TradingSignalEvent) => Promise<boolean>;
  reason?: string;
}

export class SignalProcessor implements EventProcessor {
  private name = 'SignalProcessor';
  private validationRules: SignalValidationRule[] = [];
  private processedSignals: Map<string, TradingSignalEvent> = new Map();
  private signalStats = {
    processed: 0,
    validated: 0,
    rejected: 0,
    expired: 0,
  };

  constructor() {
    this.initializeValidationRules();
  }

  public getName(): string {
    return this.name;
  }

  public canProcess(event: TradingEvent): boolean {
    return isTradingSignalEvent(event);
  }

  public async process(event: TradingEvent): Promise<EventProcessingResult> {
    const startTime = Date.now();
    const signalEvent = event as TradingSignalEvent;
    
    try {
      switch (signalEvent.type) {
        case 'SIGNAL_GENERATED':
          return await this.processGeneratedSignal(signalEvent);
        case 'SIGNAL_VALIDATED':
          return await this.processValidatedSignal(signalEvent);
        case 'SIGNAL_EXECUTED':
          return await this.processExecutedSignal(signalEvent);
        case 'SIGNAL_EXPIRED':
          return await this.processExpiredSignal(signalEvent);
        case 'SIGNAL_ERROR':
          return await this.processErrorSignal(signalEvent);
        default:
          throw new Error(`Unsupported signal event type: ${signalEvent.type}`);
      }
    } catch (error) {
      logger.error(`❌ Signal processing failed for event ${event.id}:`, error);
      
      return {
        eventId: event.id!,
        status: ProcessingStatus.FAILED,
        processingTime: Date.now() - startTime,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  /**
   * Process newly generated signal
   */
  private async processGeneratedSignal(event: TradingSignalEvent): Promise<EventProcessingResult> {
    const startTime = Date.now();
    const nextEvents: TradingEvent[] = [];

    try {
      const { signalId, symbol, price, confidenceScore } = event.data;

      // Check if signal has expired
      if (event.data.expiresAt && Date.now() > event.data.expiresAt) {
        logger.warn(`⏰ Signal ${signalId} has expired`);
        
        const expiredEvent: TradingSignalEvent = {
          ...event,
          type: 'SIGNAL_EXPIRED',
          timestamp: Date.now(),
        };
        
        nextEvents.push(expiredEvent);
        this.signalStats.expired++;
        
        return {
          eventId: event.id!,
          status: ProcessingStatus.COMPLETED,
          processingTime: Date.now() - startTime,
          result: { status: 'expired', reason: 'Signal expired before processing' },
          nextEvents,
        };
      }

      // Validate signal
      const validationResult = await this.validateSignal(event);
      
      if (validationResult.isValid) {
        // Signal is valid, proceed to risk check
        const validatedEvent: TradingSignalEvent = {
          ...event,
          type: 'SIGNAL_VALIDATED',
          timestamp: Date.now(),
          data: {
            ...event.data,
            validatedAt: Date.now(),
          },
        };

        // Generate risk check event
        const riskEvent: RiskEvent = {
          id: createEventId(),
          type: 'RISK_CHECK_PASSED', // Will be updated by risk processor
          timestamp: Date.now(),
          version: '1.0',
          source: 'signal-processor',
          correlationId: event.correlationId,
          causationId: event.id,
          userId: event.userId,
          data: {
            riskCheckId: createEventId(),
            symbol,
            riskType: 'POSITION_SIZE',
            currentValue: 0, // Will be calculated by risk processor
            threshold: 0, // Will be set by risk processor
            severity: 'LOW',
            action: 'ALLOW',
            reason: 'Signal validation passed',
          },
        };

        nextEvents.push(validatedEvent, riskEvent);
        this.signalStats.validated++;

        logger.info(`✅ Signal ${signalId} validated for ${symbol} @ ${price}`);
      } else {
        // Signal validation failed
        logger.warn(`❌ Signal ${signalId} validation failed: ${validationResult.reason}`);
        
        const errorEvent: TradingSignalEvent = {
          ...event,
          type: 'SIGNAL_ERROR',
          timestamp: Date.now(),
          metadata: {
            ...event.metadata,
            errorMessage: validationResult.reason,
            errorCode: 'VALIDATION_FAILED',
          },
        };

        nextEvents.push(errorEvent);
        this.signalStats.rejected++;
      }

      // Store processed signal
      this.processedSignals.set(signalId, event);
      this.signalStats.processed++;

      // Publish next events
      for (const nextEvent of nextEvents) {
        if (nextEvent.type.startsWith('SIGNAL_')) {
          await redisStreamsService.publishEvent(STREAM_NAMES.TRADING_SIGNALS, nextEvent);
        } else if (nextEvent.type.startsWith('RISK_')) {
          await redisStreamsService.publishEvent(STREAM_NAMES.RISK_MANAGEMENT, nextEvent);
        }
      }

      return {
        eventId: event.id!,
        status: ProcessingStatus.COMPLETED,
        processingTime: Date.now() - startTime,
        result: {
          signalId,
          symbol,
          validated: validationResult.isValid,
          reason: validationResult.reason,
        },
        nextEvents,
      };

    } catch (error) {
      throw new Error(`Generated signal processing failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Process validated signal (ready for execution)
   */
  private async processValidatedSignal(event: TradingSignalEvent): Promise<EventProcessingResult> {
    const startTime = Date.now();
    const nextEvents: TradingEvent[] = [];

    try {
      const { signalId, symbol, signalType, direction, price, targetPrice, stopLoss } = event.data;

      // Generate order event
      const orderEvent: OrderEvent = {
        id: createEventId(),
        type: 'ORDER_CREATED',
        timestamp: Date.now(),
        version: '1.0',
        source: 'signal-processor',
        correlationId: event.correlationId,
        causationId: event.id,
        userId: event.userId,
        data: {
          orderId: createEventId(),
          signalId,
          symbol,
          side: direction === 'LONG' ? 'BUY' : 'SELL',
          type: 'MARKET', // Default to market order for signals
          status: 'PENDING',
          exchange: 'default', // Should be determined by configuration
          quantity: 0, // Will be calculated by order processor based on risk management
          price: signalType === 'ENTRY' ? undefined : price, // Market order for entry
          stopPrice: stopLoss,
          filledQuantity: 0,
          remainingQuantity: 0,
        },
      };

      nextEvents.push(orderEvent);

      // Publish order event
      await redisStreamsService.publishEvent(STREAM_NAMES.ORDERS, orderEvent);

      logger.info(`📋 Created order for validated signal ${signalId}: ${direction} ${symbol}`);

      return {
        eventId: event.id!,
        status: ProcessingStatus.COMPLETED,
        processingTime: Date.now() - startTime,
        result: {
          signalId,
          symbol,
          orderCreated: true,
          orderId: orderEvent.data.orderId,
        },
        nextEvents,
      };

    } catch (error) {
      throw new Error(`Validated signal processing failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Process executed signal
   */
  private async processExecutedSignal(event: TradingSignalEvent): Promise<EventProcessingResult> {
    const startTime = Date.now();

    try {
      const { signalId, symbol } = event.data;

      // Update signal status
      const storedSignal = this.processedSignals.get(signalId);
      if (storedSignal) {
        storedSignal.data.executedAt = Date.now();
        this.processedSignals.set(signalId, storedSignal);
      }

      logger.info(`✅ Signal ${signalId} executed for ${symbol}`);

      return {
        eventId: event.id!,
        status: ProcessingStatus.COMPLETED,
        processingTime: Date.now() - startTime,
        result: {
          signalId,
          symbol,
          executed: true,
        },
      };

    } catch (error) {
      throw new Error(`Executed signal processing failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Process expired signal
   */
  private async processExpiredSignal(event: TradingSignalEvent): Promise<EventProcessingResult> {
    const startTime = Date.now();

    try {
      const { signalId, symbol } = event.data;

      // Remove from processed signals
      this.processedSignals.delete(signalId);

      logger.info(`⏰ Signal ${signalId} expired for ${symbol}`);

      return {
        eventId: event.id!,
        status: ProcessingStatus.COMPLETED,
        processingTime: Date.now() - startTime,
        result: {
          signalId,
          symbol,
          expired: true,
        },
      };

    } catch (error) {
      throw new Error(`Expired signal processing failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Process error signal
   */
  private async processErrorSignal(event: TradingSignalEvent): Promise<EventProcessingResult> {
    const startTime = Date.now();

    try {
      const { signalId, symbol } = event.data;
      const errorMessage = event.metadata?.errorMessage || 'Unknown error';

      logger.error(`❌ Signal ${signalId} error for ${symbol}: ${errorMessage}`);

      return {
        eventId: event.id!,
        status: ProcessingStatus.COMPLETED,
        processingTime: Date.now() - startTime,
        result: {
          signalId,
          symbol,
          error: true,
          errorMessage,
        },
      };

    } catch (error) {
      throw new Error(`Error signal processing failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Initialize validation rules
   */
  private initializeValidationRules(): void {
    this.validationRules = [
      {
        name: 'confidence_threshold',
        validate: async (signal) => signal.data.confidenceScore >= 60,
        reason: 'Confidence score below threshold (60%)',
      },
      {
        name: 'price_validity',
        validate: async (signal) => signal.data.price > 0,
        reason: 'Invalid price value',
      },
      {
        name: 'risk_reward_ratio',
        validate: async (signal) => signal.data.riskRewardRatio >= 1.5,
        reason: 'Risk-reward ratio below minimum (1.5)',
      },
      {
        name: 'expiry_check',
        validate: async (signal) => !signal.data.expiresAt || Date.now() < signal.data.expiresAt,
        reason: 'Signal has expired',
      },
      {
        name: 'duplicate_check',
        validate: async (signal) => !this.processedSignals.has(signal.data.signalId),
        reason: 'Duplicate signal detected',
      },
    ];
  }

  /**
   * Validate signal against all rules
   */
  private async validateSignal(signal: TradingSignalEvent): Promise<{ isValid: boolean; reason?: string }> {
    for (const rule of this.validationRules) {
      try {
        const isValid = await rule.validate(signal);
        if (!isValid) {
          return { isValid: false, reason: rule.reason || `Validation rule '${rule.name}' failed` };
        }
      } catch (error) {
        logger.error(`❌ Validation rule '${rule.name}' error:`, error);
        return { isValid: false, reason: `Validation rule '${rule.name}' error: ${error instanceof Error ? error.message : 'Unknown error'}` };
      }
    }

    return { isValid: true };
  }

  /**
   * Add custom validation rule
   */
  public addValidationRule(rule: SignalValidationRule): void {
    this.validationRules.push(rule);
    logger.info(`📝 Added validation rule: ${rule.name}`);
  }

  /**
   * Remove validation rule
   */
  public removeValidationRule(name: string): boolean {
    const index = this.validationRules.findIndex(rule => rule.name === name);
    if (index !== -1) {
      this.validationRules.splice(index, 1);
      logger.info(`🗑️ Removed validation rule: ${name}`);
      return true;
    }
    return false;
  }

  /**
   * Get processor statistics
   */
  public getStats(): {
    name: string;
    signalStats: typeof this.signalStats;
    validationRules: number;
    processedSignalsCount: number;
  } {
    return {
      name: this.name,
      signalStats: { ...this.signalStats },
      validationRules: this.validationRules.length,
      processedSignalsCount: this.processedSignals.size,
    };
  }

  /**
   * Clear processed signals cache
   */
  public clearProcessedSignals(): void {
    this.processedSignals.clear();
    logger.info(`🗑️ Cleared processed signals cache`);
  }
}
