/**
 * Redis Streams Service
 * High-performance event streaming service for real-time trading events
 */

import Redis from 'ioredis';
import { redisConnection } from '../config/redis';
import { logger } from '../utils/logger';
import {
  TradingEvent,
  StreamName,
  ConsumerGroup,
  EventProcessingResult,
  ProcessingStatus,
  createEventId,
  STREAM_NAMES,
  CONSUMER_GROUPS,
} from '../types/events';

export interface StreamMessage {
  id: string;
  fields: Record<string, string>;
}

export interface ConsumerInfo {
  name: string;
  pending: number;
  idle: number;
}

export interface StreamInfo {
  length: number;
  radixTreeKeys: number;
  radixTreeNodes: number;
  groups: number;
  lastGeneratedId: string;
  firstEntry?: StreamMessage;
  lastEntry?: StreamMessage;
}

export interface ProducerOptions {
  maxLength?: number;
  approximateMaxLength?: boolean;
  trimStrategy?: 'MAXLEN' | 'MINID';
}

export interface ConsumerOptions {
  groupName: string;
  consumerName: string;
  blockTime?: number;
  count?: number;
  startId?: string;
}

export class RedisStreamsService {
  private static instance: RedisStreamsService;
  private redis: Redis | null = null;
  private isInitialized: boolean = false;

  private constructor() {}

  public static getInstance(): RedisStreamsService {
    if (!RedisStreamsService.instance) {
      RedisStreamsService.instance = new RedisStreamsService();
    }
    return RedisStreamsService.instance;
  }

  public async initialize(): Promise<void> {
    try {
      await redisConnection.connect();
      this.redis = redisConnection.getRedis();

      // Mark as initialized before creating consumer groups
      this.isInitialized = true;

      // Create consumer groups for all streams
      await this.createConsumerGroups();

      logger.info('✅ Redis Streams service initialized successfully');
    } catch (error) {
      this.isInitialized = false;
      logger.error('❌ Failed to initialize Redis Streams service:', error);
      throw error;
    }
  }

  public async shutdown(): Promise<void> {
    try {
      await redisConnection.disconnect();
      this.redis = null;
      this.isInitialized = false;
      logger.info('🔌 Redis Streams service shutdown completed');
    } catch (error) {
      logger.error('❌ Error during Redis Streams service shutdown:', error);
      throw error;
    }
  }

  private ensureRedis(): Redis {
    if (!this.redis || !this.isInitialized) {
      throw new Error('Redis Streams service not initialized. Call initialize() first.');
    }
    return this.redis;
  }

  // ============================================================================
  // PRODUCER METHODS
  // ============================================================================

  /**
   * Publish an event to a stream
   */
  public async publishEvent(
    streamName: StreamName,
    event: TradingEvent,
    options: ProducerOptions = {}
  ): Promise<string> {
    const redis = this.ensureRedis();
    
    try {
      // Ensure event has an ID
      if (!event.id) {
        event.id = createEventId();
      }

      // Serialize event data
      const fields = this.serializeEvent(event);
      
      // Build XADD command
      const args: (string | number)[] = [streamName];
      
      // Add trimming options
      if (options.maxLength) {
        if (options.approximateMaxLength) {
          args.push('MAXLEN', '~', options.maxLength);
        } else {
          args.push('MAXLEN', options.maxLength);
        }
      }
      
      // Add event ID (use * for auto-generation)
      args.push('*');
      
      // Add fields
      Object.entries(fields).forEach(([key, value]) => {
        args.push(key, value);
      });

      const messageId = await (redis as any).xadd(...args);
      
      logger.debug(`📤 Published event to ${streamName}:`, {
        eventId: event.id,
        messageId,
        type: event.type,
        source: event.source,
      });

      return messageId;
    } catch (error) {
      logger.error(`❌ Failed to publish event to ${streamName}:`, error);
      throw error;
    }
  }

  /**
   * Publish multiple events in a pipeline for better performance
   */
  public async publishEvents(
    streamName: StreamName,
    events: TradingEvent[],
    options: ProducerOptions = {}
  ): Promise<string[]> {
    const redis = this.ensureRedis();
    
    try {
      const pipeline = redis.pipeline();
      
      for (const event of events) {
        if (!event.id) {
          event.id = createEventId();
        }

        const fields = this.serializeEvent(event);
        const args: (string | number)[] = [streamName];
        
        if (options.maxLength) {
          if (options.approximateMaxLength) {
            args.push('MAXLEN', '~', options.maxLength);
          } else {
            args.push('MAXLEN', options.maxLength);
          }
        }
        
        args.push('*');
        Object.entries(fields).forEach(([key, value]) => {
          args.push(key, value);
        });

        (pipeline as any).xadd(...args);
      }

      const results = await pipeline.exec();
      const messageIds: string[] = [];

      if (results) {
        for (let i = 0; i < results.length; i++) {
          const [error, result] = results[i];
          if (error) {
            logger.error(`❌ Failed to publish event ${i} to ${streamName}:`, error);
            throw error;
          }
          messageIds.push(result as string);
        }
      }

      logger.debug(`📤 Published ${events.length} events to ${streamName}`);
      return messageIds;
    } catch (error) {
      logger.error(`❌ Failed to publish events to ${streamName}:`, error);
      throw error;
    }
  }

  // ============================================================================
  // CONSUMER METHODS
  // ============================================================================

  /**
   * Read events from a stream as a consumer group member
   */
  public async readEvents(
    streamName: StreamName,
    options: ConsumerOptions
  ): Promise<TradingEvent[]> {
    const redis = this.ensureRedis();

    try {
      const result = await redis.xreadgroup(
        'GROUP',
        options.groupName,
        options.consumerName,
        'COUNT',
        options.count || 10,
        'BLOCK',
        options.blockTime || 1000,
        'STREAMS',
        streamName,
        '>'
      ) as any[];

      if (!result || result.length === 0) {
        return [];
      }

      const events: TradingEvent[] = [];

      for (const [stream, messages] of result) {
        for (const [messageId, fields] of messages) {
          try {
            const event = this.deserializeEvent(messageId, fields as string[]);
            events.push(event);
          } catch (error) {
            logger.error(`❌ Failed to deserialize event ${messageId}:`, error);
          }
        }
      }

      logger.debug(`📥 Read ${events.length} events from ${streamName}`);
      return events;
    } catch (error) {
      logger.error(`❌ Failed to read events from ${streamName}:`, error);
      throw error;
    }
  }

  /**
   * Read pending events for a consumer
   */
  public async readPendingEvents(
    streamName: StreamName,
    groupName: string,
    consumerName: string,
    count: number = 10
  ): Promise<TradingEvent[]> {
    const redis = this.ensureRedis();

    try {
      const result = await redis.xreadgroup(
        'GROUP',
        groupName,
        consumerName,
        'COUNT',
        count,
        'STREAMS',
        streamName,
        '0'
      ) as any[];

      if (!result || result.length === 0) {
        return [];
      }

      const events: TradingEvent[] = [];

      for (const [stream, messages] of result) {
        for (const [messageId, fields] of messages) {
          try {
            const event = this.deserializeEvent(messageId, fields as string[]);
            events.push(event);
          } catch (error) {
            logger.error(`❌ Failed to deserialize pending event ${messageId}:`, error);
          }
        }
      }

      return events;
    } catch (error) {
      logger.error(`❌ Failed to read pending events from ${streamName}:`, error);
      throw error;
    }
  }

  /**
   * Acknowledge event processing
   */
  public async acknowledgeEvent(
    streamName: StreamName,
    groupName: string,
    messageId: string
  ): Promise<void> {
    const redis = this.ensureRedis();
    
    try {
      await redis.xack(streamName, groupName, messageId);
      logger.debug(`✅ Acknowledged event ${messageId} in ${streamName}`);
    } catch (error) {
      logger.error(`❌ Failed to acknowledge event ${messageId}:`, error);
      throw error;
    }
  }

  /**
   * Acknowledge multiple events
   */
  public async acknowledgeEvents(
    streamName: StreamName,
    groupName: string,
    messageIds: string[]
  ): Promise<void> {
    const redis = this.ensureRedis();
    
    try {
      if (messageIds.length > 0) {
        await redis.xack(streamName, groupName, ...messageIds);
        logger.debug(`✅ Acknowledged ${messageIds.length} events in ${streamName}`);
      }
    } catch (error) {
      logger.error(`❌ Failed to acknowledge events:`, error);
      throw error;
    }
  }

  // ============================================================================
  // STREAM MANAGEMENT
  // ============================================================================

  /**
   * Create consumer groups for all streams
   */
  private async createConsumerGroups(): Promise<void> {
    const redis = this.ensureRedis();
    
    const streams = Object.values(STREAM_NAMES);
    const groups = Object.values(CONSUMER_GROUPS);

    for (const streamName of streams) {
      for (const groupName of groups) {
        try {
          await redis.xgroup('CREATE', streamName, groupName, '$', 'MKSTREAM');
          logger.debug(`✅ Created consumer group ${groupName} for stream ${streamName}`);
        } catch (error: any) {
          if (error.message && error.message.includes('BUSYGROUP')) {
            // Group already exists, which is fine
            logger.debug(`ℹ️ Consumer group ${groupName} already exists for stream ${streamName}`);
          } else {
            logger.error(`❌ Failed to create consumer group ${groupName} for stream ${streamName}:`, error);
          }
        }
      }
    }
  }

  /**
   * Get stream information
   */
  public async getStreamInfo(streamName: StreamName): Promise<StreamInfo | null> {
    const redis = this.ensureRedis();
    
    try {
      const info = await redis.xinfo('STREAM', streamName);
      
      return {
        length: info[1] as number,
        radixTreeKeys: info[3] as number,
        radixTreeNodes: info[5] as number,
        groups: info[7] as number,
        lastGeneratedId: info[9] as string,
        firstEntry: info[11] ? {
          id: info[11][0],
          fields: this.arrayToObject(info[11][1]),
        } : undefined,
        lastEntry: info[13] ? {
          id: info[13][0],
          fields: this.arrayToObject(info[13][1]),
        } : undefined,
      };
    } catch (error) {
      logger.error(`❌ Failed to get stream info for ${streamName}:`, error);
      return null;
    }
  }

  /**
   * Get consumer group information
   */
  public async getConsumerGroupInfo(
    streamName: StreamName,
    groupName: string
  ): Promise<ConsumerInfo[]> {
    const redis = this.ensureRedis();
    
    try {
      const consumers = await redis.xinfo('CONSUMERS', streamName, groupName);
      
      return (consumers as any[]).map((consumer: any[]) => ({
        name: consumer[1] as string,
        pending: consumer[3] as number,
        idle: consumer[5] as number,
      }));
    } catch (error) {
      logger.error(`❌ Failed to get consumer group info for ${streamName}/${groupName}:`, error);
      return [];
    }
  }

  /**
   * Trim stream to maximum length
   */
  public async trimStream(
    streamName: StreamName,
    maxLength: number,
    approximate: boolean = true
  ): Promise<number> {
    const redis = this.ensureRedis();
    
    try {
      const args = [streamName, 'MAXLEN'];
      if (approximate) {
        args.push('~');
      }
      args.push(maxLength.toString());

      const trimmed = await (redis as any).xtrim(...args);
      logger.debug(`🗑️ Trimmed ${trimmed} messages from ${streamName}`);
      return trimmed;
    } catch (error) {
      logger.error(`❌ Failed to trim stream ${streamName}:`, error);
      throw error;
    }
  }

  // ============================================================================
  // UTILITY METHODS
  // ============================================================================

  /**
   * Serialize event to Redis fields
   */
  private serializeEvent(event: TradingEvent): Record<string, string> {
    return {
      id: event.id || createEventId(),
      type: event.type,
      timestamp: event.timestamp.toString(),
      version: event.version,
      source: event.source,
      correlationId: event.correlationId || '',
      causationId: event.causationId || '',
      userId: event.userId || '',
      data: JSON.stringify(event.data),
      metadata: JSON.stringify(event.metadata || {}),
    };
  }

  /**
   * Deserialize Redis fields to event
   */
  private deserializeEvent(messageId: string, fields: string[]): TradingEvent {
    const fieldsObj = this.arrayToObject(fields);
    
    return {
      id: fieldsObj.id || messageId,
      type: fieldsObj.type as any,
      timestamp: parseInt(fieldsObj.timestamp, 10),
      version: fieldsObj.version,
      source: fieldsObj.source,
      correlationId: fieldsObj.correlationId || undefined,
      causationId: fieldsObj.causationId || undefined,
      userId: fieldsObj.userId || undefined,
      data: JSON.parse(fieldsObj.data),
      metadata: fieldsObj.metadata ? JSON.parse(fieldsObj.metadata) : undefined,
    };
  }

  /**
   * Convert Redis array to object
   */
  private arrayToObject(arr: string[]): Record<string, string> {
    const obj: Record<string, string> = {};
    for (let i = 0; i < arr.length; i += 2) {
      obj[arr[i]] = arr[i + 1];
    }
    return obj;
  }

  /**
   * Health check
   */
  public async healthCheck(): Promise<boolean> {
    try {
      return await redisConnection.healthCheck();
    } catch (error) {
      logger.error('Redis Streams service health check failed:', error);
      return false;
    }
  }

  /**
   * Get service statistics
   */
  public async getStats(): Promise<Record<string, any>> {
    const redis = this.ensureRedis();
    
    try {
      const stats: Record<string, any> = {
        isInitialized: this.isInitialized,
        streams: {},
      };

      for (const streamName of Object.values(STREAM_NAMES)) {
        const streamInfo = await this.getStreamInfo(streamName);
        if (streamInfo) {
          stats.streams[streamName] = streamInfo;
        }
      }

      return stats;
    } catch (error) {
      logger.error('Failed to get Redis Streams stats:', error);
      return { isInitialized: this.isInitialized, error: error instanceof Error ? error.message : 'Unknown error' };
    }
  }
}

// Export singleton instance
export const redisStreamsService = RedisStreamsService.getInstance();
