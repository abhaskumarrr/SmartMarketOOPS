/**
 * Mock Redis Streams Service for Testing
 * Provides a mock implementation for testing without actual Redis
 */

import { logger } from '../utils/logger';
import {
  TradingEvent,
  StreamName,
  ConsumerGroup,
  STREAM_NAMES,
  CONSUMER_GROUPS,
} from '../types/events';

export interface MockStreamMessage {
  id: string;
  event: TradingEvent;
  timestamp: number;
}

export class MockRedisStreamsService {
  private static instance: MockRedisStreamsService;
  private isInitialized: boolean = false;
  private streams: Map<StreamName, MockStreamMessage[]> = new Map();
  private consumerGroups: Map<string, Set<string>> = new Map();
  private messageIdCounter: number = 0;

  private constructor() {
    // Initialize mock streams
    Object.values(STREAM_NAMES).forEach(streamName => {
      this.streams.set(streamName, []);
    });
  }

  public static getInstance(): MockRedisStreamsService {
    if (!MockRedisStreamsService.instance) {
      MockRedisStreamsService.instance = new MockRedisStreamsService();
    }
    return MockRedisStreamsService.instance;
  }

  public async initialize(): Promise<void> {
    try {
      logger.info('🔌 Initializing Mock Redis Streams service...');
      
      // Create mock consumer groups
      await this.createConsumerGroups();
      
      this.isInitialized = true;
      logger.info('✅ Mock Redis Streams service initialized successfully');
    } catch (error) {
      logger.error('❌ Failed to initialize Mock Redis Streams service:', error);
      throw error;
    }
  }

  public async shutdown(): Promise<void> {
    try {
      this.isInitialized = false;
      this.streams.clear();
      this.consumerGroups.clear();
      this.messageIdCounter = 0;
      logger.info('🔌 Mock Redis Streams service shutdown completed');
    } catch (error) {
      logger.error('❌ Error during Mock Redis Streams service shutdown:', error);
      throw error;
    }
  }

  /**
   * Publish an event to a stream
   */
  public async publishEvent(
    streamName: StreamName,
    event: TradingEvent
  ): Promise<string> {
    if (!this.isInitialized) {
      throw new Error('Mock Redis Streams service not initialized');
    }

    try {
      const messageId = `${Date.now()}-${this.messageIdCounter++}`;
      
      const message: MockStreamMessage = {
        id: messageId,
        event: { ...event, id: event.id || messageId },
        timestamp: Date.now(),
      };

      const stream = this.streams.get(streamName) || [];
      stream.push(message);
      this.streams.set(streamName, stream);

      logger.debug(`📤 Mock: Published event to ${streamName}: ${event.type}`);
      return messageId;
    } catch (error) {
      logger.error(`❌ Failed to publish event to ${streamName}:`, error);
      throw error;
    }
  }

  /**
   * Publish multiple events in a pipeline
   */
  public async publishEvents(
    streamName: StreamName,
    events: TradingEvent[]
  ): Promise<string[]> {
    const messageIds: string[] = [];

    for (const event of events) {
      const messageId = await this.publishEvent(streamName, event);
      messageIds.push(messageId);
    }

    logger.debug(`📤 Mock: Published ${events.length} events to ${streamName}`);
    return messageIds;
  }

  /**
   * Read events from a stream as a consumer group member
   */
  public async readEvents(
    streamName: StreamName,
    options: {
      groupName: string;
      consumerName: string;
      blockTime?: number;
      count?: number;
    }
  ): Promise<TradingEvent[]> {
    if (!this.isInitialized) {
      throw new Error('Mock Redis Streams service not initialized');
    }

    try {
      const stream = this.streams.get(streamName) || [];
      const count = options.count || 10;
      
      // Get the latest messages (simulate reading from stream)
      const messages = stream.slice(-count);
      const events = messages.map(msg => msg.event);

      logger.debug(`📥 Mock: Read ${events.length} events from ${streamName}`);
      return events;
    } catch (error) {
      logger.error(`❌ Failed to read events from ${streamName}:`, error);
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
    logger.debug(`✅ Mock: Acknowledged event ${messageId} in ${streamName}`);
  }

  /**
   * Acknowledge multiple events
   */
  public async acknowledgeEvents(
    streamName: StreamName,
    groupName: string,
    messageIds: string[]
  ): Promise<void> {
    if (messageIds.length > 0) {
      logger.debug(`✅ Mock: Acknowledged ${messageIds.length} events in ${streamName}`);
    }
  }

  /**
   * Create consumer groups for all streams
   */
  private async createConsumerGroups(): Promise<void> {
    const streams = Object.values(STREAM_NAMES);
    const groups = Object.values(CONSUMER_GROUPS);

    for (const streamName of streams) {
      for (const groupName of groups) {
        const key = `${streamName}:${groupName}`;
        if (!this.consumerGroups.has(key)) {
          this.consumerGroups.set(key, new Set());
          logger.debug(`✅ Mock: Created consumer group ${groupName} for stream ${streamName}`);
        }
      }
    }
  }

  /**
   * Get stream information
   */
  public async getStreamInfo(streamName: StreamName): Promise<any> {
    const stream = this.streams.get(streamName) || [];
    
    return {
      length: stream.length,
      radixTreeKeys: 1,
      radixTreeNodes: 1,
      groups: this.getGroupsForStream(streamName),
      lastGeneratedId: stream.length > 0 ? stream[stream.length - 1].id : '0-0',
      firstEntry: stream.length > 0 ? {
        id: stream[0].id,
        fields: this.eventToFields(stream[0].event),
      } : undefined,
      lastEntry: stream.length > 0 ? {
        id: stream[stream.length - 1].id,
        fields: this.eventToFields(stream[stream.length - 1].event),
      } : undefined,
    };
  }

  /**
   * Get consumer group information
   */
  public async getConsumerGroupInfo(
    streamName: StreamName,
    groupName: string
  ): Promise<any[]> {
    const key = `${streamName}:${groupName}`;
    const consumers = this.consumerGroups.get(key) || new Set();
    
    return Array.from(consumers).map(consumerName => ({
      name: consumerName,
      pending: 0,
      idle: 0,
    }));
  }

  /**
   * Trim stream to maximum length
   */
  public async trimStream(
    streamName: StreamName,
    maxLength: number,
    approximate: boolean = true
  ): Promise<number> {
    const stream = this.streams.get(streamName) || [];
    const originalLength = stream.length;
    
    if (stream.length > maxLength) {
      const trimmed = stream.slice(-maxLength);
      this.streams.set(streamName, trimmed);
      const removedCount = originalLength - trimmed.length;
      
      logger.debug(`🗑️ Mock: Trimmed ${removedCount} messages from ${streamName}`);
      return removedCount;
    }
    
    return 0;
  }

  /**
   * Health check
   */
  public async healthCheck(): Promise<boolean> {
    return this.isInitialized;
  }

  /**
   * Get service statistics
   */
  public async getStats(): Promise<Record<string, any>> {
    const stats: Record<string, any> = {
      isInitialized: this.isInitialized,
      streams: {},
    };

    for (const [streamName, messages] of this.streams) {
      stats.streams[streamName] = {
        length: messages.length,
        lastMessageId: messages.length > 0 ? messages[messages.length - 1].id : null,
      };
    }

    return stats;
  }

  // Helper methods
  private getGroupsForStream(streamName: StreamName): number {
    let count = 0;
    for (const key of this.consumerGroups.keys()) {
      if (key.startsWith(`${streamName}:`)) {
        count++;
      }
    }
    return count;
  }

  private eventToFields(event: TradingEvent): Record<string, string> {
    return {
      id: event.id || '',
      type: event.type,
      timestamp: event.timestamp.toString(),
      version: event.version,
      source: event.source,
      data: JSON.stringify(event.data),
    };
  }

  // Mock-specific methods for testing
  public getStoredEvents(streamName: StreamName): MockStreamMessage[] {
    return this.streams.get(streamName) || [];
  }

  public getTotalEvents(): number {
    let total = 0;
    for (const [, messages] of this.streams) {
      total += messages.length;
    }
    return total;
  }

  public clearStreams(): void {
    this.streams.clear();
    Object.values(STREAM_NAMES).forEach(streamName => {
      this.streams.set(streamName, []);
    });
    this.messageIdCounter = 0;
    logger.info('🗑️ Mock: Cleared all streams');
  }

  public getMockStats(): {
    isInitialized: boolean;
    totalEvents: number;
    streamStats: Record<string, number>;
    consumerGroups: number;
  } {
    const streamStats: Record<string, number> = {};
    for (const [streamName, messages] of this.streams) {
      streamStats[streamName] = messages.length;
    }

    return {
      isInitialized: this.isInitialized,
      totalEvents: this.getTotalEvents(),
      streamStats,
      consumerGroups: this.consumerGroups.size,
    };
  }
}

// Export singleton instance
export const mockRedisStreamsService = MockRedisStreamsService.getInstance();
