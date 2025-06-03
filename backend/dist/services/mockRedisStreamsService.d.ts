/**
 * Mock Redis Streams Service for Testing
 * Provides a mock implementation for testing without actual Redis
 */
import { TradingEvent, StreamName } from '../types/events';
export interface MockStreamMessage {
    id: string;
    event: TradingEvent;
    timestamp: number;
}
export declare class MockRedisStreamsService {
    private static instance;
    private isInitialized;
    private streams;
    private consumerGroups;
    private messageIdCounter;
    private constructor();
    static getInstance(): MockRedisStreamsService;
    initialize(): Promise<void>;
    shutdown(): Promise<void>;
    /**
     * Publish an event to a stream
     */
    publishEvent(streamName: StreamName, event: TradingEvent): Promise<string>;
    /**
     * Publish multiple events in a pipeline
     */
    publishEvents(streamName: StreamName, events: TradingEvent[]): Promise<string[]>;
    /**
     * Read events from a stream as a consumer group member
     */
    readEvents(streamName: StreamName, options: {
        groupName: string;
        consumerName: string;
        blockTime?: number;
        count?: number;
    }): Promise<TradingEvent[]>;
    /**
     * Acknowledge event processing
     */
    acknowledgeEvent(streamName: StreamName, groupName: string, messageId: string): Promise<void>;
    /**
     * Acknowledge multiple events
     */
    acknowledgeEvents(streamName: StreamName, groupName: string, messageIds: string[]): Promise<void>;
    /**
     * Create consumer groups for all streams
     */
    private createConsumerGroups;
    /**
     * Get stream information
     */
    getStreamInfo(streamName: StreamName): Promise<any>;
    /**
     * Get consumer group information
     */
    getConsumerGroupInfo(streamName: StreamName, groupName: string): Promise<any[]>;
    /**
     * Trim stream to maximum length
     */
    trimStream(streamName: StreamName, maxLength: number, approximate?: boolean): Promise<number>;
    /**
     * Health check
     */
    healthCheck(): Promise<boolean>;
    /**
     * Get service statistics
     */
    getStats(): Promise<Record<string, any>>;
    private getGroupsForStream;
    private eventToFields;
    getStoredEvents(streamName: StreamName): MockStreamMessage[];
    getTotalEvents(): number;
    clearStreams(): void;
    getMockStats(): {
        isInitialized: boolean;
        totalEvents: number;
        streamStats: Record<string, number>;
        consumerGroups: number;
    };
}
export declare const mockRedisStreamsService: MockRedisStreamsService;
