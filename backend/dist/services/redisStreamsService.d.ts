/**
 * Redis Streams Service
 * High-performance event streaming service for real-time trading events
 */
import { TradingEvent, StreamName } from '../types/events';
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
export declare class RedisStreamsService {
    private static instance;
    private redis;
    private isInitialized;
    private constructor();
    static getInstance(): RedisStreamsService;
    initialize(): Promise<void>;
    shutdown(): Promise<void>;
    private ensureRedis;
    /**
     * Publish an event to a stream
     */
    publishEvent(streamName: StreamName, event: TradingEvent, options?: ProducerOptions): Promise<string>;
    /**
     * Publish multiple events in a pipeline for better performance
     */
    publishEvents(streamName: StreamName, events: TradingEvent[], options?: ProducerOptions): Promise<string[]>;
    /**
     * Read events from a stream as a consumer group member
     */
    readEvents(streamName: StreamName, options: ConsumerOptions): Promise<TradingEvent[]>;
    /**
     * Read pending events for a consumer
     */
    readPendingEvents(streamName: StreamName, groupName: string, consumerName: string, count?: number): Promise<TradingEvent[]>;
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
    getStreamInfo(streamName: StreamName): Promise<StreamInfo | null>;
    /**
     * Get consumer group information
     */
    getConsumerGroupInfo(streamName: StreamName, groupName: string): Promise<ConsumerInfo[]>;
    /**
     * Trim stream to maximum length
     */
    trimStream(streamName: StreamName, maxLength: number, approximate?: boolean): Promise<number>;
    /**
     * Serialize event to Redis fields
     */
    private serializeEvent;
    /**
     * Deserialize Redis fields to event
     */
    private deserializeEvent;
    /**
     * Convert Redis array to object
     */
    private arrayToObject;
    /**
     * Health check
     */
    healthCheck(): Promise<boolean>;
    /**
     * Get service statistics
     */
    getStats(): Promise<Record<string, any>>;
}
export declare const redisStreamsService: RedisStreamsService;
