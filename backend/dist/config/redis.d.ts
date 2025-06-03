/**
 * Redis Configuration for Event-Driven Architecture
 * Redis Streams configuration for high-performance event processing
 */
import Redis from 'ioredis';
export interface RedisConfig {
    host: string;
    port: number;
    password?: string;
    db: number;
    retryDelayOnFailover: number;
    maxRetriesPerRequest: number;
    lazyConnect: boolean;
    keepAlive: number;
    family: number;
    keyPrefix?: string;
    connectTimeout: number;
    commandTimeout: number;
}
export interface RedisStreamConfig {
    maxLength: number;
    approximateMaxLength: boolean;
    trimStrategy: 'MAXLEN' | 'MINID';
    consumerGroupName: string;
    consumerName: string;
    blockTime: number;
    count: number;
    retryInterval: number;
    deadLetterThreshold: number;
}
declare const defaultRedisConfig: RedisConfig;
declare const defaultStreamConfig: RedisStreamConfig;
export declare class RedisConnection {
    private static instance;
    private redis;
    private subscriber;
    private publisher;
    private config;
    private streamConfig;
    private isConnected;
    private connectionPromise;
    private constructor();
    static getInstance(): RedisConnection;
    connect(): Promise<void>;
    private _connect;
    disconnect(): Promise<void>;
    getRedis(): Redis;
    getSubscriber(): Redis;
    getPublisher(): Redis;
    getStreamConfig(): RedisStreamConfig;
    updateConfig(newConfig: Partial<RedisConfig>): void;
    updateStreamConfig(newConfig: Partial<RedisStreamConfig>): void;
    isReady(): boolean;
    healthCheck(): Promise<boolean>;
    getStats(): {
        isConnected: boolean;
        config: RedisConfig;
        streamConfig: RedisStreamConfig;
    };
}
export declare const redisConnection: RedisConnection;
export { defaultRedisConfig as redisConfig, defaultStreamConfig as streamConfig };
export declare function validateRedisEnvironment(): {
    valid: boolean;
    errors: string[];
};
export declare function connectWithRetry(maxAttempts?: number, delay?: number): Promise<void>;
