/**
 * Redis Configuration for Event-Driven Architecture
 * Redis Streams configuration for high-performance event processing
 */

import Redis from 'ioredis';
import { logger } from '../utils/logger';

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

// Default Redis configuration
const defaultRedisConfig: RedisConfig = {
  host: process.env.REDIS_HOST || 'localhost',
  port: parseInt(process.env.REDIS_PORT || '6379', 10),
  password: process.env.REDIS_PASSWORD,
  db: parseInt(process.env.REDIS_DB || '0', 10),
  retryDelayOnFailover: parseInt(process.env.REDIS_RETRY_DELAY || '100', 10),
  maxRetriesPerRequest: parseInt(process.env.REDIS_MAX_RETRIES || '3', 10),
  lazyConnect: true,
  keepAlive: parseInt(process.env.REDIS_KEEP_ALIVE || '30000', 10),
  family: 4,
  keyPrefix: process.env.REDIS_KEY_PREFIX || 'smoops:',
  connectTimeout: parseInt(process.env.REDIS_CONNECT_TIMEOUT || '10000', 10),
  commandTimeout: parseInt(process.env.REDIS_COMMAND_TIMEOUT || '5000', 10),
};

// Default Redis Streams configuration
const defaultStreamConfig: RedisStreamConfig = {
  maxLength: parseInt(process.env.REDIS_STREAM_MAX_LENGTH || '10000', 10),
  approximateMaxLength: process.env.REDIS_STREAM_APPROXIMATE === 'true',
  trimStrategy: (process.env.REDIS_STREAM_TRIM_STRATEGY as 'MAXLEN' | 'MINID') || 'MAXLEN',
  consumerGroupName: process.env.REDIS_CONSUMER_GROUP || 'trading-system',
  consumerName: process.env.REDIS_CONSUMER_NAME || `consumer-${process.pid}`,
  blockTime: parseInt(process.env.REDIS_BLOCK_TIME || '1000', 10),
  count: parseInt(process.env.REDIS_STREAM_COUNT || '10', 10),
  retryInterval: parseInt(process.env.REDIS_RETRY_INTERVAL || '5000', 10),
  deadLetterThreshold: parseInt(process.env.REDIS_DEAD_LETTER_THRESHOLD || '5', 10),
};

export class RedisConnection {
  private static instance: RedisConnection;
  private redis: Redis | null = null;
  private subscriber: Redis | null = null;
  private publisher: Redis | null = null;
  private config: RedisConfig;
  private streamConfig: RedisStreamConfig;
  private isConnected: boolean = false;
  private connectionPromise: Promise<void> | null = null;

  private constructor() {
    this.config = { ...defaultRedisConfig };
    this.streamConfig = { ...defaultStreamConfig };
  }

  public static getInstance(): RedisConnection {
    if (!RedisConnection.instance) {
      RedisConnection.instance = new RedisConnection();
    }
    return RedisConnection.instance;
  }

  public async connect(): Promise<void> {
    if (this.isConnected && this.redis) {
      return;
    }

    if (this.connectionPromise) {
      return this.connectionPromise;
    }

    this.connectionPromise = this._connect();
    return this.connectionPromise;
  }

  private async _connect(): Promise<void> {
    try {
      logger.info('🔌 Connecting to Redis...', {
        host: this.config.host,
        port: this.config.port,
        db: this.config.db,
      });

      // Create main Redis connection
      this.redis = new Redis({
        host: this.config.host,
        port: this.config.port,
        password: this.config.password,
        db: this.config.db,
        maxRetriesPerRequest: this.config.maxRetriesPerRequest,
        lazyConnect: this.config.lazyConnect,
        keepAlive: this.config.keepAlive,
        family: this.config.family,
        keyPrefix: this.config.keyPrefix,
        connectTimeout: this.config.connectTimeout,
        commandTimeout: this.config.commandTimeout,
      });

      // Create subscriber connection (separate connection for pub/sub)
      this.subscriber = new Redis({
        host: this.config.host,
        port: this.config.port,
        password: this.config.password,
        db: this.config.db,
        maxRetriesPerRequest: this.config.maxRetriesPerRequest,
        lazyConnect: this.config.lazyConnect,
        keepAlive: this.config.keepAlive,
        family: this.config.family,
        keyPrefix: this.config.keyPrefix,
        connectTimeout: this.config.connectTimeout,
        commandTimeout: this.config.commandTimeout,
      });

      // Create publisher connection
      this.publisher = new Redis({
        host: this.config.host,
        port: this.config.port,
        password: this.config.password,
        db: this.config.db,
        maxRetriesPerRequest: this.config.maxRetriesPerRequest,
        lazyConnect: this.config.lazyConnect,
        keepAlive: this.config.keepAlive,
        family: this.config.family,
        keyPrefix: this.config.keyPrefix,
        connectTimeout: this.config.connectTimeout,
        commandTimeout: this.config.commandTimeout,
      });

      // Set up error handlers
      this.redis.on('error', (error) => {
        logger.error('Redis main connection error:', error);
      });

      this.subscriber.on('error', (error) => {
        logger.error('Redis subscriber connection error:', error);
      });

      this.publisher.on('error', (error) => {
        logger.error('Redis publisher connection error:', error);
      });

      // Set up connection handlers
      this.redis.on('connect', () => {
        logger.info('✅ Redis main connection established');
      });

      this.subscriber.on('connect', () => {
        logger.info('✅ Redis subscriber connection established');
      });

      this.publisher.on('connect', () => {
        logger.info('✅ Redis publisher connection established');
      });

      // Test connections
      await Promise.all([
        this.redis.ping(),
        this.subscriber.ping(),
        this.publisher.ping(),
      ]);

      this.isConnected = true;
      logger.info('✅ All Redis connections established successfully');

    } catch (error) {
      this.isConnected = false;
      this.redis = null;
      this.subscriber = null;
      this.publisher = null;
      this.connectionPromise = null;
      logger.error('❌ Failed to connect to Redis:', error);
      throw new Error(`Redis connection failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  public async disconnect(): Promise<void> {
    try {
      if (this.redis) {
        this.redis.disconnect(false);
      }

      if (this.subscriber) {
        this.subscriber.disconnect(false);
      }

      if (this.publisher) {
        this.publisher.disconnect(false);
      }

      this.redis = null;
      this.subscriber = null;
      this.publisher = null;
      this.isConnected = false;
      this.connectionPromise = null;

      logger.info('🔌 All Redis connections closed');
    } catch (error) {
      logger.error('❌ Error closing Redis connections:', error);
      throw error;
    }
  }

  public getRedis(): Redis {
    if (!this.redis || !this.isConnected) {
      throw new Error('Redis not connected. Call connect() first.');
    }
    return this.redis;
  }

  public getSubscriber(): Redis {
    if (!this.subscriber || !this.isConnected) {
      throw new Error('Redis subscriber not connected. Call connect() first.');
    }
    return this.subscriber;
  }

  public getPublisher(): Redis {
    if (!this.publisher || !this.isConnected) {
      throw new Error('Redis publisher not connected. Call connect() first.');
    }
    return this.publisher;
  }

  public getStreamConfig(): RedisStreamConfig {
    return { ...this.streamConfig };
  }

  public updateConfig(newConfig: Partial<RedisConfig>): void {
    this.config = { ...this.config, ...newConfig };
  }

  public updateStreamConfig(newConfig: Partial<RedisStreamConfig>): void {
    this.streamConfig = { ...this.streamConfig, ...newConfig };
  }

  public isReady(): boolean {
    return this.isConnected && 
           this.redis !== null && 
           this.subscriber !== null && 
           this.publisher !== null;
  }

  // Health check method
  public async healthCheck(): Promise<boolean> {
    try {
      if (!this.isReady()) {
        return false;
      }

      const results = await Promise.all([
        this.redis!.ping(),
        this.subscriber!.ping(),
        this.publisher!.ping(),
      ]);

      return results.every(result => result === 'PONG');
    } catch (error) {
      logger.error('Redis health check failed:', error);
      return false;
    }
  }

  // Get connection statistics
  public getStats(): {
    isConnected: boolean;
    config: RedisConfig;
    streamConfig: RedisStreamConfig;
  } {
    return {
      isConnected: this.isConnected,
      config: this.config,
      streamConfig: this.streamConfig,
    };
  }
}

// Export singleton instance
export const redisConnection = RedisConnection.getInstance();

// Export configurations
export { defaultRedisConfig as redisConfig, defaultStreamConfig as streamConfig };

// Environment validation
export function validateRedisEnvironment(): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  // Check Redis host
  if (!process.env.REDIS_HOST && process.env.REDIS_HOST !== 'localhost') {
    // This is just a warning, localhost is fine for development
  }

  // Validate port
  const port = parseInt(process.env.REDIS_PORT || '6379', 10);
  if (isNaN(port) || port < 1 || port > 65535) {
    errors.push('Invalid Redis port number');
  }

  // Validate numeric configurations
  const numericConfigs = [
    'REDIS_DB',
    'REDIS_RETRY_DELAY',
    'REDIS_MAX_RETRIES',
    'REDIS_KEEP_ALIVE',
    'REDIS_CONNECT_TIMEOUT',
    'REDIS_COMMAND_TIMEOUT',
    'REDIS_STREAM_MAX_LENGTH',
    'REDIS_BLOCK_TIME',
    'REDIS_STREAM_COUNT',
    'REDIS_RETRY_INTERVAL',
    'REDIS_DEAD_LETTER_THRESHOLD',
  ];

  numericConfigs.forEach(configName => {
    const value = process.env[configName];
    if (value && (isNaN(parseInt(value, 10)) || parseInt(value, 10) < 0)) {
      errors.push(`Invalid ${configName} value: ${value}`);
    }
  });

  return {
    valid: errors.length === 0,
    errors,
  };
}

// Connection retry utility
export async function connectWithRetry(
  maxAttempts: number = 3,
  delay: number = 1000
): Promise<void> {
  let lastError: Error | null = null;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      await redisConnection.connect();
      return;
    } catch (error) {
      lastError = error instanceof Error ? error : new Error('Unknown connection error');
      logger.warn(`Redis connection attempt ${attempt}/${maxAttempts} failed: ${lastError.message}`);

      if (attempt < maxAttempts) {
        await new Promise(resolve => setTimeout(resolve, delay * attempt));
      }
    }
  }

  throw new Error(`Failed to connect to Redis after ${maxAttempts} attempts. Last error: ${lastError?.message}`);
}
