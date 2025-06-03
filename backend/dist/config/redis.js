"use strict";
/**
 * Redis Configuration for Event-Driven Architecture
 * Redis Streams configuration for high-performance event processing
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.streamConfig = exports.redisConfig = exports.redisConnection = exports.RedisConnection = void 0;
exports.validateRedisEnvironment = validateRedisEnvironment;
exports.connectWithRetry = connectWithRetry;
const ioredis_1 = __importDefault(require("ioredis"));
const logger_1 = require("../utils/logger");
// Default Redis configuration
const defaultRedisConfig = {
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
exports.redisConfig = defaultRedisConfig;
// Default Redis Streams configuration
const defaultStreamConfig = {
    maxLength: parseInt(process.env.REDIS_STREAM_MAX_LENGTH || '10000', 10),
    approximateMaxLength: process.env.REDIS_STREAM_APPROXIMATE === 'true',
    trimStrategy: process.env.REDIS_STREAM_TRIM_STRATEGY || 'MAXLEN',
    consumerGroupName: process.env.REDIS_CONSUMER_GROUP || 'trading-system',
    consumerName: process.env.REDIS_CONSUMER_NAME || `consumer-${process.pid}`,
    blockTime: parseInt(process.env.REDIS_BLOCK_TIME || '1000', 10),
    count: parseInt(process.env.REDIS_STREAM_COUNT || '10', 10),
    retryInterval: parseInt(process.env.REDIS_RETRY_INTERVAL || '5000', 10),
    deadLetterThreshold: parseInt(process.env.REDIS_DEAD_LETTER_THRESHOLD || '5', 10),
};
exports.streamConfig = defaultStreamConfig;
class RedisConnection {
    constructor() {
        this.redis = null;
        this.subscriber = null;
        this.publisher = null;
        this.isConnected = false;
        this.connectionPromise = null;
        this.config = { ...defaultRedisConfig };
        this.streamConfig = { ...defaultStreamConfig };
    }
    static getInstance() {
        if (!RedisConnection.instance) {
            RedisConnection.instance = new RedisConnection();
        }
        return RedisConnection.instance;
    }
    async connect() {
        if (this.isConnected && this.redis) {
            return;
        }
        if (this.connectionPromise) {
            return this.connectionPromise;
        }
        this.connectionPromise = this._connect();
        return this.connectionPromise;
    }
    async _connect() {
        try {
            logger_1.logger.info('🔌 Connecting to Redis...', {
                host: this.config.host,
                port: this.config.port,
                db: this.config.db,
            });
            // Create main Redis connection
            this.redis = new ioredis_1.default({
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
            this.subscriber = new ioredis_1.default({
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
            this.publisher = new ioredis_1.default({
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
                logger_1.logger.error('Redis main connection error:', error);
            });
            this.subscriber.on('error', (error) => {
                logger_1.logger.error('Redis subscriber connection error:', error);
            });
            this.publisher.on('error', (error) => {
                logger_1.logger.error('Redis publisher connection error:', error);
            });
            // Set up connection handlers
            this.redis.on('connect', () => {
                logger_1.logger.info('✅ Redis main connection established');
            });
            this.subscriber.on('connect', () => {
                logger_1.logger.info('✅ Redis subscriber connection established');
            });
            this.publisher.on('connect', () => {
                logger_1.logger.info('✅ Redis publisher connection established');
            });
            // Test connections
            await Promise.all([
                this.redis.ping(),
                this.subscriber.ping(),
                this.publisher.ping(),
            ]);
            this.isConnected = true;
            logger_1.logger.info('✅ All Redis connections established successfully');
        }
        catch (error) {
            this.isConnected = false;
            this.redis = null;
            this.subscriber = null;
            this.publisher = null;
            this.connectionPromise = null;
            logger_1.logger.error('❌ Failed to connect to Redis:', error);
            throw new Error(`Redis connection failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
    }
    async disconnect() {
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
            logger_1.logger.info('🔌 All Redis connections closed');
        }
        catch (error) {
            logger_1.logger.error('❌ Error closing Redis connections:', error);
            throw error;
        }
    }
    getRedis() {
        if (!this.redis || !this.isConnected) {
            throw new Error('Redis not connected. Call connect() first.');
        }
        return this.redis;
    }
    getSubscriber() {
        if (!this.subscriber || !this.isConnected) {
            throw new Error('Redis subscriber not connected. Call connect() first.');
        }
        return this.subscriber;
    }
    getPublisher() {
        if (!this.publisher || !this.isConnected) {
            throw new Error('Redis publisher not connected. Call connect() first.');
        }
        return this.publisher;
    }
    getStreamConfig() {
        return { ...this.streamConfig };
    }
    updateConfig(newConfig) {
        this.config = { ...this.config, ...newConfig };
    }
    updateStreamConfig(newConfig) {
        this.streamConfig = { ...this.streamConfig, ...newConfig };
    }
    isReady() {
        return this.isConnected &&
            this.redis !== null &&
            this.subscriber !== null &&
            this.publisher !== null;
    }
    // Health check method
    async healthCheck() {
        try {
            if (!this.isReady()) {
                return false;
            }
            const results = await Promise.all([
                this.redis.ping(),
                this.subscriber.ping(),
                this.publisher.ping(),
            ]);
            return results.every(result => result === 'PONG');
        }
        catch (error) {
            logger_1.logger.error('Redis health check failed:', error);
            return false;
        }
    }
    // Get connection statistics
    getStats() {
        return {
            isConnected: this.isConnected,
            config: this.config,
            streamConfig: this.streamConfig,
        };
    }
}
exports.RedisConnection = RedisConnection;
// Export singleton instance
exports.redisConnection = RedisConnection.getInstance();
// Environment validation
function validateRedisEnvironment() {
    const errors = [];
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
async function connectWithRetry(maxAttempts = 3, delay = 1000) {
    let lastError = null;
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
            await exports.redisConnection.connect();
            return;
        }
        catch (error) {
            lastError = error instanceof Error ? error : new Error('Unknown connection error');
            logger_1.logger.warn(`Redis connection attempt ${attempt}/${maxAttempts} failed: ${lastError.message}`);
            if (attempt < maxAttempts) {
                await new Promise(resolve => setTimeout(resolve, delay * attempt));
            }
        }
    }
    throw new Error(`Failed to connect to Redis after ${maxAttempts} attempts. Last error: ${lastError?.message}`);
}
//# sourceMappingURL=redis.js.map