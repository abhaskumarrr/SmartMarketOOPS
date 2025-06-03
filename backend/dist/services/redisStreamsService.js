"use strict";
/**
 * Redis Streams Service
 * High-performance event streaming service for real-time trading events
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.redisStreamsService = exports.RedisStreamsService = void 0;
const redis_1 = require("../config/redis");
const logger_1 = require("../utils/logger");
const events_1 = require("../types/events");
class RedisStreamsService {
    constructor() {
        this.redis = null;
        this.isInitialized = false;
    }
    static getInstance() {
        if (!RedisStreamsService.instance) {
            RedisStreamsService.instance = new RedisStreamsService();
        }
        return RedisStreamsService.instance;
    }
    async initialize() {
        try {
            await redis_1.redisConnection.connect();
            this.redis = redis_1.redisConnection.getRedis();
            // Mark as initialized before creating consumer groups
            this.isInitialized = true;
            // Create consumer groups for all streams
            await this.createConsumerGroups();
            logger_1.logger.info('✅ Redis Streams service initialized successfully');
        }
        catch (error) {
            this.isInitialized = false;
            logger_1.logger.error('❌ Failed to initialize Redis Streams service:', error);
            throw error;
        }
    }
    async shutdown() {
        try {
            await redis_1.redisConnection.disconnect();
            this.redis = null;
            this.isInitialized = false;
            logger_1.logger.info('🔌 Redis Streams service shutdown completed');
        }
        catch (error) {
            logger_1.logger.error('❌ Error during Redis Streams service shutdown:', error);
            throw error;
        }
    }
    ensureRedis() {
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
    async publishEvent(streamName, event, options = {}) {
        const redis = this.ensureRedis();
        try {
            // Ensure event has an ID
            if (!event.id) {
                event.id = (0, events_1.createEventId)();
            }
            // Serialize event data
            const fields = this.serializeEvent(event);
            // Build XADD command
            const args = [streamName];
            // Add trimming options
            if (options.maxLength) {
                if (options.approximateMaxLength) {
                    args.push('MAXLEN', '~', options.maxLength);
                }
                else {
                    args.push('MAXLEN', options.maxLength);
                }
            }
            // Add event ID (use * for auto-generation)
            args.push('*');
            // Add fields
            Object.entries(fields).forEach(([key, value]) => {
                args.push(key, value);
            });
            const messageId = await redis.xadd(...args);
            logger_1.logger.debug(`📤 Published event to ${streamName}:`, {
                eventId: event.id,
                messageId,
                type: event.type,
                source: event.source,
            });
            return messageId;
        }
        catch (error) {
            logger_1.logger.error(`❌ Failed to publish event to ${streamName}:`, error);
            throw error;
        }
    }
    /**
     * Publish multiple events in a pipeline for better performance
     */
    async publishEvents(streamName, events, options = {}) {
        const redis = this.ensureRedis();
        try {
            const pipeline = redis.pipeline();
            for (const event of events) {
                if (!event.id) {
                    event.id = (0, events_1.createEventId)();
                }
                const fields = this.serializeEvent(event);
                const args = [streamName];
                if (options.maxLength) {
                    if (options.approximateMaxLength) {
                        args.push('MAXLEN', '~', options.maxLength);
                    }
                    else {
                        args.push('MAXLEN', options.maxLength);
                    }
                }
                args.push('*');
                Object.entries(fields).forEach(([key, value]) => {
                    args.push(key, value);
                });
                pipeline.xadd(...args);
            }
            const results = await pipeline.exec();
            const messageIds = [];
            if (results) {
                for (let i = 0; i < results.length; i++) {
                    const [error, result] = results[i];
                    if (error) {
                        logger_1.logger.error(`❌ Failed to publish event ${i} to ${streamName}:`, error);
                        throw error;
                    }
                    messageIds.push(result);
                }
            }
            logger_1.logger.debug(`📤 Published ${events.length} events to ${streamName}`);
            return messageIds;
        }
        catch (error) {
            logger_1.logger.error(`❌ Failed to publish events to ${streamName}:`, error);
            throw error;
        }
    }
    // ============================================================================
    // CONSUMER METHODS
    // ============================================================================
    /**
     * Read events from a stream as a consumer group member
     */
    async readEvents(streamName, options) {
        const redis = this.ensureRedis();
        try {
            const result = await redis.xreadgroup('GROUP', options.groupName, options.consumerName, 'COUNT', options.count || 10, 'BLOCK', options.blockTime || 1000, 'STREAMS', streamName, '>');
            if (!result || result.length === 0) {
                return [];
            }
            const events = [];
            for (const [stream, messages] of result) {
                for (const [messageId, fields] of messages) {
                    try {
                        const event = this.deserializeEvent(messageId, fields);
                        events.push(event);
                    }
                    catch (error) {
                        logger_1.logger.error(`❌ Failed to deserialize event ${messageId}:`, error);
                    }
                }
            }
            logger_1.logger.debug(`📥 Read ${events.length} events from ${streamName}`);
            return events;
        }
        catch (error) {
            logger_1.logger.error(`❌ Failed to read events from ${streamName}:`, error);
            throw error;
        }
    }
    /**
     * Read pending events for a consumer
     */
    async readPendingEvents(streamName, groupName, consumerName, count = 10) {
        const redis = this.ensureRedis();
        try {
            const result = await redis.xreadgroup('GROUP', groupName, consumerName, 'COUNT', count, 'STREAMS', streamName, '0');
            if (!result || result.length === 0) {
                return [];
            }
            const events = [];
            for (const [stream, messages] of result) {
                for (const [messageId, fields] of messages) {
                    try {
                        const event = this.deserializeEvent(messageId, fields);
                        events.push(event);
                    }
                    catch (error) {
                        logger_1.logger.error(`❌ Failed to deserialize pending event ${messageId}:`, error);
                    }
                }
            }
            return events;
        }
        catch (error) {
            logger_1.logger.error(`❌ Failed to read pending events from ${streamName}:`, error);
            throw error;
        }
    }
    /**
     * Acknowledge event processing
     */
    async acknowledgeEvent(streamName, groupName, messageId) {
        const redis = this.ensureRedis();
        try {
            await redis.xack(streamName, groupName, messageId);
            logger_1.logger.debug(`✅ Acknowledged event ${messageId} in ${streamName}`);
        }
        catch (error) {
            logger_1.logger.error(`❌ Failed to acknowledge event ${messageId}:`, error);
            throw error;
        }
    }
    /**
     * Acknowledge multiple events
     */
    async acknowledgeEvents(streamName, groupName, messageIds) {
        const redis = this.ensureRedis();
        try {
            if (messageIds.length > 0) {
                await redis.xack(streamName, groupName, ...messageIds);
                logger_1.logger.debug(`✅ Acknowledged ${messageIds.length} events in ${streamName}`);
            }
        }
        catch (error) {
            logger_1.logger.error(`❌ Failed to acknowledge events:`, error);
            throw error;
        }
    }
    // ============================================================================
    // STREAM MANAGEMENT
    // ============================================================================
    /**
     * Create consumer groups for all streams
     */
    async createConsumerGroups() {
        const redis = this.ensureRedis();
        const streams = Object.values(events_1.STREAM_NAMES);
        const groups = Object.values(events_1.CONSUMER_GROUPS);
        for (const streamName of streams) {
            for (const groupName of groups) {
                try {
                    await redis.xgroup('CREATE', streamName, groupName, '$', 'MKSTREAM');
                    logger_1.logger.debug(`✅ Created consumer group ${groupName} for stream ${streamName}`);
                }
                catch (error) {
                    if (error.message && error.message.includes('BUSYGROUP')) {
                        // Group already exists, which is fine
                        logger_1.logger.debug(`ℹ️ Consumer group ${groupName} already exists for stream ${streamName}`);
                    }
                    else {
                        logger_1.logger.error(`❌ Failed to create consumer group ${groupName} for stream ${streamName}:`, error);
                    }
                }
            }
        }
    }
    /**
     * Get stream information
     */
    async getStreamInfo(streamName) {
        const redis = this.ensureRedis();
        try {
            const info = await redis.xinfo('STREAM', streamName);
            return {
                length: info[1],
                radixTreeKeys: info[3],
                radixTreeNodes: info[5],
                groups: info[7],
                lastGeneratedId: info[9],
                firstEntry: info[11] ? {
                    id: info[11][0],
                    fields: this.arrayToObject(info[11][1]),
                } : undefined,
                lastEntry: info[13] ? {
                    id: info[13][0],
                    fields: this.arrayToObject(info[13][1]),
                } : undefined,
            };
        }
        catch (error) {
            logger_1.logger.error(`❌ Failed to get stream info for ${streamName}:`, error);
            return null;
        }
    }
    /**
     * Get consumer group information
     */
    async getConsumerGroupInfo(streamName, groupName) {
        const redis = this.ensureRedis();
        try {
            const consumers = await redis.xinfo('CONSUMERS', streamName, groupName);
            return consumers.map((consumer) => ({
                name: consumer[1],
                pending: consumer[3],
                idle: consumer[5],
            }));
        }
        catch (error) {
            logger_1.logger.error(`❌ Failed to get consumer group info for ${streamName}/${groupName}:`, error);
            return [];
        }
    }
    /**
     * Trim stream to maximum length
     */
    async trimStream(streamName, maxLength, approximate = true) {
        const redis = this.ensureRedis();
        try {
            const args = [streamName, 'MAXLEN'];
            if (approximate) {
                args.push('~');
            }
            args.push(maxLength.toString());
            const trimmed = await redis.xtrim(...args);
            logger_1.logger.debug(`🗑️ Trimmed ${trimmed} messages from ${streamName}`);
            return trimmed;
        }
        catch (error) {
            logger_1.logger.error(`❌ Failed to trim stream ${streamName}:`, error);
            throw error;
        }
    }
    // ============================================================================
    // UTILITY METHODS
    // ============================================================================
    /**
     * Serialize event to Redis fields
     */
    serializeEvent(event) {
        return {
            id: event.id || (0, events_1.createEventId)(),
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
    deserializeEvent(messageId, fields) {
        const fieldsObj = this.arrayToObject(fields);
        return {
            id: fieldsObj.id || messageId,
            type: fieldsObj.type,
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
    arrayToObject(arr) {
        const obj = {};
        for (let i = 0; i < arr.length; i += 2) {
            obj[arr[i]] = arr[i + 1];
        }
        return obj;
    }
    /**
     * Health check
     */
    async healthCheck() {
        try {
            return await redis_1.redisConnection.healthCheck();
        }
        catch (error) {
            logger_1.logger.error('Redis Streams service health check failed:', error);
            return false;
        }
    }
    /**
     * Get service statistics
     */
    async getStats() {
        const redis = this.ensureRedis();
        try {
            const stats = {
                isInitialized: this.isInitialized,
                streams: {},
            };
            for (const streamName of Object.values(events_1.STREAM_NAMES)) {
                const streamInfo = await this.getStreamInfo(streamName);
                if (streamInfo) {
                    stats.streams[streamName] = streamInfo;
                }
            }
            return stats;
        }
        catch (error) {
            logger_1.logger.error('Failed to get Redis Streams stats:', error);
            return { isInitialized: this.isInitialized, error: error instanceof Error ? error.message : 'Unknown error' };
        }
    }
}
exports.RedisStreamsService = RedisStreamsService;
// Export singleton instance
exports.redisStreamsService = RedisStreamsService.getInstance();
//# sourceMappingURL=redisStreamsService.js.map