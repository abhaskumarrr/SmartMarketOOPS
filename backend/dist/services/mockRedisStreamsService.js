"use strict";
/**
 * Mock Redis Streams Service for Testing
 * Provides a mock implementation for testing without actual Redis
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.mockRedisStreamsService = exports.MockRedisStreamsService = void 0;
const logger_1 = require("../utils/logger");
const events_1 = require("../types/events");
class MockRedisStreamsService {
    constructor() {
        this.isInitialized = false;
        this.streams = new Map();
        this.consumerGroups = new Map();
        this.messageIdCounter = 0;
        // Initialize mock streams
        Object.values(events_1.STREAM_NAMES).forEach(streamName => {
            this.streams.set(streamName, []);
        });
    }
    static getInstance() {
        if (!MockRedisStreamsService.instance) {
            MockRedisStreamsService.instance = new MockRedisStreamsService();
        }
        return MockRedisStreamsService.instance;
    }
    async initialize() {
        try {
            logger_1.logger.info('🔌 Initializing Mock Redis Streams service...');
            // Create mock consumer groups
            await this.createConsumerGroups();
            this.isInitialized = true;
            logger_1.logger.info('✅ Mock Redis Streams service initialized successfully');
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to initialize Mock Redis Streams service:', error);
            throw error;
        }
    }
    async shutdown() {
        try {
            this.isInitialized = false;
            this.streams.clear();
            this.consumerGroups.clear();
            this.messageIdCounter = 0;
            logger_1.logger.info('🔌 Mock Redis Streams service shutdown completed');
        }
        catch (error) {
            logger_1.logger.error('❌ Error during Mock Redis Streams service shutdown:', error);
            throw error;
        }
    }
    /**
     * Publish an event to a stream
     */
    async publishEvent(streamName, event) {
        if (!this.isInitialized) {
            throw new Error('Mock Redis Streams service not initialized');
        }
        try {
            const messageId = `${Date.now()}-${this.messageIdCounter++}`;
            const message = {
                id: messageId,
                event: { ...event, id: event.id || messageId },
                timestamp: Date.now(),
            };
            const stream = this.streams.get(streamName) || [];
            stream.push(message);
            this.streams.set(streamName, stream);
            logger_1.logger.debug(`📤 Mock: Published event to ${streamName}: ${event.type}`);
            return messageId;
        }
        catch (error) {
            logger_1.logger.error(`❌ Failed to publish event to ${streamName}:`, error);
            throw error;
        }
    }
    /**
     * Publish multiple events in a pipeline
     */
    async publishEvents(streamName, events) {
        const messageIds = [];
        for (const event of events) {
            const messageId = await this.publishEvent(streamName, event);
            messageIds.push(messageId);
        }
        logger_1.logger.debug(`📤 Mock: Published ${events.length} events to ${streamName}`);
        return messageIds;
    }
    /**
     * Read events from a stream as a consumer group member
     */
    async readEvents(streamName, options) {
        if (!this.isInitialized) {
            throw new Error('Mock Redis Streams service not initialized');
        }
        try {
            const stream = this.streams.get(streamName) || [];
            const count = options.count || 10;
            // Get the latest messages (simulate reading from stream)
            const messages = stream.slice(-count);
            const events = messages.map(msg => msg.event);
            logger_1.logger.debug(`📥 Mock: Read ${events.length} events from ${streamName}`);
            return events;
        }
        catch (error) {
            logger_1.logger.error(`❌ Failed to read events from ${streamName}:`, error);
            throw error;
        }
    }
    /**
     * Acknowledge event processing
     */
    async acknowledgeEvent(streamName, groupName, messageId) {
        logger_1.logger.debug(`✅ Mock: Acknowledged event ${messageId} in ${streamName}`);
    }
    /**
     * Acknowledge multiple events
     */
    async acknowledgeEvents(streamName, groupName, messageIds) {
        if (messageIds.length > 0) {
            logger_1.logger.debug(`✅ Mock: Acknowledged ${messageIds.length} events in ${streamName}`);
        }
    }
    /**
     * Create consumer groups for all streams
     */
    async createConsumerGroups() {
        const streams = Object.values(events_1.STREAM_NAMES);
        const groups = Object.values(events_1.CONSUMER_GROUPS);
        for (const streamName of streams) {
            for (const groupName of groups) {
                const key = `${streamName}:${groupName}`;
                if (!this.consumerGroups.has(key)) {
                    this.consumerGroups.set(key, new Set());
                    logger_1.logger.debug(`✅ Mock: Created consumer group ${groupName} for stream ${streamName}`);
                }
            }
        }
    }
    /**
     * Get stream information
     */
    async getStreamInfo(streamName) {
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
    async getConsumerGroupInfo(streamName, groupName) {
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
    async trimStream(streamName, maxLength, approximate = true) {
        const stream = this.streams.get(streamName) || [];
        const originalLength = stream.length;
        if (stream.length > maxLength) {
            const trimmed = stream.slice(-maxLength);
            this.streams.set(streamName, trimmed);
            const removedCount = originalLength - trimmed.length;
            logger_1.logger.debug(`🗑️ Mock: Trimmed ${removedCount} messages from ${streamName}`);
            return removedCount;
        }
        return 0;
    }
    /**
     * Health check
     */
    async healthCheck() {
        return this.isInitialized;
    }
    /**
     * Get service statistics
     */
    async getStats() {
        const stats = {
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
    getGroupsForStream(streamName) {
        let count = 0;
        for (const key of this.consumerGroups.keys()) {
            if (key.startsWith(`${streamName}:`)) {
                count++;
            }
        }
        return count;
    }
    eventToFields(event) {
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
    getStoredEvents(streamName) {
        return this.streams.get(streamName) || [];
    }
    getTotalEvents() {
        let total = 0;
        for (const [, messages] of this.streams) {
            total += messages.length;
        }
        return total;
    }
    clearStreams() {
        this.streams.clear();
        Object.values(events_1.STREAM_NAMES).forEach(streamName => {
            this.streams.set(streamName, []);
        });
        this.messageIdCounter = 0;
        logger_1.logger.info('🗑️ Mock: Cleared all streams');
    }
    getMockStats() {
        const streamStats = {};
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
exports.MockRedisStreamsService = MockRedisStreamsService;
// Export singleton instance
exports.mockRedisStreamsService = MockRedisStreamsService.getInstance();
//# sourceMappingURL=mockRedisStreamsService.js.map