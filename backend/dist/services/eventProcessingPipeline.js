"use strict";
/**
 * Event Processing Pipeline
 * Core event-driven processing pipeline for trading system
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.eventProcessingPipeline = exports.EventProcessingPipeline = void 0;
const events_1 = require("events");
const redisStreamsService_1 = require("./redisStreamsService");
const logger_1 = require("../utils/logger");
const events_2 = require("../types/events");
class EventProcessingPipeline extends events_1.EventEmitter {
    constructor(config = {}) {
        super();
        this.processors = new Map();
        this.isRunning = false;
        this.startTime = 0;
        this.processingTimes = [];
        this.circuitBreakerOpen = false;
        this.circuitBreakerOpenTime = 0;
        this.consecutiveFailures = 0;
        this.config = {
            consumerName: `pipeline-${process.pid}`,
            batchSize: 10,
            blockTime: 1000,
            retryAttempts: 3,
            retryDelay: 1000,
            deadLetterThreshold: 5,
            processingTimeout: 30000,
            enableCircuitBreaker: true,
            circuitBreakerThreshold: 10,
            circuitBreakerTimeout: 60000,
            ...config,
        };
        this.stats = {
            processedEvents: 0,
            failedEvents: 0,
            retryEvents: 0,
            deadLetterEvents: 0,
            averageProcessingTime: 0,
            throughput: 0,
            circuitBreakerOpen: false,
            lastProcessedAt: 0,
            uptime: 0,
        };
    }
    /**
     * Register an event processor
     */
    registerProcessor(processor) {
        this.processors.set(processor.getName(), processor);
        logger_1.logger.info(`📝 Registered event processor: ${processor.getName()}`);
    }
    /**
     * Start the event processing pipeline
     */
    async start() {
        if (this.isRunning) {
            logger_1.logger.warn('⚠️ Event processing pipeline is already running');
            return;
        }
        try {
            await redisStreamsService_1.redisStreamsService.initialize();
            this.isRunning = true;
            this.startTime = Date.now();
            logger_1.logger.info('🚀 Starting event processing pipeline...');
            // Start processing loops for each stream
            const processingPromises = [
                this.processStream(events_2.STREAM_NAMES.MARKET_DATA, events_2.CONSUMER_GROUPS.SIGNAL_PROCESSOR),
                this.processStream(events_2.STREAM_NAMES.TRADING_SIGNALS, events_2.CONSUMER_GROUPS.ORDER_EXECUTOR),
                this.processStream(events_2.STREAM_NAMES.ORDERS, events_2.CONSUMER_GROUPS.PORTFOLIO_MANAGER),
                this.processStream(events_2.STREAM_NAMES.RISK_MANAGEMENT, events_2.CONSUMER_GROUPS.RISK_MANAGER),
                this.processStream(events_2.STREAM_NAMES.SYSTEM, events_2.CONSUMER_GROUPS.MONITORING),
                this.processStream(events_2.STREAM_NAMES.BOTS, events_2.CONSUMER_GROUPS.MONITORING),
            ];
            // Start stats update loop
            this.startStatsUpdate();
            this.emit('started');
            logger_1.logger.info('✅ Event processing pipeline started successfully');
            // Wait for all processing loops (they run indefinitely)
            await Promise.all(processingPromises);
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to start event processing pipeline:', error);
            this.isRunning = false;
            throw error;
        }
    }
    /**
     * Stop the event processing pipeline
     */
    async stop() {
        if (!this.isRunning) {
            logger_1.logger.warn('⚠️ Event processing pipeline is not running');
            return;
        }
        logger_1.logger.info('🛑 Stopping event processing pipeline...');
        this.isRunning = false;
        try {
            await redisStreamsService_1.redisStreamsService.shutdown();
            this.emit('stopped');
            logger_1.logger.info('✅ Event processing pipeline stopped successfully');
        }
        catch (error) {
            logger_1.logger.error('❌ Error stopping event processing pipeline:', error);
            throw error;
        }
    }
    /**
     * Process events from a specific stream
     */
    async processStream(streamName, groupName) {
        logger_1.logger.info(`🔄 Starting event processing for stream: ${streamName} (group: ${groupName})`);
        while (this.isRunning) {
            try {
                // Check circuit breaker
                if (this.isCircuitBreakerOpen()) {
                    await this.sleep(this.config.circuitBreakerTimeout);
                    continue;
                }
                // Read events from stream
                const events = await redisStreamsService_1.redisStreamsService.readEvents(streamName, {
                    groupName,
                    consumerName: this.config.consumerName,
                    blockTime: this.config.blockTime,
                    count: this.config.batchSize,
                });
                if (events.length === 0) {
                    continue;
                }
                logger_1.logger.debug(`📥 Received ${events.length} events from ${streamName}`);
                // Process events in parallel
                const processingPromises = events.map(event => this.processEvent(event, streamName, groupName));
                const results = await Promise.allSettled(processingPromises);
                // Handle results and acknowledgments
                const messageIds = [];
                let successCount = 0;
                let failureCount = 0;
                for (let i = 0; i < results.length; i++) {
                    const result = results[i];
                    const event = events[i];
                    if (result.status === 'fulfilled' && result.value.status === events_2.ProcessingStatus.COMPLETED) {
                        messageIds.push(event.id);
                        successCount++;
                        this.consecutiveFailures = 0;
                    }
                    else {
                        failureCount++;
                        this.consecutiveFailures++;
                        // Handle failed event
                        await this.handleFailedEvent(event, streamName, groupName, result.status === 'rejected' ? result.reason :
                            result.status === 'fulfilled' ? result.value.error : 'Unknown error');
                    }
                }
                // Acknowledge successfully processed events
                if (messageIds.length > 0) {
                    await redisStreamsService_1.redisStreamsService.acknowledgeEvents(streamName, groupName, messageIds);
                }
                // Update stats
                this.stats.processedEvents += successCount;
                this.stats.failedEvents += failureCount;
                this.stats.lastProcessedAt = Date.now();
                // Check circuit breaker
                if (this.config.enableCircuitBreaker &&
                    this.consecutiveFailures >= this.config.circuitBreakerThreshold) {
                    this.openCircuitBreaker();
                }
                logger_1.logger.debug(`✅ Processed ${successCount}/${events.length} events from ${streamName}`);
            }
            catch (error) {
                logger_1.logger.error(`❌ Error processing stream ${streamName}:`, error);
                this.consecutiveFailures++;
                // Back off on error
                await this.sleep(this.config.retryDelay);
            }
        }
        logger_1.logger.info(`🛑 Stopped processing stream: ${streamName}`);
    }
    /**
     * Process a single event
     */
    async processEvent(event, streamName, groupName) {
        const startTime = Date.now();
        try {
            // Find appropriate processor
            const processor = this.findProcessor(event);
            if (!processor) {
                throw new Error(`No processor found for event type: ${event.type}`);
            }
            // Process event with timeout
            const result = await Promise.race([
                processor.process(event),
                this.createTimeoutPromise(this.config.processingTimeout),
            ]);
            const processingTime = Date.now() - startTime;
            this.updateProcessingTime(processingTime);
            // Emit processing result
            this.emit('eventProcessed', {
                event,
                result,
                processingTime,
                processor: processor.getName(),
            });
            return {
                ...result,
                processingTime,
            };
        }
        catch (error) {
            const processingTime = Date.now() - startTime;
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            logger_1.logger.error(`❌ Failed to process event ${event.id}:`, error);
            this.emit('eventFailed', {
                event,
                error: errorMessage,
                processingTime,
            });
            return {
                eventId: event.id,
                status: events_2.ProcessingStatus.FAILED,
                processingTime,
                error: errorMessage,
            };
        }
    }
    /**
     * Find appropriate processor for an event
     */
    findProcessor(event) {
        for (const processor of this.processors.values()) {
            if (processor.canProcess(event)) {
                return processor;
            }
        }
        return null;
    }
    /**
     * Handle failed event processing
     */
    async handleFailedEvent(event, streamName, groupName, error) {
        try {
            // Get retry count from metadata
            const retryCount = (event.metadata?.retryCount || 0) + 1;
            if (retryCount <= this.config.retryAttempts) {
                // Retry the event
                event.metadata = {
                    ...event.metadata,
                    retryCount,
                    errorMessage: error,
                    lastRetryAt: Date.now(),
                };
                // Re-publish to stream for retry
                await redisStreamsService_1.redisStreamsService.publishEvent(streamName, event);
                this.stats.retryEvents++;
                logger_1.logger.warn(`🔄 Retrying event ${event.id} (attempt ${retryCount}/${this.config.retryAttempts})`);
            }
            else {
                // Send to dead letter queue
                await this.sendToDeadLetterQueue(event, error);
                this.stats.deadLetterEvents++;
                logger_1.logger.error(`💀 Sent event ${event.id} to dead letter queue after ${retryCount} attempts`);
            }
        }
        catch (retryError) {
            logger_1.logger.error(`❌ Failed to handle failed event ${event.id}:`, retryError);
        }
    }
    /**
     * Send event to dead letter queue
     */
    async sendToDeadLetterQueue(event, error) {
        const deadLetterEvent = {
            id: `dl-${event.id}`,
            type: 'SYSTEM_ERROR',
            timestamp: Date.now(),
            version: '1.0',
            source: 'event-processing-pipeline',
            correlationId: event.correlationId,
            causationId: event.id,
            data: {
                component: 'event-processing-pipeline',
                status: 'UNHEALTHY',
                message: `Dead letter: ${error}`,
                errorDetails: {
                    originalEvent: event,
                    error,
                    deadLetterReason: 'Max retry attempts exceeded',
                },
            },
        };
        await redisStreamsService_1.redisStreamsService.publishEvent(events_2.STREAM_NAMES.SYSTEM, deadLetterEvent);
    }
    /**
     * Circuit breaker methods
     */
    isCircuitBreakerOpen() {
        if (!this.circuitBreakerOpen) {
            return false;
        }
        // Check if circuit breaker timeout has passed
        if (Date.now() - this.circuitBreakerOpenTime >= this.config.circuitBreakerTimeout) {
            this.closeCircuitBreaker();
            return false;
        }
        return true;
    }
    openCircuitBreaker() {
        this.circuitBreakerOpen = true;
        this.circuitBreakerOpenTime = Date.now();
        this.stats.circuitBreakerOpen = true;
        logger_1.logger.warn('🚨 Circuit breaker opened due to consecutive failures');
        this.emit('circuitBreakerOpened');
    }
    closeCircuitBreaker() {
        this.circuitBreakerOpen = false;
        this.circuitBreakerOpenTime = 0;
        this.consecutiveFailures = 0;
        this.stats.circuitBreakerOpen = false;
        logger_1.logger.info('✅ Circuit breaker closed');
        this.emit('circuitBreakerClosed');
    }
    /**
     * Update processing time statistics
     */
    updateProcessingTime(processingTime) {
        this.processingTimes.push(processingTime);
        // Keep only last 1000 processing times
        if (this.processingTimes.length > 1000) {
            this.processingTimes = this.processingTimes.slice(-1000);
        }
        // Calculate average
        this.stats.averageProcessingTime =
            this.processingTimes.reduce((sum, time) => sum + time, 0) / this.processingTimes.length;
    }
    /**
     * Start stats update loop
     */
    startStatsUpdate() {
        const updateInterval = setInterval(() => {
            if (!this.isRunning) {
                clearInterval(updateInterval);
                return;
            }
            // Calculate throughput (events per second)
            const uptime = (Date.now() - this.startTime) / 1000;
            this.stats.uptime = uptime;
            this.stats.throughput = this.stats.processedEvents / uptime;
            this.emit('statsUpdated', this.stats);
        }, 5000); // Update every 5 seconds
    }
    /**
     * Utility methods
     */
    async sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    createTimeoutPromise(timeout) {
        return new Promise((_, reject) => {
            setTimeout(() => reject(new Error('Processing timeout')), timeout);
        });
    }
    /**
     * Get pipeline statistics
     */
    getStats() {
        return { ...this.stats };
    }
    /**
     * Get pipeline configuration
     */
    getConfig() {
        return { ...this.config };
    }
    /**
     * Health check
     */
    async healthCheck() {
        try {
            return this.isRunning && await redisStreamsService_1.redisStreamsService.healthCheck();
        }
        catch (error) {
            return false;
        }
    }
}
exports.EventProcessingPipeline = EventProcessingPipeline;
// Export singleton instance
exports.eventProcessingPipeline = new EventProcessingPipeline();
//# sourceMappingURL=eventProcessingPipeline.js.map