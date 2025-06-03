"use strict";
/**
 * Event-Driven Trading System
 * Main orchestrator for event-driven trading architecture
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.eventDrivenTradingSystem = exports.EventDrivenTradingSystem = void 0;
const events_1 = require("events");
const redisStreamsService_1 = require("./redisStreamsService");
const eventProcessingPipeline_1 = require("./eventProcessingPipeline");
const marketDataProcessor_1 = require("../processors/marketDataProcessor");
const signalProcessor_1 = require("../processors/signalProcessor");
const logger_1 = require("../utils/logger");
const events_2 = require("../types/events");
class EventDrivenTradingSystem extends events_1.EventEmitter {
    constructor(config = {}) {
        super();
        this.isRunning = false;
        this.startTime = 0;
        this.processors = new Map();
        this.config = {
            enableMarketDataProcessing: true,
            enableSignalProcessing: true,
            enableOrderProcessing: true,
            enableRiskManagement: true,
            enablePortfolioManagement: true,
            enableSystemMonitoring: true,
            marketDataSources: ['delta-exchange', 'binance'],
            signalGenerationModels: ['transformer-v1', 'price-change-detector'],
            riskManagementRules: ['position-size', 'daily-loss', 'drawdown'],
            ...config,
        };
        this.stats = {
            uptime: 0,
            eventsProcessed: 0,
            eventsPerSecond: 0,
            activeStreams: 0,
            activeProcessors: 0,
            systemHealth: 'HEALTHY',
            lastEventTime: 0,
            errors: 0,
        };
    }
    static getInstance(config) {
        if (!EventDrivenTradingSystem.instance) {
            EventDrivenTradingSystem.instance = new EventDrivenTradingSystem(config);
        }
        return EventDrivenTradingSystem.instance;
    }
    /**
     * Initialize and start the event-driven trading system
     */
    async start() {
        if (this.isRunning) {
            logger_1.logger.warn('⚠️ Event-driven trading system is already running');
            return;
        }
        try {
            logger_1.logger.info('🚀 Starting event-driven trading system...');
            // Initialize Redis Streams service
            await redisStreamsService_1.redisStreamsService.initialize();
            // Register event processors
            await this.registerProcessors();
            // Start event processing pipeline
            await eventProcessingPipeline_1.eventProcessingPipeline.start();
            // Set up event listeners
            this.setupEventListeners();
            // Start system monitoring
            this.startSystemMonitoring();
            this.isRunning = true;
            this.startTime = Date.now();
            // Publish system started event
            await this.publishSystemEvent('SYSTEM_STARTED', {
                component: 'event-driven-trading-system',
                status: 'HEALTHY',
                message: 'Event-driven trading system started successfully',
                uptime: 0,
            });
            this.emit('started');
            logger_1.logger.info('✅ Event-driven trading system started successfully');
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to start event-driven trading system:', error);
            this.isRunning = false;
            throw error;
        }
    }
    /**
     * Stop the event-driven trading system
     */
    async stop() {
        if (!this.isRunning) {
            logger_1.logger.warn('⚠️ Event-driven trading system is not running');
            return;
        }
        try {
            logger_1.logger.info('🛑 Stopping event-driven trading system...');
            // Publish system stopping event
            await this.publishSystemEvent('SYSTEM_STOPPED', {
                component: 'event-driven-trading-system',
                status: 'DOWN',
                message: 'Event-driven trading system stopping',
                uptime: Date.now() - this.startTime,
            });
            // Stop event processing pipeline
            await eventProcessingPipeline_1.eventProcessingPipeline.stop();
            // Shutdown Redis Streams service
            await redisStreamsService_1.redisStreamsService.shutdown();
            this.isRunning = false;
            this.emit('stopped');
            logger_1.logger.info('✅ Event-driven trading system stopped successfully');
        }
        catch (error) {
            logger_1.logger.error('❌ Error stopping event-driven trading system:', error);
            throw error;
        }
    }
    /**
     * Register event processors
     */
    async registerProcessors() {
        try {
            // Market Data Processor
            if (this.config.enableMarketDataProcessing) {
                const marketDataProcessor = new marketDataProcessor_1.MarketDataProcessor();
                eventProcessingPipeline_1.eventProcessingPipeline.registerProcessor(marketDataProcessor);
                this.processors.set('marketData', marketDataProcessor);
                logger_1.logger.info('📊 Registered market data processor');
            }
            // Signal Processor
            if (this.config.enableSignalProcessing) {
                const signalProcessor = new signalProcessor_1.SignalProcessor();
                eventProcessingPipeline_1.eventProcessingPipeline.registerProcessor(signalProcessor);
                this.processors.set('signal', signalProcessor);
                logger_1.logger.info('🎯 Registered signal processor');
            }
            // TODO: Add more processors as needed
            // - Order Processor
            // - Risk Management Processor
            // - Portfolio Management Processor
            // - System Monitoring Processor
            this.stats.activeProcessors = this.processors.size;
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to register processors:', error);
            throw error;
        }
    }
    /**
     * Set up event listeners for monitoring
     */
    setupEventListeners() {
        // Listen to pipeline events
        eventProcessingPipeline_1.eventProcessingPipeline.on('eventProcessed', (data) => {
            this.stats.eventsProcessed++;
            this.stats.lastEventTime = Date.now();
            this.emit('eventProcessed', data);
        });
        eventProcessingPipeline_1.eventProcessingPipeline.on('eventFailed', (data) => {
            this.stats.errors++;
            this.emit('eventFailed', data);
            logger_1.logger.error(`❌ Event processing failed:`, data);
        });
        eventProcessingPipeline_1.eventProcessingPipeline.on('circuitBreakerOpened', () => {
            this.stats.systemHealth = 'DEGRADED';
            this.emit('circuitBreakerOpened');
            logger_1.logger.warn('🚨 Circuit breaker opened - system degraded');
        });
        eventProcessingPipeline_1.eventProcessingPipeline.on('circuitBreakerClosed', () => {
            this.stats.systemHealth = 'HEALTHY';
            this.emit('circuitBreakerClosed');
            logger_1.logger.info('✅ Circuit breaker closed - system healthy');
        });
        eventProcessingPipeline_1.eventProcessingPipeline.on('statsUpdated', (pipelineStats) => {
            this.updateSystemStats(pipelineStats);
        });
    }
    /**
     * Start system monitoring
     */
    startSystemMonitoring() {
        const monitoringInterval = setInterval(async () => {
            if (!this.isRunning) {
                clearInterval(monitoringInterval);
                return;
            }
            try {
                // Update uptime
                this.stats.uptime = Date.now() - this.startTime;
                // Calculate events per second
                this.stats.eventsPerSecond = this.stats.eventsProcessed / (this.stats.uptime / 1000);
                // Health check
                const isHealthy = await this.performHealthCheck();
                if (!isHealthy && this.stats.systemHealth === 'HEALTHY') {
                    this.stats.systemHealth = 'UNHEALTHY';
                    await this.publishSystemEvent('SYSTEM_ALERT', {
                        component: 'event-driven-trading-system',
                        status: 'UNHEALTHY',
                        message: 'System health check failed',
                        uptime: this.stats.uptime,
                    });
                }
                // Emit stats update
                this.emit('statsUpdated', this.stats);
            }
            catch (error) {
                logger_1.logger.error('❌ System monitoring error:', error);
            }
        }, 10000); // Every 10 seconds
    }
    /**
     * Perform system health check
     */
    async performHealthCheck() {
        try {
            // Check Redis Streams service
            const redisHealthy = await redisStreamsService_1.redisStreamsService.healthCheck();
            // Check event processing pipeline
            const pipelineHealthy = await eventProcessingPipeline_1.eventProcessingPipeline.healthCheck();
            return redisHealthy && pipelineHealthy;
        }
        catch (error) {
            logger_1.logger.error('❌ Health check failed:', error);
            return false;
        }
    }
    /**
     * Update system statistics
     */
    updateSystemStats(pipelineStats) {
        // Update stats based on pipeline statistics
        this.stats.eventsProcessed = pipelineStats.processedEvents;
        this.stats.errors = pipelineStats.failedEvents;
        // Determine system health
        if (pipelineStats.circuitBreakerOpen) {
            this.stats.systemHealth = 'DEGRADED';
        }
        else if (this.stats.errors > 100) {
            this.stats.systemHealth = 'UNHEALTHY';
        }
        else {
            this.stats.systemHealth = 'HEALTHY';
        }
    }
    // ============================================================================
    // EVENT PUBLISHING METHODS
    // ============================================================================
    /**
     * Publish market data event
     */
    async publishMarketDataEvent(symbol, price, volume, exchange = 'default', additionalData = {}) {
        const event = {
            id: (0, events_2.createEventId)(),
            type: 'MARKET_DATA_RECEIVED',
            timestamp: Date.now(),
            version: '1.0',
            source: 'trading-system',
            correlationId: (0, events_2.createCorrelationId)(),
            data: {
                symbol,
                exchange,
                price,
                volume,
                timestamp: Date.now(),
                ...additionalData,
            },
        };
        return await redisStreamsService_1.redisStreamsService.publishEvent(events_2.STREAM_NAMES.MARKET_DATA, event);
    }
    /**
     * Publish trading signal event
     */
    async publishTradingSignalEvent(signalData, correlationId) {
        const event = {
            id: (0, events_2.createEventId)(),
            type: 'SIGNAL_GENERATED',
            timestamp: Date.now(),
            version: '1.0',
            source: 'trading-system',
            correlationId: correlationId || (0, events_2.createCorrelationId)(),
            data: {
                signalId: (0, events_2.createEventId)(),
                symbol: '',
                signalType: 'ENTRY',
                direction: 'LONG',
                strength: 'MODERATE',
                timeframe: '1m',
                price: 0,
                confidenceScore: 0,
                expectedReturn: 0,
                expectedRisk: 0,
                riskRewardRatio: 1,
                modelSource: 'unknown',
                ...signalData,
            },
        };
        return await redisStreamsService_1.redisStreamsService.publishEvent(events_2.STREAM_NAMES.TRADING_SIGNALS, event);
    }
    /**
     * Publish system event
     */
    async publishSystemEvent(type, data) {
        const event = {
            id: (0, events_2.createEventId)(),
            type,
            timestamp: Date.now(),
            version: '1.0',
            source: 'trading-system',
            data: {
                component: 'unknown',
                status: 'HEALTHY',
                message: '',
                ...data,
            },
        };
        return await redisStreamsService_1.redisStreamsService.publishEvent(events_2.STREAM_NAMES.SYSTEM, event);
    }
    /**
     * Publish bot event
     */
    async publishBotEvent(type, botData, userId) {
        const event = {
            id: (0, events_2.createEventId)(),
            type,
            timestamp: Date.now(),
            version: '1.0',
            source: 'trading-system',
            userId,
            data: {
                botId: '',
                botName: '',
                status: 'STOPPED',
                symbol: '',
                strategy: '',
                timeframe: '',
                ...botData,
            },
        };
        return await redisStreamsService_1.redisStreamsService.publishEvent(events_2.STREAM_NAMES.BOTS, event);
    }
    // ============================================================================
    // UTILITY METHODS
    // ============================================================================
    /**
     * Get system statistics
     */
    getStats() {
        return { ...this.stats };
    }
    /**
     * Get system configuration
     */
    getConfig() {
        return { ...this.config };
    }
    /**
     * Update system configuration
     */
    updateConfig(newConfig) {
        this.config = { ...this.config, ...newConfig };
        logger_1.logger.info('⚙️ System configuration updated');
    }
    /**
     * Get processor statistics
     */
    getProcessorStats() {
        const stats = {};
        for (const [name, processor] of this.processors) {
            if (typeof processor.getStats === 'function') {
                stats[name] = processor.getStats();
            }
        }
        return stats;
    }
    /**
     * Check if system is running
     */
    isSystemRunning() {
        return this.isRunning;
    }
    /**
     * Get system health status
     */
    getHealthStatus() {
        return this.stats.systemHealth;
    }
}
exports.EventDrivenTradingSystem = EventDrivenTradingSystem;
// Export singleton instance
exports.eventDrivenTradingSystem = EventDrivenTradingSystem.getInstance();
//# sourceMappingURL=eventDrivenTradingSystem.js.map