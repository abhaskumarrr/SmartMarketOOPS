"use strict";
/**
 * Mock QuestDB Service for Testing
 * Provides a mock implementation for testing without actual QuestDB
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.mockQuestdbService = exports.MockQuestDBService = void 0;
const logger_1 = require("../utils/logger");
class MockQuestDBService {
    constructor() {
        this.isInitialized = false;
        this.data = new Map();
        // Initialize mock data storage
        this.data.set('metrics', []);
        this.data.set('trading_signals', []);
        this.data.set('ml_predictions', []);
        this.data.set('performance_metrics', []);
    }
    static getInstance() {
        if (!MockQuestDBService.instance) {
            MockQuestDBService.instance = new MockQuestDBService();
        }
        return MockQuestDBService.instance;
    }
    async initialize() {
        try {
            logger_1.logger.info('🔌 Initializing Mock QuestDB service...');
            this.isInitialized = true;
            logger_1.logger.info('✅ Mock QuestDB service initialized successfully');
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to initialize Mock QuestDB service:', error);
            throw error;
        }
    }
    async shutdown() {
        try {
            this.isInitialized = false;
            this.data.clear();
            logger_1.logger.info('🔌 Mock QuestDB service shutdown completed');
        }
        catch (error) {
            logger_1.logger.error('❌ Error during Mock QuestDB service shutdown:', error);
            throw error;
        }
    }
    // Metric operations
    async insertMetric(data) {
        if (!this.isInitialized) {
            throw new Error('Mock QuestDB service not initialized');
        }
        try {
            const metrics = this.data.get('metrics') || [];
            metrics.push({
                ...data,
                timestamp: data.timestamp instanceof Date ? data.timestamp : new Date(data.timestamp),
                inserted_at: new Date(),
            });
            this.data.set('metrics', metrics);
            logger_1.logger.debug(`📊 Mock: Inserted metric ${data.name} = ${data.value}`);
        }
        catch (error) {
            logger_1.logger.error('Error inserting metric:', error);
            throw error;
        }
    }
    async insertMetrics(metrics) {
        for (const metric of metrics) {
            await this.insertMetric(metric);
        }
        logger_1.logger.debug(`📊 Mock: Inserted ${metrics.length} metrics`);
    }
    // Trading Signal operations
    async insertTradingSignal(data) {
        if (!this.isInitialized) {
            throw new Error('Mock QuestDB service not initialized');
        }
        try {
            const signals = this.data.get('trading_signals') || [];
            signals.push({
                ...data,
                timestamp: data.timestamp instanceof Date ? data.timestamp : new Date(data.timestamp),
                inserted_at: new Date(),
            });
            this.data.set('trading_signals', signals);
            logger_1.logger.debug(`🎯 Mock: Inserted trading signal ${data.id} for ${data.symbol}`);
        }
        catch (error) {
            logger_1.logger.error('Error inserting trading signal:', error);
            throw error;
        }
    }
    // ML Prediction operations
    async insertMLPrediction(data) {
        if (!this.isInitialized) {
            throw new Error('Mock QuestDB service not initialized');
        }
        try {
            const predictions = this.data.get('ml_predictions') || [];
            predictions.push({
                ...data,
                timestamp: data.timestamp instanceof Date ? data.timestamp : new Date(data.timestamp),
                inserted_at: new Date(),
            });
            this.data.set('ml_predictions', predictions);
            logger_1.logger.debug(`🤖 Mock: Inserted ML prediction ${data.id} for ${data.symbol}`);
        }
        catch (error) {
            logger_1.logger.error('Error inserting ML prediction:', error);
            throw error;
        }
    }
    // Performance Metric operations
    async insertPerformanceMetric(data) {
        if (!this.isInitialized) {
            throw new Error('Mock QuestDB service not initialized');
        }
        try {
            const metrics = this.data.get('performance_metrics') || [];
            metrics.push({
                ...data,
                timestamp: data.timestamp instanceof Date ? data.timestamp : new Date(data.timestamp),
                inserted_at: new Date(),
            });
            this.data.set('performance_metrics', metrics);
            logger_1.logger.debug(`⚡ Mock: Inserted performance metric ${data.metric} = ${data.value}`);
        }
        catch (error) {
            logger_1.logger.error('Error inserting performance metric:', error);
            throw error;
        }
    }
    // Query operations
    async executeQuery(query) {
        logger_1.logger.debug(`🔍 Mock: Executing query: ${query}`);
        // Simple mock query responses
        if (query.includes('count()')) {
            return [{ count: this.getTotalRecords() }];
        }
        if (query.includes('metrics')) {
            return this.data.get('metrics') || [];
        }
        if (query.includes('trading_signals')) {
            return this.data.get('trading_signals') || [];
        }
        if (query.includes('ml_predictions')) {
            return this.data.get('ml_predictions') || [];
        }
        if (query.includes('performance_metrics')) {
            return this.data.get('performance_metrics') || [];
        }
        return [];
    }
    // Health check and statistics
    async healthCheck() {
        return this.isInitialized;
    }
    async getTableStats(tableName) {
        const data = this.data.get(tableName) || [];
        return { row_count: data.length };
    }
    async flush() {
        logger_1.logger.debug('🔄 Mock: Flush called (no-op)');
    }
    isReady() {
        return this.isInitialized;
    }
    // Mock-specific methods for testing
    getStoredData(tableName) {
        return this.data.get(tableName) || [];
    }
    getTotalRecords() {
        let total = 0;
        for (const [, records] of this.data) {
            total += records.length;
        }
        return total;
    }
    clearData() {
        this.data.clear();
        this.data.set('metrics', []);
        this.data.set('trading_signals', []);
        this.data.set('ml_predictions', []);
        this.data.set('performance_metrics', []);
        logger_1.logger.info('🗑️ Mock: Cleared all data');
    }
    getStats() {
        const tableStats = {};
        for (const [tableName, records] of this.data) {
            tableStats[tableName] = records.length;
        }
        return {
            isInitialized: this.isInitialized,
            totalRecords: this.getTotalRecords(),
            tableStats,
        };
    }
}
exports.MockQuestDBService = MockQuestDBService;
// Export singleton instance
exports.mockQuestdbService = MockQuestDBService.getInstance();
//# sourceMappingURL=mockQuestdbService.js.map