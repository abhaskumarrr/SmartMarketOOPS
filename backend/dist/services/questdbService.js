"use strict";
/**
 * QuestDB Service
 * High-performance time-series database service for SmartMarketOOPS
 * Provides optimized operations for financial time-series data
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.questdbService = exports.QuestDBService = void 0;
const questdb_1 = require("../config/questdb");
const logger_1 = require("../utils/logger");
class QuestDBService {
    constructor() {
        this.client = null;
    }
    static getInstance() {
        if (!QuestDBService.instance) {
            QuestDBService.instance = new QuestDBService();
        }
        return QuestDBService.instance;
    }
    async initialize() {
        try {
            await questdb_1.questdbConnection.connect();
            this.client = questdb_1.questdbConnection.getClient();
            logger_1.logger.info('✅ QuestDB service initialized successfully');
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to initialize QuestDB service:', error);
            throw error;
        }
    }
    async shutdown() {
        try {
            await questdb_1.questdbConnection.disconnect();
            this.client = null;
            logger_1.logger.info('🔌 QuestDB service shutdown completed');
        }
        catch (error) {
            logger_1.logger.error('❌ Error during QuestDB service shutdown:', error);
            throw error;
        }
    }
    ensureClient() {
        if (!this.client || !questdb_1.questdbConnection.isReady()) {
            throw new Error('QuestDB service not initialized. Call initialize() first.');
        }
        return this.client;
    }
    // Metric operations
    async insertMetric(data) {
        const client = this.ensureClient();
        try {
            // Use the Sender API to build the line protocol
            client
                .table('metrics')
                .symbol('name', data.name);
            // Add tags as symbols
            if (data.tags) {
                for (const [key, value] of Object.entries(data.tags)) {
                    client.symbol(key, String(value));
                }
            }
            // Add value as float column
            client.floatColumn('value', data.value);
            // Set timestamp and send (don't await here, let auto-flush handle it)
            if (data.timestamp instanceof Date) {
                client.at(data.timestamp.getTime() * 1000000); // Convert to nanoseconds
            }
            else {
                client.atNow();
            }
        }
        catch (error) {
            logger_1.logger.error('Error inserting metric:', error);
            throw error;
        }
    }
    async insertMetrics(metrics) {
        const client = this.ensureClient();
        try {
            for (const metric of metrics) {
                await this.insertMetric(metric);
            }
            await client.flush();
        }
        catch (error) {
            logger_1.logger.error('Error inserting metrics batch:', error);
            throw error;
        }
    }
    // Trading Signal operations
    async insertTradingSignal(data) {
        const client = this.ensureClient();
        try {
            client
                .table('trading_signals')
                .symbol('id', data.id)
                .symbol('symbol', data.symbol)
                .symbol('type', data.type)
                .symbol('direction', data.direction)
                .symbol('strength', data.strength)
                .symbol('timeframe', data.timeframe)
                .symbol('source', data.source)
                .floatColumn('price', data.price)
                .floatColumn('confidence_score', data.confidenceScore)
                .floatColumn('expected_return', data.expectedReturn)
                .floatColumn('expected_risk', data.expectedRisk)
                .floatColumn('risk_reward_ratio', data.riskRewardRatio);
            if (data.targetPrice !== undefined) {
                client.floatColumn('target_price', data.targetPrice);
            }
            if (data.stopLoss !== undefined) {
                client.floatColumn('stop_loss', data.stopLoss);
            }
            if (data.timestamp instanceof Date) {
                client.at(data.timestamp.getTime() * 1000000);
            }
            else {
                client.atNow();
            }
        }
        catch (error) {
            logger_1.logger.error('Error inserting trading signal:', error);
            throw error;
        }
    }
    // ML Prediction operations
    async insertMLPrediction(data) {
        const client = this.ensureClient();
        try {
            // Convert arrays to JSON strings for storage
            const valuesJson = JSON.stringify(data.values);
            const confidenceJson = JSON.stringify(data.confidenceScores);
            client
                .table('ml_predictions')
                .symbol('id', data.id)
                .symbol('model_id', data.modelId)
                .symbol('symbol', data.symbol)
                .symbol('timeframe', data.timeframe)
                .symbol('prediction_type', data.predictionType)
                .stringColumn('values', valuesJson)
                .stringColumn('confidence_scores', confidenceJson);
            if (data.timestamp instanceof Date) {
                client.at(data.timestamp.getTime() * 1000000);
            }
            else {
                client.atNow();
            }
        }
        catch (error) {
            logger_1.logger.error('Error inserting ML prediction:', error);
            throw error;
        }
    }
    // Performance Metric operations
    async insertPerformanceMetric(data) {
        const client = this.ensureClient();
        try {
            client
                .table('performance_metrics')
                .symbol('system', data.system)
                .symbol('component', data.component)
                .symbol('metric', data.metric)
                .symbol('unit', data.unit);
            // Add tags as symbols
            if (data.tags) {
                for (const [key, value] of Object.entries(data.tags)) {
                    client.symbol(key, String(value));
                }
            }
            client.floatColumn('value', data.value);
            if (data.timestamp instanceof Date) {
                client.at(data.timestamp.getTime() * 1000000);
            }
            else {
                client.atNow();
            }
        }
        catch (error) {
            logger_1.logger.error('Error inserting performance metric:', error);
            throw error;
        }
    }
    // Market Data operations
    async insertMarketData(data) {
        const client = this.ensureClient();
        try {
            client
                .table('market_data')
                .symbol('symbol', data.symbol)
                .symbol('exchange', data.exchange)
                .symbol('timeframe', data.timeframe)
                .floatColumn('open', data.open)
                .floatColumn('high', data.high)
                .floatColumn('low', data.low)
                .floatColumn('close', data.close)
                .floatColumn('volume', data.volume);
            client.at(data.timestamp.getTime() * 1000000);
        }
        catch (error) {
            logger_1.logger.error('Error inserting market data:', error);
            throw error;
        }
    }
    async insertMarketDataBatch(dataPoints) {
        const client = this.ensureClient();
        try {
            for (const data of dataPoints) {
                client
                    .table('market_data')
                    .symbol('symbol', data.symbol)
                    .symbol('exchange', data.exchange)
                    .symbol('timeframe', data.timeframe)
                    .floatColumn('open', data.open)
                    .floatColumn('high', data.high)
                    .floatColumn('low', data.low)
                    .floatColumn('close', data.close)
                    .floatColumn('volume', data.volume);
                client.at(data.timestamp.getTime() * 1000000);
            }
            await client.flush();
        }
        catch (error) {
            logger_1.logger.error('Error inserting market data batch:', error);
            throw error;
        }
    }
    // Trade operations
    async insertTrade(data) {
        const client = this.ensureClient();
        try {
            client
                .table('trades')
                .symbol('id', data.id)
                .symbol('symbol', data.symbol)
                .symbol('side', data.side)
                .symbol('strategy', data.strategy)
                .symbol('reason', data.reason)
                .floatColumn('entry_price', data.entryPrice)
                .floatColumn('exit_price', data.exitPrice)
                .floatColumn('quantity', data.quantity)
                .floatColumn('pnl', data.pnl)
                .floatColumn('pnl_percent', data.pnlPercent)
                .floatColumn('commission', data.commission)
                .floatColumn('duration', data.duration)
                .timestampColumn('entry_time', data.entryTime.getTime() * 1000000)
                .timestampColumn('exit_time', data.exitTime.getTime() * 1000000);
            client.at(data.timestamp.getTime() * 1000000);
        }
        catch (error) {
            logger_1.logger.error('Error inserting trade:', error);
            throw error;
        }
    }
    // Portfolio operations
    async insertPortfolioSnapshot(data) {
        const client = this.ensureClient();
        try {
            client
                .table('portfolio_snapshots')
                .floatColumn('total_value', data.totalValue)
                .floatColumn('cash', data.cash)
                .floatColumn('total_pnl', data.totalPnl)
                .floatColumn('total_pnl_percent', data.totalPnlPercent)
                .floatColumn('drawdown', data.drawdown)
                .floatColumn('max_drawdown', data.maxDrawdown)
                .floatColumn('leverage', data.leverage)
                .floatColumn('position_count', data.positionCount);
            client.at(data.timestamp.getTime() * 1000000);
        }
        catch (error) {
            logger_1.logger.error('Error inserting portfolio snapshot:', error);
            throw error;
        }
    }
    // Batch operations for high-performance ingestion
    async batchInsert(tableName, data, formatter) {
        const client = this.ensureClient();
        try {
            // For batch operations, we'll use individual insertions
            // The formatter function is not used with the new Sender API
            for (const item of data) {
                // This is a generic method, so we'll handle it based on the item structure
                if ('name' in item && 'value' in item) {
                    // Treat as metric
                    await this.insertMetric(item);
                }
            }
            await client.flush();
        }
        catch (error) {
            logger_1.logger.error(`Error in batch insert for ${tableName}:`, error);
            throw error;
        }
    }
    // Query operations (using HTTP API for complex queries)
    async executeQuery(query) {
        try {
            // For complex queries, we'll use HTTP API
            const response = await fetch(`http://${questdb_1.questdbConnection.getConfig().host}:9000/exec?query=${encodeURIComponent(query)}`, {
                method: 'GET',
                headers: {
                    'Accept': 'application/json',
                },
            });
            if (!response.ok) {
                throw new Error(`Query failed: ${response.statusText}`);
            }
            const result = await response.json();
            return result.dataset || [];
        }
        catch (error) {
            logger_1.logger.error('Error executing query:', error);
            throw error;
        }
    }
    // Optimized time-series queries
    async getMetricsByTimeRange(metricName, startTime, endTime, options) {
        const startTs = startTime.toISOString();
        const endTs = endTime.toISOString();
        let query = `SELECT * FROM metrics WHERE name = '${metricName}' AND timestamp >= '${startTs}' AND timestamp <= '${endTs}'`;
        if (options?.orderBy) {
            query += ` ORDER BY ${options.orderBy} ${options.orderDirection || 'ASC'}`;
        }
        else {
            query += ` ORDER BY timestamp ASC`;
        }
        if (options?.limit) {
            query += ` LIMIT ${options.limit}`;
        }
        return this.executeQuery(query);
    }
    async getTradingSignalsBySymbol(symbol, startTime, endTime, options) {
        const startTs = startTime.toISOString();
        const endTs = endTime.toISOString();
        let query = `SELECT * FROM trading_signals WHERE symbol = '${symbol}' AND timestamp >= '${startTs}' AND timestamp <= '${endTs}'`;
        if (options?.orderBy) {
            query += ` ORDER BY ${options.orderBy} ${options.orderDirection || 'ASC'}`;
        }
        else {
            query += ` ORDER BY timestamp DESC`;
        }
        if (options?.limit) {
            query += ` LIMIT ${options.limit}`;
        }
        return this.executeQuery(query);
    }
    async getLatestMetrics(metricNames, limit = 100) {
        const namesStr = metricNames.map(name => `'${name}'`).join(',');
        const query = `SELECT * FROM metrics WHERE name IN (${namesStr}) ORDER BY timestamp DESC LIMIT ${limit}`;
        return this.executeQuery(query);
    }
    // Aggregation queries for analytics
    async getMetricAggregation(metricName, aggregation, interval, startTime, endTime) {
        const startTs = startTime.toISOString();
        const endTs = endTime.toISOString();
        const query = `
      SELECT 
        timestamp,
        ${aggregation}(value) as ${aggregation.toLowerCase()}_value
      FROM metrics 
      WHERE name = '${metricName}' 
        AND timestamp >= '${startTs}' 
        AND timestamp <= '${endTs}'
      SAMPLE BY ${interval}
      ORDER BY timestamp ASC
    `;
        return this.executeQuery(query);
    }
    // Health check and statistics
    async healthCheck() {
        try {
            await questdb_1.questdbConnection.healthCheck();
            return true;
        }
        catch (error) {
            logger_1.logger.error('QuestDB service health check failed:', error);
            return false;
        }
    }
    async getTableStats(tableName) {
        const query = `SELECT count() as row_count FROM ${tableName}`;
        const result = await this.executeQuery(query);
        return result[0] || { row_count: 0 };
    }
    // Utility methods
    async flush() {
        const client = this.ensureClient();
        await client.flush();
    }
    isReady() {
        return questdb_1.questdbConnection.isReady() && this.client !== null;
    }
}
exports.QuestDBService = QuestDBService;
// Export singleton instance
exports.questdbService = QuestDBService.getInstance();
//# sourceMappingURL=questdbService.js.map