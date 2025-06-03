"use strict";
/**
 * QuestDB Migration Pipeline
 * Migrates time-series data from PostgreSQL to QuestDB for performance optimization
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.QuestDBMigrationPipeline = void 0;
const prisma_1 = require("../../generated/prisma");
const questdbService_1 = require("../services/questdbService");
const logger_1 = require("../utils/logger");
const data_validator_1 = require("./data-validator");
class QuestDBMigrationPipeline {
    constructor() {
        this.migrationProgress = new Map();
        this.prisma = new prisma_1.PrismaClient();
        this.validator = new data_validator_1.DataValidator();
    }
    /**
     * Initialize the migration pipeline
     */
    async initialize() {
        try {
            await this.prisma.$connect();
            await questdbService_1.questdbService.initialize();
            logger_1.logger.info('✅ Migration pipeline initialized successfully');
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to initialize migration pipeline:', error);
            throw error;
        }
    }
    /**
     * Shutdown the migration pipeline
     */
    async shutdown() {
        try {
            await this.prisma.$disconnect();
            await questdbService_1.questdbService.shutdown();
            logger_1.logger.info('🔌 Migration pipeline shutdown completed');
        }
        catch (error) {
            logger_1.logger.error('❌ Error during migration pipeline shutdown:', error);
            throw error;
        }
    }
    /**
     * Migrate all time-series tables
     */
    async migrateAll(config = {}) {
        const defaultConfig = {
            sourceTable: '',
            targetTable: '',
            batchSize: 1000,
            parallelWorkers: 2,
            validateData: true,
            dryRun: false,
            ...config,
        };
        const tables = [
            { source: 'Metric', target: 'metrics', migrator: 'migrateMetrics' },
            { source: 'TradingSignal', target: 'trading_signals', migrator: 'migrateTradingSignals' },
            { source: 'MLPrediction', target: 'ml_predictions', migrator: 'migrateMLPredictions' },
            { source: 'PerformanceMetric', target: 'performance_metrics', migrator: 'migratePerformanceMetrics' },
        ];
        const results = [];
        for (const table of tables) {
            try {
                logger_1.logger.info(`🚀 Starting migration for ${table.source} -> ${table.target}`);
                const tableConfig = {
                    ...defaultConfig,
                    sourceTable: table.source,
                    targetTable: table.target,
                };
                const result = await this[table.migrator](tableConfig);
                results.push(result);
                logger_1.logger.info(`✅ Completed migration for ${table.source}: ${result.migratedRecords} records`);
            }
            catch (error) {
                logger_1.logger.error(`❌ Failed to migrate ${table.source}:`, error);
                results.push({
                    success: false,
                    totalRecords: 0,
                    migratedRecords: 0,
                    failedRecords: 0,
                    duration: 0,
                    averageThroughput: 0,
                    errors: [{ record: null, error: error instanceof Error ? error.message : 'Unknown error' }],
                });
            }
        }
        return results;
    }
    /**
     * Migrate Metrics table
     */
    async migrateMetrics(config) {
        const startTime = Date.now();
        let totalRecords = 0;
        let migratedRecords = 0;
        let failedRecords = 0;
        const errors = [];
        try {
            // Get total count
            totalRecords = await this.prisma.metric.count({
                where: this.buildDateFilter(config),
            });
            logger_1.logger.info(`📊 Migrating ${totalRecords} metrics records`);
            if (config.dryRun) {
                logger_1.logger.info('🔍 Dry run mode - no data will be migrated');
                return {
                    success: true,
                    totalRecords,
                    migratedRecords: 0,
                    failedRecords: 0,
                    duration: Date.now() - startTime,
                    averageThroughput: 0,
                    errors: [],
                };
            }
            // Process in batches
            let offset = 0;
            while (offset < totalRecords) {
                const batch = await this.prisma.metric.findMany({
                    where: this.buildDateFilter(config),
                    skip: offset,
                    take: config.batchSize,
                    orderBy: { recordedAt: 'asc' },
                });
                if (batch.length === 0)
                    break;
                // Convert to QuestDB format
                const questdbMetrics = batch.map(metric => ({
                    timestamp: metric.recordedAt,
                    name: metric.name,
                    value: metric.value,
                    tags: metric.tags || {},
                }));
                // Validate data if enabled
                if (config.validateData) {
                    for (const metric of questdbMetrics) {
                        const validation = this.validator.validateMetric(metric);
                        if (!validation.valid) {
                            errors.push({ record: metric, error: validation.errors.join(', ') });
                            failedRecords++;
                            continue;
                        }
                    }
                }
                // Insert into QuestDB
                try {
                    await questdbService_1.questdbService.insertMetrics(questdbMetrics);
                    migratedRecords += questdbMetrics.length;
                }
                catch (error) {
                    logger_1.logger.error(`Error inserting metrics batch at offset ${offset}:`, error);
                    failedRecords += questdbMetrics.length;
                    errors.push({ record: batch, error: error instanceof Error ? error.message : 'Batch insert failed' });
                }
                offset += config.batchSize;
                // Update progress
                this.updateProgress('metrics', {
                    totalRecords,
                    processedRecords: offset,
                    successfulRecords: migratedRecords,
                    failedRecords,
                    progress: (offset / totalRecords) * 100,
                    estimatedTimeRemaining: this.calculateETA(startTime, offset, totalRecords),
                    throughput: migratedRecords / ((Date.now() - startTime) / 1000),
                    errors: errors.slice(-10).map(e => ({ ...e, timestamp: new Date() })), // Keep only last 10 errors
                });
                // Log progress
                if (offset % (config.batchSize * 10) === 0) {
                    logger_1.logger.info(`📈 Metrics migration progress: ${offset}/${totalRecords} (${((offset / totalRecords) * 100).toFixed(1)}%)`);
                }
            }
            const duration = Date.now() - startTime;
            const averageThroughput = migratedRecords / (duration / 1000);
            logger_1.logger.info(`✅ Metrics migration completed: ${migratedRecords}/${totalRecords} records in ${duration}ms (${averageThroughput.toFixed(2)} records/sec)`);
            return {
                success: failedRecords === 0,
                totalRecords,
                migratedRecords,
                failedRecords,
                duration,
                averageThroughput,
                errors,
            };
        }
        catch (error) {
            logger_1.logger.error('❌ Metrics migration failed:', error);
            throw error;
        }
    }
    /**
     * Migrate TradingSignals table
     */
    async migrateTradingSignals(config) {
        const startTime = Date.now();
        let totalRecords = 0;
        let migratedRecords = 0;
        let failedRecords = 0;
        const errors = [];
        try {
            // Get total count
            totalRecords = await this.prisma.tradingSignal.count({
                where: this.buildDateFilter(config, 'generatedAt'),
            });
            logger_1.logger.info(`📊 Migrating ${totalRecords} trading signals records`);
            if (config.dryRun) {
                return {
                    success: true,
                    totalRecords,
                    migratedRecords: 0,
                    failedRecords: 0,
                    duration: Date.now() - startTime,
                    averageThroughput: 0,
                    errors: [],
                };
            }
            // Process in batches
            let offset = 0;
            while (offset < totalRecords) {
                const batch = await this.prisma.tradingSignal.findMany({
                    where: this.buildDateFilter(config, 'generatedAt'),
                    skip: offset,
                    take: config.batchSize,
                    orderBy: { generatedAt: 'asc' },
                });
                if (batch.length === 0)
                    break;
                // Convert to QuestDB format
                const questdbSignals = batch.map(signal => ({
                    timestamp: signal.generatedAt,
                    id: signal.id,
                    symbol: signal.symbol,
                    type: signal.type,
                    direction: signal.direction,
                    strength: signal.strength,
                    timeframe: signal.timeframe,
                    price: signal.price,
                    targetPrice: signal.targetPrice || undefined,
                    stopLoss: signal.stopLoss || undefined,
                    confidenceScore: signal.confidenceScore,
                    expectedReturn: signal.expectedReturn,
                    expectedRisk: signal.expectedRisk,
                    riskRewardRatio: signal.riskRewardRatio,
                    source: signal.source,
                    metadata: signal.metadata || {},
                }));
                // Validate and insert
                try {
                    for (const signal of questdbSignals) {
                        await questdbService_1.questdbService.insertTradingSignal(signal);
                    }
                    migratedRecords += questdbSignals.length;
                }
                catch (error) {
                    logger_1.logger.error(`Error inserting trading signals batch at offset ${offset}:`, error);
                    failedRecords += questdbSignals.length;
                    errors.push({ record: batch, error: error instanceof Error ? error.message : 'Batch insert failed' });
                }
                offset += config.batchSize;
                // Update progress
                this.updateProgress('trading_signals', {
                    totalRecords,
                    processedRecords: offset,
                    successfulRecords: migratedRecords,
                    failedRecords,
                    progress: (offset / totalRecords) * 100,
                    estimatedTimeRemaining: this.calculateETA(startTime, offset, totalRecords),
                    throughput: migratedRecords / ((Date.now() - startTime) / 1000),
                    errors: errors.slice(-10).map(e => ({ ...e, timestamp: new Date() })),
                });
                if (offset % (config.batchSize * 10) === 0) {
                    logger_1.logger.info(`📈 Trading signals migration progress: ${offset}/${totalRecords} (${((offset / totalRecords) * 100).toFixed(1)}%)`);
                }
            }
            const duration = Date.now() - startTime;
            const averageThroughput = migratedRecords / (duration / 1000);
            return {
                success: failedRecords === 0,
                totalRecords,
                migratedRecords,
                failedRecords,
                duration,
                averageThroughput,
                errors,
            };
        }
        catch (error) {
            logger_1.logger.error('❌ Trading signals migration failed:', error);
            throw error;
        }
    }
    /**
     * Migrate MLPredictions table
     */
    async migrateMLPredictions(config) {
        const startTime = Date.now();
        let totalRecords = 0;
        let migratedRecords = 0;
        let failedRecords = 0;
        const errors = [];
        try {
            totalRecords = await this.prisma.mLPrediction.count({
                where: this.buildDateFilter(config, 'generatedAt'),
            });
            logger_1.logger.info(`📊 Migrating ${totalRecords} ML predictions records`);
            if (config.dryRun) {
                return {
                    success: true,
                    totalRecords,
                    migratedRecords: 0,
                    failedRecords: 0,
                    duration: Date.now() - startTime,
                    averageThroughput: 0,
                    errors: [],
                };
            }
            let offset = 0;
            while (offset < totalRecords) {
                const batch = await this.prisma.mLPrediction.findMany({
                    where: this.buildDateFilter(config, 'generatedAt'),
                    skip: offset,
                    take: config.batchSize,
                    orderBy: { generatedAt: 'asc' },
                });
                if (batch.length === 0)
                    break;
                const questdbPredictions = batch.map(prediction => ({
                    timestamp: prediction.generatedAt,
                    id: prediction.id,
                    modelId: prediction.modelId,
                    symbol: prediction.symbol,
                    timeframe: prediction.timeframe,
                    predictionType: prediction.predictionType,
                    values: Array.isArray(prediction.values) ? prediction.values : JSON.parse(prediction.values),
                    confidenceScores: Array.isArray(prediction.confidenceScores) ? prediction.confidenceScores : JSON.parse(prediction.confidenceScores),
                    metadata: prediction.metadata || {},
                }));
                try {
                    for (const prediction of questdbPredictions) {
                        await questdbService_1.questdbService.insertMLPrediction(prediction);
                    }
                    migratedRecords += questdbPredictions.length;
                }
                catch (error) {
                    failedRecords += questdbPredictions.length;
                    errors.push({ record: batch, error: error instanceof Error ? error.message : 'Batch insert failed' });
                }
                offset += config.batchSize;
            }
            const duration = Date.now() - startTime;
            return {
                success: failedRecords === 0,
                totalRecords,
                migratedRecords,
                failedRecords,
                duration,
                averageThroughput: migratedRecords / (duration / 1000),
                errors,
            };
        }
        catch (error) {
            logger_1.logger.error('❌ ML predictions migration failed:', error);
            throw error;
        }
    }
    /**
     * Migrate PerformanceMetrics table
     */
    async migratePerformanceMetrics(config) {
        const startTime = Date.now();
        let totalRecords = 0;
        let migratedRecords = 0;
        let failedRecords = 0;
        const errors = [];
        try {
            totalRecords = await this.prisma.performanceMetric.count({
                where: this.buildDateFilter(config, 'timestamp'),
            });
            logger_1.logger.info(`📊 Migrating ${totalRecords} performance metrics records`);
            if (config.dryRun) {
                return {
                    success: true,
                    totalRecords,
                    migratedRecords: 0,
                    failedRecords: 0,
                    duration: Date.now() - startTime,
                    averageThroughput: 0,
                    errors: [],
                };
            }
            let offset = 0;
            while (offset < totalRecords) {
                const batch = await this.prisma.performanceMetric.findMany({
                    where: this.buildDateFilter(config, 'timestamp'),
                    skip: offset,
                    take: config.batchSize,
                    orderBy: { timestamp: 'asc' },
                });
                if (batch.length === 0)
                    break;
                const questdbMetrics = batch.map(metric => ({
                    timestamp: metric.timestamp,
                    system: metric.system,
                    component: metric.component,
                    metric: metric.metric,
                    unit: metric.unit,
                    value: metric.value,
                    tags: metric.tags || {},
                }));
                try {
                    for (const metric of questdbMetrics) {
                        await questdbService_1.questdbService.insertPerformanceMetric(metric);
                    }
                    migratedRecords += questdbMetrics.length;
                }
                catch (error) {
                    failedRecords += questdbMetrics.length;
                    errors.push({ record: batch, error: error instanceof Error ? error.message : 'Batch insert failed' });
                }
                offset += config.batchSize;
            }
            const duration = Date.now() - startTime;
            return {
                success: failedRecords === 0,
                totalRecords,
                migratedRecords,
                failedRecords,
                duration,
                averageThroughput: migratedRecords / (duration / 1000),
                errors,
            };
        }
        catch (error) {
            logger_1.logger.error('❌ Performance metrics migration failed:', error);
            throw error;
        }
    }
    /**
     * Get migration progress for a table
     */
    getMigrationProgress(tableName) {
        return this.migrationProgress.get(tableName) || null;
    }
    /**
     * Get migration progress for all tables
     */
    getAllMigrationProgress() {
        const progress = {};
        this.migrationProgress.forEach((value, key) => {
            progress[key] = value;
        });
        return progress;
    }
    // Private helper methods
    buildDateFilter(config, dateField = 'recordedAt') {
        const filter = {};
        if (config.startDate || config.endDate) {
            filter[dateField] = {};
            if (config.startDate) {
                filter[dateField].gte = config.startDate;
            }
            if (config.endDate) {
                filter[dateField].lte = config.endDate;
            }
        }
        return filter;
    }
    updateProgress(tableName, progress) {
        this.migrationProgress.set(tableName, progress);
    }
    calculateETA(startTime, processed, total) {
        if (processed === 0)
            return 0;
        const elapsed = Date.now() - startTime;
        const rate = processed / elapsed;
        const remaining = total - processed;
        return remaining / rate / 1000; // Convert to seconds
    }
}
exports.QuestDBMigrationPipeline = QuestDBMigrationPipeline;
//# sourceMappingURL=questdb-migration.js.map