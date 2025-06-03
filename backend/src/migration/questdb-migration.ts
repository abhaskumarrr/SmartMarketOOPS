/**
 * QuestDB Migration Pipeline
 * Migrates time-series data from PostgreSQL to QuestDB for performance optimization
 */

import { PrismaClient } from '../../generated/prisma';
import { questdbService } from '../services/questdbService';
import { logger } from '../utils/logger';
import { DataValidator } from './data-validator';
import {
  QuestDBMigrationConfig,
  QuestDBMigrationProgress,
  QuestDBMigrationResult,
  QuestDBTableName,
} from '../types/questdb';
import {
  MetricData,
  TradingSignalData,
  MLPredictionData,
  PerformanceMetricData,
} from '../services/questdbService';

export class QuestDBMigrationPipeline {
  private prisma: PrismaClient;
  private validator: DataValidator;
  private migrationProgress: Map<string, QuestDBMigrationProgress> = new Map();

  constructor() {
    this.prisma = new PrismaClient();
    this.validator = new DataValidator();
  }

  /**
   * Initialize the migration pipeline
   */
  public async initialize(): Promise<void> {
    try {
      await this.prisma.$connect();
      await questdbService.initialize();
      logger.info('✅ Migration pipeline initialized successfully');
    } catch (error) {
      logger.error('❌ Failed to initialize migration pipeline:', error);
      throw error;
    }
  }

  /**
   * Shutdown the migration pipeline
   */
  public async shutdown(): Promise<void> {
    try {
      await this.prisma.$disconnect();
      await questdbService.shutdown();
      logger.info('🔌 Migration pipeline shutdown completed');
    } catch (error) {
      logger.error('❌ Error during migration pipeline shutdown:', error);
      throw error;
    }
  }

  /**
   * Migrate all time-series tables
   */
  public async migrateAll(config: Partial<QuestDBMigrationConfig> = {}): Promise<QuestDBMigrationResult[]> {
    const defaultConfig: QuestDBMigrationConfig = {
      sourceTable: '',
      targetTable: '',
      batchSize: 1000,
      parallelWorkers: 2,
      validateData: true,
      dryRun: false,
      ...config,
    };

    const tables: Array<{ source: string; target: QuestDBTableName; migrator: string }> = [
      { source: 'Metric', target: 'metrics', migrator: 'migrateMetrics' },
      { source: 'TradingSignal', target: 'trading_signals', migrator: 'migrateTradingSignals' },
      { source: 'MLPrediction', target: 'ml_predictions', migrator: 'migrateMLPredictions' },
      { source: 'PerformanceMetric', target: 'performance_metrics', migrator: 'migratePerformanceMetrics' },
    ];

    const results: QuestDBMigrationResult[] = [];

    for (const table of tables) {
      try {
        logger.info(`🚀 Starting migration for ${table.source} -> ${table.target}`);
        
        const tableConfig = {
          ...defaultConfig,
          sourceTable: table.source,
          targetTable: table.target,
        };

        const result = await (this as any)[table.migrator](tableConfig);
        results.push(result);

        logger.info(`✅ Completed migration for ${table.source}: ${result.migratedRecords} records`);
      } catch (error) {
        logger.error(`❌ Failed to migrate ${table.source}:`, error);
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
  public async migrateMetrics(config: QuestDBMigrationConfig): Promise<QuestDBMigrationResult> {
    const startTime = Date.now();
    let totalRecords = 0;
    let migratedRecords = 0;
    let failedRecords = 0;
    const errors: Array<{ record: any; error: string }> = [];

    try {
      // Get total count
      totalRecords = await this.prisma.metric.count({
        where: this.buildDateFilter(config),
      });

      logger.info(`📊 Migrating ${totalRecords} metrics records`);

      if (config.dryRun) {
        logger.info('🔍 Dry run mode - no data will be migrated');
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

        if (batch.length === 0) break;

        // Convert to QuestDB format
        const questdbMetrics: MetricData[] = batch.map(metric => ({
          timestamp: metric.recordedAt,
          name: metric.name,
          value: metric.value,
          tags: metric.tags as Record<string, string | number> || {},
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
          await questdbService.insertMetrics(questdbMetrics);
          migratedRecords += questdbMetrics.length;
        } catch (error) {
          logger.error(`Error inserting metrics batch at offset ${offset}:`, error);
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
          logger.info(`📈 Metrics migration progress: ${offset}/${totalRecords} (${((offset / totalRecords) * 100).toFixed(1)}%)`);
        }
      }

      const duration = Date.now() - startTime;
      const averageThroughput = migratedRecords / (duration / 1000);

      logger.info(`✅ Metrics migration completed: ${migratedRecords}/${totalRecords} records in ${duration}ms (${averageThroughput.toFixed(2)} records/sec)`);

      return {
        success: failedRecords === 0,
        totalRecords,
        migratedRecords,
        failedRecords,
        duration,
        averageThroughput,
        errors,
      };

    } catch (error) {
      logger.error('❌ Metrics migration failed:', error);
      throw error;
    }
  }

  /**
   * Migrate TradingSignals table
   */
  public async migrateTradingSignals(config: QuestDBMigrationConfig): Promise<QuestDBMigrationResult> {
    const startTime = Date.now();
    let totalRecords = 0;
    let migratedRecords = 0;
    let failedRecords = 0;
    const errors: Array<{ record: any; error: string }> = [];

    try {
      // Get total count
      totalRecords = await this.prisma.tradingSignal.count({
        where: this.buildDateFilter(config, 'generatedAt'),
      });

      logger.info(`📊 Migrating ${totalRecords} trading signals records`);

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

        if (batch.length === 0) break;

        // Convert to QuestDB format
        const questdbSignals: TradingSignalData[] = batch.map(signal => ({
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
          metadata: signal.metadata as Record<string, any> || {},
        }));

        // Validate and insert
        try {
          for (const signal of questdbSignals) {
            await questdbService.insertTradingSignal(signal);
          }
          migratedRecords += questdbSignals.length;
        } catch (error) {
          logger.error(`Error inserting trading signals batch at offset ${offset}:`, error);
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
          logger.info(`📈 Trading signals migration progress: ${offset}/${totalRecords} (${((offset / totalRecords) * 100).toFixed(1)}%)`);
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

    } catch (error) {
      logger.error('❌ Trading signals migration failed:', error);
      throw error;
    }
  }

  /**
   * Migrate MLPredictions table
   */
  public async migrateMLPredictions(config: QuestDBMigrationConfig): Promise<QuestDBMigrationResult> {
    const startTime = Date.now();
    let totalRecords = 0;
    let migratedRecords = 0;
    let failedRecords = 0;
    const errors: Array<{ record: any; error: string }> = [];

    try {
      totalRecords = await this.prisma.mLPrediction.count({
        where: this.buildDateFilter(config, 'generatedAt'),
      });

      logger.info(`📊 Migrating ${totalRecords} ML predictions records`);

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

        if (batch.length === 0) break;

        const questdbPredictions: MLPredictionData[] = batch.map(prediction => ({
          timestamp: prediction.generatedAt,
          id: prediction.id,
          modelId: prediction.modelId,
          symbol: prediction.symbol,
          timeframe: prediction.timeframe,
          predictionType: prediction.predictionType,
          values: Array.isArray(prediction.values) ? prediction.values : JSON.parse(prediction.values as string),
          confidenceScores: Array.isArray(prediction.confidenceScores) ? prediction.confidenceScores : JSON.parse(prediction.confidenceScores as string),
          metadata: prediction.metadata as Record<string, any> || {},
        }));

        try {
          for (const prediction of questdbPredictions) {
            await questdbService.insertMLPrediction(prediction);
          }
          migratedRecords += questdbPredictions.length;
        } catch (error) {
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

    } catch (error) {
      logger.error('❌ ML predictions migration failed:', error);
      throw error;
    }
  }

  /**
   * Migrate PerformanceMetrics table
   */
  public async migratePerformanceMetrics(config: QuestDBMigrationConfig): Promise<QuestDBMigrationResult> {
    const startTime = Date.now();
    let totalRecords = 0;
    let migratedRecords = 0;
    let failedRecords = 0;
    const errors: Array<{ record: any; error: string }> = [];

    try {
      totalRecords = await this.prisma.performanceMetric.count({
        where: this.buildDateFilter(config, 'timestamp'),
      });

      logger.info(`📊 Migrating ${totalRecords} performance metrics records`);

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

        if (batch.length === 0) break;

        const questdbMetrics: PerformanceMetricData[] = batch.map(metric => ({
          timestamp: metric.timestamp,
          system: metric.system as any,
          component: metric.component,
          metric: metric.metric,
          unit: metric.unit,
          value: metric.value,
          tags: metric.tags as Record<string, string | number> || {},
        }));

        try {
          for (const metric of questdbMetrics) {
            await questdbService.insertPerformanceMetric(metric);
          }
          migratedRecords += questdbMetrics.length;
        } catch (error) {
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

    } catch (error) {
      logger.error('❌ Performance metrics migration failed:', error);
      throw error;
    }
  }

  /**
   * Get migration progress for a table
   */
  public getMigrationProgress(tableName: string): QuestDBMigrationProgress | null {
    return this.migrationProgress.get(tableName) || null;
  }

  /**
   * Get migration progress for all tables
   */
  public getAllMigrationProgress(): Record<string, QuestDBMigrationProgress> {
    const progress: Record<string, QuestDBMigrationProgress> = {};
    this.migrationProgress.forEach((value, key) => {
      progress[key] = value;
    });
    return progress;
  }

  // Private helper methods

  private buildDateFilter(config: QuestDBMigrationConfig, dateField: string = 'recordedAt'): any {
    const filter: any = {};
    
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

  private updateProgress(tableName: string, progress: QuestDBMigrationProgress): void {
    this.migrationProgress.set(tableName, progress);
  }

  private calculateETA(startTime: number, processed: number, total: number): number {
    if (processed === 0) return 0;
    
    const elapsed = Date.now() - startTime;
    const rate = processed / elapsed;
    const remaining = total - processed;
    
    return remaining / rate / 1000; // Convert to seconds
  }
}
