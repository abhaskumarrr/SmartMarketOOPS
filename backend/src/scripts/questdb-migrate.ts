#!/usr/bin/env node

/**
 * QuestDB Migration CLI Script
 * Command-line interface for migrating data from PostgreSQL to QuestDB
 */

import { Command } from 'commander';
import { QuestDBMigrationPipeline } from '../migration/questdb-migration';
import { questdbConnection, validateQuestDBEnvironment } from '../config/questdb';
import { logger } from '../utils/logger';
import { QuestDBMigrationConfig } from '../types/questdb';

const program = new Command();

program
  .name('questdb-migrate')
  .description('Migrate time-series data from PostgreSQL to QuestDB')
  .version('1.0.0');

program
  .command('migrate')
  .description('Migrate all time-series tables to QuestDB')
  .option('-b, --batch-size <size>', 'Batch size for migration', '1000')
  .option('-w, --workers <count>', 'Number of parallel workers', '2')
  .option('--dry-run', 'Perform a dry run without migrating data')
  .option('--validate', 'Validate data during migration', true)
  .option('--start-date <date>', 'Start date for migration (ISO format)')
  .option('--end-date <date>', 'End date for migration (ISO format)')
  .option('--tables <tables>', 'Comma-separated list of tables to migrate (metrics,trading_signals,ml_predictions,performance_metrics)')
  .action(async (options) => {
    try {
      logger.info('🚀 Starting QuestDB migration...');

      // Validate environment
      const envValidation = validateQuestDBEnvironment();
      if (!envValidation.valid) {
        logger.error('❌ Environment validation failed:', envValidation.errors);
        process.exit(1);
      }

      // Parse options
      const config: Partial<QuestDBMigrationConfig> = {
        batchSize: parseInt(options.batchSize, 10),
        parallelWorkers: parseInt(options.workers, 10),
        validateData: options.validate,
        dryRun: options.dryRun,
      };

      if (options.startDate) {
        config.startDate = new Date(options.startDate);
        if (isNaN(config.startDate.getTime())) {
          logger.error('❌ Invalid start date format. Use ISO format (YYYY-MM-DDTHH:mm:ss.sssZ)');
          process.exit(1);
        }
      }

      if (options.endDate) {
        config.endDate = new Date(options.endDate);
        if (isNaN(config.endDate.getTime())) {
          logger.error('❌ Invalid end date format. Use ISO format (YYYY-MM-DDTHH:mm:ss.sssZ)');
          process.exit(1);
        }
      }

      // Initialize migration pipeline
      const pipeline = new QuestDBMigrationPipeline();
      await pipeline.initialize();

      // Run migration
      const results = await pipeline.migrateAll(config);

      // Display results
      logger.info('📊 Migration Results:');
      let totalMigrated = 0;
      let totalFailed = 0;

      for (const result of results) {
        totalMigrated += result.migratedRecords;
        totalFailed += result.failedRecords;
        
        logger.info(`  ${result.success ? '✅' : '❌'} ${result.migratedRecords}/${result.totalRecords} records (${(result.averageThroughput).toFixed(2)} records/sec)`);
        
        if (result.errors.length > 0) {
          logger.warn(`    Errors: ${result.errors.length}`);
          result.errors.slice(0, 5).forEach(error => {
            logger.warn(`      - ${error.error}`);
          });
        }
      }

      logger.info(`🎉 Migration completed: ${totalMigrated} records migrated, ${totalFailed} failed`);

      await pipeline.shutdown();
      process.exit(totalFailed > 0 ? 1 : 0);

    } catch (error) {
      logger.error('❌ Migration failed:', error);
      process.exit(1);
    }
  });

program
  .command('status')
  .description('Check migration status and progress')
  .action(async () => {
    try {
      const pipeline = new QuestDBMigrationPipeline();
      await pipeline.initialize();

      const progress = pipeline.getAllMigrationProgress();
      
      if (Object.keys(progress).length === 0) {
        logger.info('📊 No active migrations found');
      } else {
        logger.info('📊 Migration Progress:');
        for (const [table, prog] of Object.entries(progress)) {
          logger.info(`  ${table}: ${prog.processedRecords}/${prog.totalRecords} (${prog.progress.toFixed(1)}%)`);
          logger.info(`    Success: ${prog.successfulRecords}, Failed: ${prog.failedRecords}`);
          logger.info(`    Throughput: ${prog.throughput.toFixed(2)} records/sec`);
          logger.info(`    ETA: ${prog.estimatedTimeRemaining.toFixed(0)} seconds`);
        }
      }

      await pipeline.shutdown();

    } catch (error) {
      logger.error('❌ Failed to get migration status:', error);
      process.exit(1);
    }
  });

program
  .command('validate')
  .description('Validate QuestDB connection and environment')
  .action(async () => {
    try {
      logger.info('🔍 Validating QuestDB environment...');

      // Validate environment variables
      const envValidation = validateQuestDBEnvironment();
      if (!envValidation.valid) {
        logger.error('❌ Environment validation failed:');
        envValidation.errors.forEach(error => logger.error(`  - ${error}`));
        process.exit(1);
      }
      logger.info('✅ Environment variables are valid');

      // Test QuestDB connection
      logger.info('🔌 Testing QuestDB connection...');
      await questdbConnection.connect();
      
      const healthCheck = await questdbConnection.healthCheck();
      if (!healthCheck) {
        logger.error('❌ QuestDB health check failed');
        process.exit(1);
      }
      
      logger.info('✅ QuestDB connection successful');
      
      // Display connection info
      const stats = questdbConnection.getStats();
      logger.info('📊 Connection Details:', {
        host: stats.clientOptions.host,
        port: stats.clientOptions.port,
        tls: stats.clientOptions.tls,
        isConnected: stats.isConnected,
      });

      await questdbConnection.disconnect();
      logger.info('🎉 Validation completed successfully');

    } catch (error) {
      logger.error('❌ Validation failed:', error);
      process.exit(1);
    }
  });

program
  .command('test-insert')
  .description('Test data insertion into QuestDB')
  .option('-c, --count <count>', 'Number of test records to insert', '100')
  .action(async (options) => {
    try {
      const count = parseInt(options.count, 10);
      logger.info(`🧪 Testing QuestDB insertion with ${count} records...`);

      const pipeline = new QuestDBMigrationPipeline();
      await pipeline.initialize();

      // Test metric insertion
      const testMetrics = Array.from({ length: count }, (_, i) => ({
        timestamp: new Date(Date.now() - i * 60000), // 1 minute intervals
        name: `test_metric_${i % 10}`,
        value: Math.random() * 100,
        tags: {
          test: 'true',
          batch: Math.floor(i / 10).toString(),
        },
      }));

      const startTime = Date.now();
      
      // Import questdbService properly
      const { questdbService } = await import('../services/questdbService');
      await questdbService.insertMetrics(testMetrics);
      
      const duration = Date.now() - startTime;
      const throughput = count / (duration / 1000);

      logger.info(`✅ Test insertion completed:`);
      logger.info(`  Records: ${count}`);
      logger.info(`  Duration: ${duration}ms`);
      logger.info(`  Throughput: ${throughput.toFixed(2)} records/sec`);

      await pipeline.shutdown();

    } catch (error) {
      logger.error('❌ Test insertion failed:', error);
      process.exit(1);
    }
  });

program
  .command('query-test')
  .description('Test QuestDB query performance')
  .action(async () => {
    try {
      logger.info('🔍 Testing QuestDB query performance...');

      const pipeline = new QuestDBMigrationPipeline();
      await pipeline.initialize();

      const { questdbService } = await import('../services/questdbService');

      // Test various queries
      const queries = [
        'SELECT count() FROM metrics',
        'SELECT count() FROM trading_signals',
        'SELECT count() FROM ml_predictions',
        'SELECT count() FROM performance_metrics',
      ];

      for (const query of queries) {
        const startTime = Date.now();
        try {
          const result = await questdbService.executeQuery(query);
          const duration = Date.now() - startTime;
          logger.info(`✅ Query: ${query}`);
          logger.info(`  Result: ${JSON.stringify(result[0] || {})}`);
          logger.info(`  Duration: ${duration}ms`);
        } catch (error) {
          logger.warn(`⚠️ Query failed: ${query} - ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
      }

      await pipeline.shutdown();

    } catch (error) {
      logger.error('❌ Query test failed:', error);
      process.exit(1);
    }
  });

// Handle uncaught errors
process.on('uncaughtException', (error) => {
  logger.error('💥 Uncaught Exception:', error);
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  logger.error('💥 Unhandled Rejection at:', { promise, reason });
  process.exit(1);
});

// Parse command line arguments
program.parse();

// If no command provided, show help
if (!process.argv.slice(2).length) {
  program.outputHelp();
}
