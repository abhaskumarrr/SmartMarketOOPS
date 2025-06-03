#!/usr/bin/env node

/**
 * Event-Driven System CLI Script
 * Command-line interface for managing the event-driven trading system
 */

import { Command } from 'commander';
import { eventDrivenTradingSystem } from '../services/eventDrivenTradingSystem';
import { redisStreamsService } from '../services/redisStreamsService';
import { redisConnection, validateRedisEnvironment } from '../config/redis';
import { logger } from '../utils/logger';
import { STREAM_NAMES, createEventId } from '../types/events';

const program = new Command();

program
  .name('event-driven-system')
  .description('Manage the event-driven trading system')
  .version('1.0.0');

program
  .command('start')
  .description('Start the event-driven trading system')
  .option('--market-data', 'Enable market data processing', true)
  .option('--signals', 'Enable signal processing', true)
  .option('--orders', 'Enable order processing', true)
  .option('--risk', 'Enable risk management', true)
  .option('--portfolio', 'Enable portfolio management', true)
  .option('--monitoring', 'Enable system monitoring', true)
  .action(async (options) => {
    try {
      logger.info('🚀 Starting event-driven trading system...');

      // Validate environment
      const envValidation = validateRedisEnvironment();
      if (!envValidation.valid) {
        logger.error('❌ Environment validation failed:', envValidation.errors);
        process.exit(1);
      }

      // Configure system
      eventDrivenTradingSystem.updateConfig({
        enableMarketDataProcessing: options.marketData,
        enableSignalProcessing: options.signals,
        enableOrderProcessing: options.orders,
        enableRiskManagement: options.risk,
        enablePortfolioManagement: options.portfolio,
        enableSystemMonitoring: options.monitoring,
      });

      // Start system
      await eventDrivenTradingSystem.start();

      // Set up graceful shutdown
      process.on('SIGINT', async () => {
        logger.info('🛑 Received SIGINT, shutting down gracefully...');
        await eventDrivenTradingSystem.stop();
        process.exit(0);
      });

      process.on('SIGTERM', async () => {
        logger.info('🛑 Received SIGTERM, shutting down gracefully...');
        await eventDrivenTradingSystem.stop();
        process.exit(0);
      });

      // Keep process running
      logger.info('✅ Event-driven trading system started. Press Ctrl+C to stop.');

    } catch (error) {
      logger.error('❌ Failed to start event-driven trading system:', error);
      process.exit(1);
    }
  });

program
  .command('stop')
  .description('Stop the event-driven trading system')
  .action(async () => {
    try {
      logger.info('🛑 Stopping event-driven trading system...');
      await eventDrivenTradingSystem.stop();
      logger.info('✅ Event-driven trading system stopped successfully');
      process.exit(0);
    } catch (error) {
      logger.error('❌ Failed to stop event-driven trading system:', error);
      process.exit(1);
    }
  });

program
  .command('status')
  .description('Check system status and statistics')
  .action(async () => {
    try {
      const isRunning = eventDrivenTradingSystem.isSystemRunning();
      const stats = eventDrivenTradingSystem.getStats();
      const config = eventDrivenTradingSystem.getConfig();
      const processorStats = eventDrivenTradingSystem.getProcessorStats();

      logger.info('📊 Event-Driven Trading System Status:');
      logger.info(`  Running: ${isRunning ? '✅ Yes' : '❌ No'}`);
      logger.info(`  Health: ${stats.systemHealth}`);
      logger.info(`  Uptime: ${Math.floor(stats.uptime / 1000)}s`);
      logger.info(`  Events Processed: ${stats.eventsProcessed}`);
      logger.info(`  Events/Second: ${stats.eventsPerSecond.toFixed(2)}`);
      logger.info(`  Errors: ${stats.errors}`);
      logger.info(`  Active Processors: ${stats.activeProcessors}`);

      logger.info('⚙️ Configuration:');
      Object.entries(config).forEach(([key, value]) => {
        logger.info(`  ${key}: ${value}`);
      });

      if (Object.keys(processorStats).length > 0) {
        logger.info('🔧 Processor Statistics:');
        Object.entries(processorStats).forEach(([name, stats]) => {
          logger.info(`  ${name}:`, stats);
        });
      }

    } catch (error) {
      logger.error('❌ Failed to get system status:', error);
      process.exit(1);
    }
  });

program
  .command('validate')
  .description('Validate Redis connection and streams')
  .action(async () => {
    try {
      logger.info('🔍 Validating Redis environment...');

      // Validate environment variables
      const envValidation = validateRedisEnvironment();
      if (!envValidation.valid) {
        logger.error('❌ Environment validation failed:');
        envValidation.errors.forEach(error => logger.error(`  - ${error}`));
        process.exit(1);
      }
      logger.info('✅ Environment variables are valid');

      // Test Redis connection
      logger.info('🔌 Testing Redis connection...');
      await redisConnection.connect();
      
      const healthCheck = await redisConnection.healthCheck();
      if (!healthCheck) {
        logger.error('❌ Redis health check failed');
        process.exit(1);
      }
      
      logger.info('✅ Redis connection successful');

      // Test Redis Streams service
      logger.info('📡 Testing Redis Streams service...');
      await redisStreamsService.initialize();
      
      const streamsHealthy = await redisStreamsService.healthCheck();
      if (!streamsHealthy) {
        logger.error('❌ Redis Streams service health check failed');
        process.exit(1);
      }

      logger.info('✅ Redis Streams service healthy');

      // Display connection info
      const stats = redisConnection.getStats();
      logger.info('📊 Connection Details:', {
        host: stats.config.host,
        port: stats.config.port,
        db: stats.config.db,
        isConnected: stats.isConnected,
      });

      await redisStreamsService.shutdown();
      await redisConnection.disconnect();
      logger.info('🎉 Validation completed successfully');

    } catch (error) {
      logger.error('❌ Validation failed:', error);
      process.exit(1);
    }
  });

program
  .command('streams')
  .description('Manage Redis streams')
  .option('--list', 'List all streams')
  .option('--info <stream>', 'Get stream information')
  .option('--trim <stream>', 'Trim stream to max length')
  .option('--max-length <length>', 'Maximum length for trimming', '1000')
  .action(async (options) => {
    try {
      await redisStreamsService.initialize();

      if (options.list) {
        logger.info('📡 Available Streams:');
        Object.entries(STREAM_NAMES).forEach(([key, streamName]) => {
          logger.info(`  ${key}: ${streamName}`);
        });
      }

      if (options.info) {
        const streamInfo = await redisStreamsService.getStreamInfo(options.info);
        if (streamInfo) {
          logger.info(`📊 Stream Info for ${options.info}:`, streamInfo);
        } else {
          logger.warn(`⚠️ Stream ${options.info} not found or empty`);
        }
      }

      if (options.trim) {
        const maxLength = parseInt(options.maxLength, 10);
        const trimmed = await redisStreamsService.trimStream(options.trim, maxLength);
        logger.info(`🗑️ Trimmed ${trimmed} messages from ${options.trim}`);
      }

      await redisStreamsService.shutdown();

    } catch (error) {
      logger.error('❌ Stream management failed:', error);
      process.exit(1);
    }
  });

program
  .command('test-events')
  .description('Test event publishing and processing')
  .option('-c, --count <count>', 'Number of test events to publish', '10')
  .option('-t, --type <type>', 'Event type to test', 'market-data')
  .action(async (options) => {
    try {
      const count = parseInt(options.count, 10);
      const eventType = options.type;

      logger.info(`🧪 Testing event publishing: ${count} ${eventType} events...`);

      await redisStreamsService.initialize();

      const startTime = Date.now();

      for (let i = 0; i < count; i++) {
        switch (eventType) {
          case 'market-data':
            await eventDrivenTradingSystem.publishMarketDataEvent(
              'BTCUSD',
              45000 + Math.random() * 1000,
              Math.random() * 100,
              'test-exchange'
            );
            break;

          case 'signal':
            await eventDrivenTradingSystem.publishTradingSignalEvent({
              signalId: createEventId(),
              symbol: 'BTCUSD',
              signalType: 'ENTRY',
              direction: Math.random() > 0.5 ? 'LONG' : 'SHORT',
              strength: 'MODERATE',
              timeframe: '1m',
              price: 45000 + Math.random() * 1000,
              confidenceScore: 60 + Math.random() * 30,
              expectedReturn: 0.02 + Math.random() * 0.03,
              expectedRisk: 0.01 + Math.random() * 0.02,
              riskRewardRatio: 1.5 + Math.random() * 1.5,
              modelSource: 'test-model',
            });
            break;

          default:
            throw new Error(`Unsupported event type: ${eventType}`);
        }

        if (i % 10 === 0) {
          logger.info(`📤 Published ${i}/${count} events...`);
        }
      }

      const duration = Date.now() - startTime;
      const throughput = count / (duration / 1000);

      logger.info(`✅ Test completed:`);
      logger.info(`  Events: ${count}`);
      logger.info(`  Duration: ${duration}ms`);
      logger.info(`  Throughput: ${throughput.toFixed(2)} events/sec`);

      await redisStreamsService.shutdown();

    } catch (error) {
      logger.error('❌ Event testing failed:', error);
      process.exit(1);
    }
  });

program
  .command('monitor')
  .description('Monitor system events in real-time')
  .option('-s, --stream <stream>', 'Stream to monitor', 'market-data-stream')
  .option('-g, --group <group>', 'Consumer group', 'monitoring-group')
  .action(async (options) => {
    try {
      logger.info(`👁️ Monitoring stream: ${options.stream} (group: ${options.group})`);

      await redisStreamsService.initialize();

      // Monitor events
      let eventCount = 0;
      const startTime = Date.now();

      while (true) {
        try {
          const events = await redisStreamsService.readEvents(options.stream, {
            groupName: options.group,
            consumerName: `monitor-${process.pid}`,
            blockTime: 1000,
            count: 10,
          });

          for (const event of events) {
            eventCount++;
            logger.info(`📥 Event ${eventCount}: ${event.type} (${event.source})`);
            logger.info(`  Data:`, JSON.stringify(event.data, null, 2));
          }

          if (events.length > 0) {
            const messageIds = events.map(e => e.id!);
            await redisStreamsService.acknowledgeEvents(options.stream, options.group, messageIds);
          }

        } catch (error) {
          logger.error('❌ Monitoring error:', error);
          await new Promise(resolve => setTimeout(resolve, 5000));
        }
      }

    } catch (error) {
      logger.error('❌ Monitoring failed:', error);
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
