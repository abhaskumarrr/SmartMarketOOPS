#!/usr/bin/env node

/**
 * AI Position Manager Runner
 * Manages your Delta Exchange positions with AI-powered dynamic take profit
 */

import DeltaExchangeAPI from '../services/deltaApiService';
import { AIPositionManager } from '../services/aiPositionManager';
import { logger } from '../utils/logger';

class AIPositionManagerRunner {
  private deltaApi: DeltaExchangeAPI;
  private aiManager: AIPositionManager;
  private isRunning: boolean = false;

  constructor() {
    this.deltaApi = new DeltaExchangeAPI({ testnet: true });
    this.aiManager = new AIPositionManager(this.deltaApi);
  }

  /**
   * Start AI position management
   */
  public async start(): Promise<void> {
    try {
      logger.info('🤖 STARTING AI POSITION MANAGEMENT SYSTEM');
      logger.info('=' .repeat(80));
      
      // Get credentials
      const credentials = {
        key: process.env.DELTA_EXCHANGE_API_KEY || '',
        secret: process.env.DELTA_EXCHANGE_API_SECRET || ''
      };

      if (!credentials.key || !credentials.secret) {
        logger.error('❌ Delta API credentials not found');
        logger.info('🔧 Please set DELTA_EXCHANGE_API_KEY and DELTA_EXCHANGE_API_SECRET');
        return;
      }

      logger.info('🔑 Credentials found');
      logger.info(`   API Key: ${credentials.key.substring(0, 8)}...`);

      // Initialize Delta API
      await this.deltaApi.initialize(credentials);
      logger.info('✅ Delta Exchange API initialized');

      // Test connection
      await this.testConnection();

      // Start AI management
      await this.aiManager.startManagement();
      this.isRunning = true;

      logger.info('🚀 AI Position Management System is now running!');
      logger.info('📊 Monitoring your positions every 30 seconds...');
      logger.info('🤖 AI will automatically manage take profits, stop losses, and exits');
      logger.info('\n💡 Features active:');
      logger.info('   ✅ Dynamic take profit levels');
      logger.info('   ✅ Trailing stop losses');
      logger.info('   ✅ Partial exit optimization');
      logger.info('   ✅ Market regime adaptation');
      logger.info('   ✅ Risk management');

      // Keep running
      await this.keepRunning();

    } catch (error) {
      logger.error('❌ Failed to start AI position manager:', error.message);
      
      if (error.message.includes('ip_not_whitelisted')) {
        logger.info('\n🔧 IP WHITELISTING REQUIRED:');
        logger.info('   1. Login to your Delta Exchange account');
        logger.info('   2. Go to API Management section');
        logger.info('   3. Edit your API key settings');
        logger.info('   4. Add your IP address: 223.226.141.59');
        logger.info('   5. Save changes and try again');
      }
    }
  }

  /**
   * Test connection to Delta Exchange
   */
  private async testConnection(): Promise<void> {
    try {
      // Test public endpoint
      const markets = await this.deltaApi.getMarkets();
      logger.info(`✅ Connection test passed - ${markets.length} markets available`);

      // Test authenticated endpoint
      try {
        const positions = await this.deltaApi.getPositions();
        logger.info(`✅ Authentication successful - ${positions.length} positions found`);
        
        if (positions.length > 0) {
          logger.info('🎯 POSITIONS TO MANAGE:');
          positions.forEach((pos, index) => {
            const side = parseFloat(pos.size) > 0 ? 'LONG' : 'SHORT';
            logger.info(`   ${index + 1}. ${pos.symbol}: ${side} ${Math.abs(parseFloat(pos.size))} @ $${pos.entry_price}`);
          });
        } else {
          logger.info('📊 No active positions found - AI will monitor for new positions');
        }
      } catch (authError) {
        if (authError.message.includes('ip_not_whitelisted')) {
          throw new Error('ip_not_whitelisted_for_api_key');
        }
        throw authError;
      }

    } catch (error) {
      throw error;
    }
  }

  /**
   * Keep the system running
   */
  private async keepRunning(): Promise<void> {
    // Set up graceful shutdown
    process.on('SIGINT', () => {
      logger.info('\n🛑 Received shutdown signal...');
      this.stop();
      process.exit(0);
    });

    process.on('SIGTERM', () => {
      logger.info('\n🛑 Received termination signal...');
      this.stop();
      process.exit(0);
    });

    // Display status every 5 minutes
    setInterval(() => {
      this.displayStatus();
    }, 300000); // 5 minutes

    // Keep process alive
    while (this.isRunning) {
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }

  /**
   * Display current status
   */
  private displayStatus(): void {
    const managedPositions = this.aiManager.getManagedPositions();
    
    logger.info('\n🤖 AI POSITION MANAGER STATUS:');
    logger.info(`   📊 Positions under management: ${managedPositions.length}`);
    logger.info(`   🕐 Last update: ${new Date().toLocaleTimeString()}`);
    
    if (managedPositions.length > 0) {
      logger.info('   📈 Position Summary:');
      managedPositions.forEach(pos => {
        const profitPercent = ((pos.currentPrice - pos.entryPrice) / pos.entryPrice) * 100;
        const executedExits = pos.partialExits.filter(e => e.executed).length;
        
        logger.info(`     ${pos.symbol}: ${pos.side} $${pos.unrealizedPnl.toFixed(2)} (${profitPercent.toFixed(1)}%)`);
        logger.info(`       Exits: ${executedExits}/${pos.partialExits.length}, Stop: $${pos.stopLoss.toFixed(2)}`);
      });
    }
  }

  /**
   * Stop AI position management
   */
  public stop(): void {
    this.isRunning = false;
    this.aiManager.stopManagement();
    logger.info('🛑 AI Position Management System stopped');
  }
}

/**
 * Main execution
 */
async function main() {
  const runner = new AIPositionManagerRunner();
  await runner.start();
}

// Run if this script is executed directly
if (require.main === module) {
  main().catch(error => {
    logger.error('💥 AI Position Manager failed:', error);
    process.exit(1);
  });
}

export { AIPositionManagerRunner };
