#!/usr/bin/env node

/**
 * Simulate Position Management
 * Demonstrates AI position management with simulated Delta position
 */

import { AIPositionManager, ManagedPosition } from '../services/aiPositionManager';
import DeltaExchangeAPI from '../services/deltaApiService';
import { logger } from '../utils/logger';

class PositionSimulator {
  private aiManager: AIPositionManager;
  private simulatedPosition: any;
  private currentPrice: number;
  private priceDirection: number = 1;
  private iteration: number = 0;

  constructor() {
    // Create a mock Delta API for simulation
    const mockDeltaApi = this.createMockDeltaApi();
    this.aiManager = new AIPositionManager(mockDeltaApi);
    
    // Create a simulated position (like the one you might have)
    this.simulatedPosition = {
      symbol: 'BTC_USDT',
      size: '0.01', // 0.01 BTC long position
      entry_price: '95000', // Entry at $95,000
      unrealized_pnl: '0',
    };
    
    this.currentPrice = 95000; // Starting price
  }

  /**
   * Create mock Delta API for simulation
   */
  private createMockDeltaApi(): any {
    return {
      getPositions: async () => [this.simulatedPosition],
      getTicker: async (symbol: string) => ({
        close: this.currentPrice.toString(),
        last_price: this.currentPrice.toString(),
      }),
      // Mock other methods as needed
    };
  }

  /**
   * Run position management simulation
   */
  public async runSimulation(): Promise<void> {
    logger.info('🎮 STARTING POSITION MANAGEMENT SIMULATION');
    logger.info('=' .repeat(80));
    
    logger.info('📊 SIMULATED POSITION:');
    logger.info(`   Symbol: ${this.simulatedPosition.symbol}`);
    logger.info(`   Size: ${this.simulatedPosition.size} BTC (LONG)`);
    logger.info(`   Entry Price: $${this.simulatedPosition.entry_price}`);
    logger.info(`   Current Price: $${this.currentPrice.toFixed(2)}`);
    
    logger.info('\n🤖 Starting AI Position Management...');
    
    // Start AI management
    await this.aiManager.startManagement();
    
    // Simulate price movements and AI responses
    const maxIterations = 20;
    
    for (let i = 0; i < maxIterations; i++) {
      this.iteration = i + 1;
      
      // Simulate price movement
      this.simulateMarketMovement();
      
      // Update position P&L
      this.updatePositionPnL();
      
      // Log current state
      this.logCurrentState();
      
      // Wait 3 seconds between iterations
      await this.sleep(3000);
      
      // Check if position was closed by AI
      const managedPositions = this.aiManager.getManagedPositions();
      const activePosition = managedPositions.find(p => p.status === 'MANAGING');
      
      if (!activePosition) {
        logger.info('🏁 Position closed by AI - simulation complete');
        break;
      }
    }
    
    // Stop AI management
    this.aiManager.stopManagement();
    
    // Display final results
    this.displayFinalResults();
  }

  /**
   * Simulate realistic market movement
   */
  private simulateMarketMovement(): void {
    // Simulate realistic BTC price movement
    const volatility = 0.02; // 2% volatility
    const randomChange = (Math.random() - 0.5) * 2 * volatility;
    
    // Add some trend (60% chance to continue current direction)
    if (Math.random() > 0.4) {
      this.priceDirection *= -1; // Change direction
    }
    
    const trendInfluence = this.priceDirection * 0.005; // 0.5% trend
    const totalChange = randomChange + trendInfluence;
    
    this.currentPrice *= (1 + totalChange);
    
    // Keep price within reasonable bounds
    this.currentPrice = Math.max(80000, Math.min(120000, this.currentPrice));
  }

  /**
   * Update position P&L
   */
  private updatePositionPnL(): void {
    const entryPrice = parseFloat(this.simulatedPosition.entry_price);
    const size = parseFloat(this.simulatedPosition.size);
    const priceChange = this.currentPrice - entryPrice;
    const unrealizedPnl = priceChange * size;
    
    this.simulatedPosition.unrealized_pnl = unrealizedPnl.toString();
  }

  /**
   * Log current state
   */
  private logCurrentState(): void {
    const entryPrice = parseFloat(this.simulatedPosition.entry_price);
    const unrealizedPnl = parseFloat(this.simulatedPosition.unrealized_pnl);
    const profitPercent = ((this.currentPrice - entryPrice) / entryPrice) * 100;
    
    logger.info(`\n📊 Iteration ${this.iteration}:`);
    logger.info(`   Current Price: $${this.currentPrice.toFixed(2)}`);
    logger.info(`   Unrealized P&L: $${unrealizedPnl.toFixed(2)} (${profitPercent.toFixed(2)}%)`);
    
    // Show AI management status
    const managedPositions = this.aiManager.getManagedPositions();
    if (managedPositions.length > 0) {
      const position = managedPositions[0];
      const executedExits = position.partialExits.filter(e => e.executed).length;
      logger.info(`   AI Status: ${position.status}`);
      logger.info(`   Partial Exits: ${executedExits}/${position.partialExits.length}`);
      logger.info(`   Stop Loss: $${position.stopLoss.toFixed(2)}`);
      
      if (position.aiRecommendations.length > 0) {
        const lastRecommendation = position.aiRecommendations[position.aiRecommendations.length - 1];
        logger.info(`   Last AI Action: ${lastRecommendation.split(': ')[1]}`);
      }
    }
  }

  /**
   * Display final results
   */
  private displayFinalResults(): void {
    const managedPositions = this.aiManager.getManagedPositions();
    
    logger.info('\n🎯 SIMULATION RESULTS:');
    logger.info('=' .repeat(80));
    
    if (managedPositions.length > 0) {
      const position = managedPositions[0];
      const entryPrice = parseFloat(this.simulatedPosition.entry_price);
      const finalPnl = parseFloat(this.simulatedPosition.unrealized_pnl);
      const profitPercent = ((this.currentPrice - entryPrice) / entryPrice) * 100;
      const executedExits = position.partialExits.filter(e => e.executed);
      
      logger.info('📊 POSITION SUMMARY:');
      logger.info(`   Entry Price: $${entryPrice.toFixed(2)}`);
      logger.info(`   Final Price: $${this.currentPrice.toFixed(2)}`);
      logger.info(`   Total P&L: $${finalPnl.toFixed(2)} (${profitPercent.toFixed(2)}%)`);
      logger.info(`   Position Status: ${position.status}`);
      
      logger.info('\n🤖 AI MANAGEMENT PERFORMANCE:');
      logger.info(`   Partial Exits Executed: ${executedExits.length}/${position.partialExits.length}`);
      
      if (executedExits.length > 0) {
        logger.info('   Exit Details:');
        executedExits.forEach(exit => {
          logger.info(`     Level ${exit.level}: ${exit.percentage}% at $${exit.targetPrice.toFixed(2)}`);
        });
      }
      
      logger.info(`   Final Stop Loss: $${position.stopLoss.toFixed(2)}`);
      logger.info(`   Total AI Recommendations: ${position.aiRecommendations.length}`);
      
      if (position.aiRecommendations.length > 0) {
        logger.info('\n📝 AI RECOMMENDATION HISTORY:');
        position.aiRecommendations.forEach((rec, index) => {
          const [timestamp, action] = rec.split(': ');
          logger.info(`   ${index + 1}. ${action}`);
        });
      }
    }
    
    logger.info('\n✅ SIMULATION COMPLETE');
    logger.info('🚀 This demonstrates how the AI will manage your real Delta Exchange position!');
  }

  /**
   * Sleep utility
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

/**
 * Main execution
 */
async function main() {
  const simulator = new PositionSimulator();
  await simulator.runSimulation();
}

// Run if this script is executed directly
if (require.main === module) {
  main().catch(error => {
    logger.error('💥 Simulation failed:', error);
    process.exit(1);
  });
}

export { PositionSimulator };
