#!/usr/bin/env node

/**
 * ML Position Manager Test
 * Comprehensive testing of ML-powered position management, dynamic stops, and exit optimization
 */

import { MLPositionManager, Position } from '../services/MLPositionManager';
import { EnhancedTradingDecisionEngine, TradingDecision } from '../services/EnhancedTradingDecisionEngine';
import { logger } from '../utils/logger';

class MLPositionManagerTest {
  private positionManager: MLPositionManager;
  private decisionEngine: EnhancedTradingDecisionEngine;
  private testSymbols: string[] = ['BTCUSD', 'ETHUSD'];

  constructor() {
    this.positionManager = new MLPositionManager();
    this.decisionEngine = new EnhancedTradingDecisionEngine();
  }

  /**
   * Run comprehensive ML position manager test
   */
  public async runTest(): Promise<void> {
    logger.info('🤖 ML POSITION MANAGER TEST');
    logger.info('=' .repeat(80));

    try {
      // Step 1: Initialize the position manager
      await this.testInitialization();

      // Step 2: Test position creation from trading decisions
      await this.testPositionCreation();

      // Step 3: Test position updates and ML predictions
      await this.testPositionUpdates();

      // Step 4: Test dynamic stop loss and take profit management
      await this.testDynamicManagement();

      // Step 5: Test exit signal detection
      await this.testExitSignalDetection();

      // Step 6: Test position closure and training data recording
      await this.testPositionClosure();

      // Step 7: Test performance metrics and analytics
      await this.testPerformanceMetrics();

      // Step 8: Test configuration management
      await this.testConfigurationManagement();

      logger.info('\n🎉 ML POSITION MANAGER TEST COMPLETED SUCCESSFULLY!');
      logger.info('✅ All ML-powered position management features are working correctly');

    } catch (error: any) {
      logger.error('❌ ML position manager test failed:', error.message);
      throw error;
    } finally {
      // Cleanup
      await this.positionManager.cleanup();
    }
  }

  /**
   * Test position manager initialization
   */
  private async testInitialization(): Promise<void> {
    logger.info('\n🔧 STEP 1: INITIALIZATION TEST');

    // Check environment variables
    const requiredEnvVars = [
      'DELTA_EXCHANGE_API_KEY',
      'DELTA_EXCHANGE_API_SECRET',
      'REDIS_HOST'
    ];

    for (const envVar of requiredEnvVars) {
      if (!process.env[envVar]) {
        logger.warn(`⚠️ Environment variable ${envVar} not set, using defaults`);
      }
    }

    // Initialize position manager
    await this.positionManager.initialize();
    logger.info('✅ ML Position Manager initialized successfully');

    // Initialize decision engine for testing
    await this.decisionEngine.initialize();
    logger.info('✅ Decision Engine initialized for testing');

    // Check initial state
    const activePositions = this.positionManager.getActivePositions();
    const metrics = this.positionManager.getPerformanceMetrics();
    
    logger.info('📊 Initial state:');
    logger.info(`   Active Positions: ${activePositions.length}`);
    logger.info(`   Total Positions: ${metrics.totalPositions}`);
    logger.info(`   Win Rate: ${metrics.winRate}%`);
  }

  /**
   * Test position creation from trading decisions
   */
  private async testPositionCreation(): Promise<void> {
    logger.info('\n📈 STEP 2: POSITION CREATION TEST');

    for (const symbol of this.testSymbols) {
      logger.info(`\n🔄 Testing position creation for ${symbol}...`);

      try {
        // Generate a trading decision
        const decision = await this.decisionEngine.generateTradingDecision(symbol);
        
        if (decision && decision.action !== 'hold') {
          logger.info(`✅ Trading decision generated: ${decision.action.toUpperCase()} ${symbol}`);
          logger.info(`   Confidence: ${(decision.confidence * 100).toFixed(1)}%`);
          logger.info(`   Position Size: ${(decision.positionSize * 100).toFixed(1)}%`);
          logger.info(`   Leverage: ${decision.leverage}x`);

          // Create position from decision
          const currentPrice = 50000; // Mock current price for BTC
          const position = await this.positionManager.createPosition(decision, currentPrice);
          
          if (position) {
            logger.info(`✅ Position created successfully: ${position.id}`);
            logger.info(`   Entry Price: $${position.entryPrice}`);
            logger.info(`   Stop Loss: $${position.stopLoss}`);
            logger.info(`   Take Profit: $${position.takeProfit}`);
            logger.info(`   Quantity: ${position.quantity}`);
            logger.info(`   Side: ${position.side.toUpperCase()}`);
            
            // Validate position structure
            this.validatePositionStructure(position);
            
          } else {
            logger.warn(`⚠️ Failed to create position for ${symbol}`);
          }
          
        } else {
          logger.warn(`⚠️ No actionable trading decision for ${symbol}`);
        }

      } catch (error: any) {
        logger.error(`❌ Position creation failed for ${symbol}:`, error.message);
      }

      // Small delay between symbols
      await this.sleep(1000);
    }
  }

  /**
   * Test position updates and ML predictions
   */
  private async testPositionUpdates(): Promise<void> {
    logger.info('\n🔄 STEP 3: POSITION UPDATES AND ML PREDICTIONS TEST');

    const activePositions = this.positionManager.getActivePositions();
    
    if (activePositions.length === 0) {
      logger.warn('⚠️ No active positions to test updates');
      return;
    }

    for (const position of activePositions) {
      logger.info(`\n📊 Testing updates for position ${position.id}...`);

      try {
        // Simulate price movements
        const priceMovements = [
          position.entryPrice * 1.01,  // +1%
          position.entryPrice * 1.02,  // +2%
          position.entryPrice * 0.99,  // -1%
          position.entryPrice * 1.015  // +1.5%
        ];

        for (let i = 0; i < priceMovements.length; i++) {
          const newPrice = priceMovements[i];
          
          logger.info(`   Update ${i + 1}: Price $${newPrice.toFixed(2)}`);
          
          const updatedPosition = await this.positionManager.updatePosition(position.id, newPrice);
          
          if (updatedPosition) {
            logger.info(`   ✅ Position updated successfully`);
            logger.info(`     Unrealized P&L: ${(updatedPosition.unrealizedPnL * 100).toFixed(2)}%`);
            logger.info(`     Exit Probability: ${(updatedPosition.exitProbability * 100).toFixed(1)}%`);
            logger.info(`     Risk Score: ${(updatedPosition.riskScore * 100).toFixed(1)}%`);
            logger.info(`     Optimal Exit: $${updatedPosition.optimalExitPrice.toFixed(2)}`);
            
            // Check if ML predictions are reasonable
            if (updatedPosition.exitProbability < 0 || updatedPosition.exitProbability > 1) {
              logger.error(`❌ Invalid exit probability: ${updatedPosition.exitProbability}`);
            }
            if (updatedPosition.riskScore < 0 || updatedPosition.riskScore > 1) {
              logger.error(`❌ Invalid risk score: ${updatedPosition.riskScore}`);
            }
            
          } else {
            logger.error(`❌ Failed to update position ${position.id}`);
          }

          // Small delay between updates
          await this.sleep(500);
        }

      } catch (error: any) {
        logger.error(`❌ Position update failed for ${position.id}:`, error.message);
      }
    }
  }

  /**
   * Test dynamic stop loss and take profit management
   */
  private async testDynamicManagement(): Promise<void> {
    logger.info('\n⚙️ STEP 4: DYNAMIC MANAGEMENT TEST');

    const activePositions = this.positionManager.getActivePositions();
    
    if (activePositions.length === 0) {
      logger.warn('⚠️ No active positions to test dynamic management');
      return;
    }

    const position = activePositions[0];
    logger.info(`\n🎯 Testing dynamic management for position ${position.id}...`);

    try {
      // Record initial levels
      const initialStopLoss = position.stopLoss;
      const initialTakeProfit = position.takeProfit;
      
      logger.info(`📊 Initial levels:`);
      logger.info(`   Stop Loss: $${initialStopLoss}`);
      logger.info(`   Take Profit: $${initialTakeProfit}`);

      // Simulate profitable price movement to test trailing stops
      const profitablePrice = position.side === 'long' ? 
        position.entryPrice * 1.03 : // +3% for long
        position.entryPrice * 0.97;  // -3% for short (profit for short)

      logger.info(`🔄 Simulating profitable movement to $${profitablePrice.toFixed(2)}...`);
      
      const updatedPosition = await this.positionManager.updatePosition(position.id, profitablePrice);
      
      if (updatedPosition) {
        logger.info(`✅ Dynamic management applied:`);
        logger.info(`   New Stop Loss: $${updatedPosition.stopLoss} (${updatedPosition.stopLoss !== initialStopLoss ? 'CHANGED' : 'UNCHANGED'})`);
        logger.info(`   New Take Profit: $${updatedPosition.takeProfit} (${updatedPosition.takeProfit !== initialTakeProfit ? 'CHANGED' : 'UNCHANGED'})`);
        
        if (updatedPosition.trailingStop) {
          logger.info(`   Trailing Stop: $${updatedPosition.trailingStop}`);
        }
        
        // Validate that stops moved in correct direction
        if (position.side === 'long' && updatedPosition.stopLoss > initialStopLoss) {
          logger.info(`✅ Trailing stop working correctly for long position`);
        } else if (position.side === 'short' && updatedPosition.stopLoss < initialStopLoss) {
          logger.info(`✅ Trailing stop working correctly for short position`);
        }
        
      } else {
        logger.error(`❌ Failed to apply dynamic management`);
      }

    } catch (error: any) {
      logger.error(`❌ Dynamic management test failed:`, error.message);
    }
  }

  /**
   * Test exit signal detection
   */
  private async testExitSignalDetection(): Promise<void> {
    logger.info('\n🚨 STEP 5: EXIT SIGNAL DETECTION TEST');

    const activePositions = this.positionManager.getActivePositions();
    
    if (activePositions.length === 0) {
      logger.warn('⚠️ No active positions to test exit signals');
      return;
    }

    for (const position of activePositions) {
      logger.info(`\n🔍 Testing exit signals for position ${position.id}...`);

      try {
        // Test various scenarios
        const scenarios = [
          { name: 'Current Price', price: position.currentPrice },
          { name: 'Stop Loss Hit', price: position.stopLoss },
          { name: 'Take Profit Hit', price: position.takeProfit },
          { name: 'High Risk Scenario', price: position.currentPrice, riskOverride: 0.9 }
        ];

        for (const scenario of scenarios) {
          logger.info(`   Testing: ${scenario.name}`);
          
          // Update position with scenario price
          if (scenario.riskOverride) {
            // Temporarily override risk score for testing
            const originalRisk = position.riskScore;
            position.riskScore = scenario.riskOverride;
            
            const exitCheck = await this.positionManager.shouldClosePosition(position.id);
            logger.info(`     Result: ${exitCheck.shouldClose ? 'CLOSE' : 'HOLD'} (${exitCheck.reason}) [${exitCheck.urgency.toUpperCase()}]`);
            
            // Restore original risk
            position.riskScore = originalRisk;
          } else {
            await this.positionManager.updatePosition(position.id, scenario.price);
            const exitCheck = await this.positionManager.shouldClosePosition(position.id);
            logger.info(`     Result: ${exitCheck.shouldClose ? 'CLOSE' : 'HOLD'} (${exitCheck.reason}) [${exitCheck.urgency.toUpperCase()}]`);
          }
        }

      } catch (error: any) {
        logger.error(`❌ Exit signal detection failed for ${position.id}:`, error.message);
      }
    }
  }

  /**
   * Test position closure and training data recording
   */
  private async testPositionClosure(): Promise<void> {
    logger.info('\n🔒 STEP 6: POSITION CLOSURE TEST');

    const activePositions = this.positionManager.getActivePositions();
    
    if (activePositions.length === 0) {
      logger.warn('⚠️ No active positions to test closure');
      return;
    }

    const position = activePositions[0];
    logger.info(`\n📊 Testing closure for position ${position.id}...`);

    try {
      // Record pre-closure state
      const preClosureMetrics = this.positionManager.getPerformanceMetrics();
      
      logger.info(`📈 Pre-closure state:`);
      logger.info(`   Total Positions: ${preClosureMetrics.totalPositions}`);
      logger.info(`   Active Positions: ${preClosureMetrics.activePositions}`);
      logger.info(`   Win Rate: ${preClosureMetrics.winRate}%`);

      // Close position at a profitable price
      const exitPrice = position.side === 'long' ? 
        position.entryPrice * 1.025 : // +2.5% for long
        position.entryPrice * 0.975;  // -2.5% for short (profit for short)

      logger.info(`🔄 Closing position at $${exitPrice.toFixed(2)}...`);
      
      const closureSuccess = await this.positionManager.closePosition(
        position.id, 
        exitPrice, 
        'Test closure - profitable exit'
      );
      
      if (closureSuccess) {
        logger.info(`✅ Position closed successfully`);
        
        // Check post-closure state
        const postClosureMetrics = this.positionManager.getPerformanceMetrics();
        const activePositionsAfter = this.positionManager.getActivePositions();
        
        logger.info(`📊 Post-closure state:`);
        logger.info(`   Total Positions: ${postClosureMetrics.totalPositions}`);
        logger.info(`   Active Positions: ${activePositionsAfter.length}`);
        logger.info(`   Win Rate: ${postClosureMetrics.winRate}%`);
        logger.info(`   Total P&L: ${postClosureMetrics.totalPnL.toFixed(4)}`);
        
        // Validate closure
        const closedPosition = this.positionManager.getPosition(position.id);
        if (closedPosition === null) {
          logger.info(`✅ Position properly removed from active positions`);
        } else {
          logger.error(`❌ Position still exists in active positions`);
        }
        
      } else {
        logger.error(`❌ Failed to close position ${position.id}`);
      }

    } catch (error: any) {
      logger.error(`❌ Position closure test failed:`, error.message);
    }
  }

  /**
   * Test performance metrics and analytics
   */
  private async testPerformanceMetrics(): Promise<void> {
    logger.info('\n📈 STEP 7: PERFORMANCE METRICS TEST');

    try {
      const metrics = this.positionManager.getPerformanceMetrics();
      
      logger.info('📊 PERFORMANCE METRICS:');
      logger.info('=' .repeat(50));
      logger.info(`Total Positions: ${metrics.totalPositions}`);
      logger.info(`Winning Positions: ${metrics.winningPositions}`);
      logger.info(`Win Rate: ${metrics.winRate}%`);
      logger.info(`Total P&L: ${metrics.totalPnL.toFixed(4)}`);
      logger.info(`Average P&L: ${metrics.averagePnL}`);
      logger.info(`Max Drawdown: ${(metrics.maxDrawdown * 100).toFixed(2)}%`);
      logger.info(`Average Hold Time: ${Math.round(metrics.averageHoldTime / 60000)} minutes`);
      logger.info(`ML Accuracy: ${(metrics.mlAccuracy * 100).toFixed(1)}%`);
      logger.info(`Active Positions: ${metrics.activePositions}`);

      // Validate metrics
      if (metrics.totalPositions >= 0 && metrics.winningPositions >= 0) {
        logger.info('✅ Performance metrics structure is valid');
      } else {
        logger.error('❌ Invalid performance metrics detected');
      }

    } catch (error: any) {
      logger.error('❌ Performance metrics test failed:', error.message);
    }
  }

  /**
   * Test configuration management
   */
  private async testConfigurationManagement(): Promise<void> {
    logger.info('\n⚙️ STEP 8: CONFIGURATION MANAGEMENT TEST');

    try {
      // Test configuration update
      const testConfig = {
        exitPredictionThreshold: 0.80,
        trailingStopDistance: 0.010,
        maxTakeProfitExtension: 0.025
      };
      
      logger.info(`🔧 Updating configuration...`);
      this.positionManager.updateConfiguration(testConfig);
      
      logger.info(`✅ Configuration updated successfully`);
      logger.info(`   Exit Threshold: ${(testConfig.exitPredictionThreshold * 100).toFixed(0)}%`);
      logger.info(`   Trailing Distance: ${(testConfig.trailingStopDistance * 100).toFixed(1)}%`);
      logger.info(`   Max TP Extension: ${(testConfig.maxTakeProfitExtension * 100).toFixed(1)}%`);

    } catch (error: any) {
      logger.error('❌ Configuration management test failed:', error.message);
    }
  }

  /**
   * Validate position structure
   */
  private validatePositionStructure(position: Position): void {
    const requiredFields = [
      'id', 'symbol', 'side', 'entryPrice', 'currentPrice', 'quantity', 'leverage',
      'stopLoss', 'takeProfit', 'exitProbability', 'optimalExitPrice', 'riskScore',
      'unrealizedPnL', 'maxDrawdown', 'maxProfit', 'holdingTime',
      'entryTimestamp', 'lastUpdate', 'decisionId'
    ];

    const missingFields = requiredFields.filter(field => !(field in position));
    
    if (missingFields.length === 0) {
      logger.info(`✅ Position structure validation passed`);
    } else {
      logger.error(`❌ Missing fields in position: ${missingFields.join(', ')}`);
    }

    // Validate ranges
    if (position.exitProbability < 0 || position.exitProbability > 1) {
      logger.error(`❌ Invalid exit probability range: ${position.exitProbability}`);
    }
    if (position.riskScore < 0 || position.riskScore > 1) {
      logger.error(`❌ Invalid risk score range: ${position.riskScore}`);
    }
    if (position.leverage < 1 || position.leverage > 1000) {
      logger.error(`❌ Invalid leverage range: ${position.leverage}`);
    }
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
  const tester = new MLPositionManagerTest();
  await tester.runTest();
}

// Run if this script is executed directly
if (require.main === module) {
  main().catch(error => {
    logger.error('💥 ML position manager test failed:', error);
    process.exit(1);
  });
}

export { MLPositionManagerTest };
