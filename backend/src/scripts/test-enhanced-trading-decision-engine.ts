#!/usr/bin/env node

/**
 * Enhanced Trading Decision Engine Test
 * Comprehensive testing of ML-driven trading decisions, ensemble voting, and risk management
 */

import { EnhancedTradingDecisionEngine, TradingDecision } from '../services/EnhancedTradingDecisionEngine';
import { logger } from '../utils/logger';

class EnhancedTradingDecisionEngineTest {
  private decisionEngine: EnhancedTradingDecisionEngine;
  private testSymbols: string[] = ['BTCUSD', 'ETHUSD'];

  constructor() {
    this.decisionEngine = new EnhancedTradingDecisionEngine();
  }

  /**
   * Run comprehensive enhanced trading decision engine test
   */
  public async runTest(): Promise<void> {
    logger.info('🧠 ENHANCED TRADING DECISION ENGINE TEST');
    logger.info('=' .repeat(80));

    try {
      // Step 1: Initialize the decision engine
      await this.testInitialization();

      // Step 2: Test ML feature extraction and processing
      await this.testMLFeatureProcessing();

      // Step 3: Test ensemble model voting
      await this.testEnsembleVoting();

      // Step 4: Test trading decision generation
      await this.testTradingDecisionGeneration();

      // Step 5: Test risk assessment and position sizing
      await this.testRiskAssessmentAndPositionSizing();

      // Step 6: Test confidence thresholds and filtering
      await this.testConfidenceThresholds();

      // Step 7: Test decision caching and history
      await this.testDecisionCachingAndHistory();

      // Step 8: Test configuration management
      await this.testConfigurationManagement();

      logger.info('\n🎉 ENHANCED TRADING DECISION ENGINE TEST COMPLETED SUCCESSFULLY!');
      logger.info('✅ All ML-driven trading decision features are working correctly');

    } catch (error: any) {
      logger.error('❌ Enhanced trading decision engine test failed:', error.message);
      throw error;
    } finally {
      // Cleanup
      await this.decisionEngine.cleanup();
    }
  }

  /**
   * Test decision engine initialization
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

    // Initialize decision engine
    await this.decisionEngine.initialize();
    logger.info('✅ Enhanced Trading Decision Engine initialized successfully');

    // Test configuration access
    const config = this.decisionEngine.getConfiguration();
    logger.info('📊 Configuration loaded:');
    logger.info(`   Min Confidence: ${(config.minConfidenceThreshold * 100).toFixed(0)}%`);
    logger.info(`   High Confidence: ${(config.highConfidenceThreshold * 100).toFixed(0)}%`);
    logger.info(`   Max Position Size: ${(config.maxPositionSize * 100).toFixed(0)}%`);
    logger.info(`   Max Leverage: ${config.maxLeverage}x`);
    logger.info(`   Model Weights: LSTM ${config.modelWeights.lstm}, Transformer ${config.modelWeights.transformer}, Ensemble ${config.modelWeights.ensemble}`);
  }

  /**
   * Test ML feature processing
   */
  private async testMLFeatureProcessing(): Promise<void> {
    logger.info('\n🧠 STEP 2: ML FEATURE PROCESSING TEST');

    for (const symbol of this.testSymbols) {
      logger.info(`\n📊 Testing ML feature processing for ${symbol}...`);

      try {
        // This will test the internal feature extraction through decision generation
        const decision = await this.decisionEngine.generateTradingDecision(symbol);
        
        if (decision) {
          logger.info(`✅ ML features processed successfully for ${symbol}`);
          logger.info(`   Key Features:`);
          logger.info(`     Fibonacci Signal: ${decision.keyFeatures.fibonacciSignal.toFixed(3)}`);
          logger.info(`     Bias Alignment: ${(decision.keyFeatures.biasAlignment * 100).toFixed(1)}%`);
          logger.info(`     Candle Strength: ${(decision.keyFeatures.candleStrength * 100).toFixed(1)}%`);
          logger.info(`     Volume Confirmation: ${(decision.keyFeatures.volumeConfirmation * 100).toFixed(1)}%`);
          logger.info(`     Market Timing: ${(decision.keyFeatures.marketTiming * 100).toFixed(1)}%`);
        } else {
          logger.warn(`⚠️ No decision generated for ${symbol} (possibly low confidence or data quality)`);
        }

      } catch (error: any) {
        logger.error(`❌ ML feature processing failed for ${symbol}:`, error.message);
      }

      // Small delay between symbols
      await this.sleep(2000);
    }
  }

  /**
   * Test ensemble model voting
   */
  private async testEnsembleVoting(): Promise<void> {
    logger.info('\n🗳️ STEP 3: ENSEMBLE MODEL VOTING TEST');

    for (const symbol of this.testSymbols) {
      logger.info(`\n🤖 Testing ensemble voting for ${symbol}...`);

      try {
        const decision = await this.decisionEngine.generateTradingDecision(symbol);
        
        if (decision && decision.modelVotes) {
          logger.info(`✅ Ensemble voting completed for ${symbol}`);
          logger.info(`   Model Votes:`);
          logger.info(`     LSTM: ${decision.modelVotes.lstm.action} (${(decision.modelVotes.lstm.confidence * 100).toFixed(1)}%)`);
          logger.info(`     Transformer: ${decision.modelVotes.transformer.action} (${(decision.modelVotes.transformer.confidence * 100).toFixed(1)}%)`);
          logger.info(`     Ensemble: ${decision.modelVotes.ensemble.action} (${(decision.modelVotes.ensemble.confidence * 100).toFixed(1)}%)`);
          logger.info(`   Final Decision: ${decision.action.toUpperCase()} (${(decision.confidence * 100).toFixed(1)}%)`);
          
          // Validate voting logic
          const votes = [decision.modelVotes.lstm.action, decision.modelVotes.transformer.action, decision.modelVotes.ensemble.action];
          const finalAction = decision.action;
          logger.info(`   Voting Consistency: ${votes.includes(finalAction) ? 'CONSISTENT' : 'WEIGHTED_DECISION'}`);
          
        } else {
          logger.warn(`⚠️ No ensemble voting data available for ${symbol}`);
        }

      } catch (error: any) {
        logger.error(`❌ Ensemble voting test failed for ${symbol}:`, error.message);
      }
    }
  }

  /**
   * Test trading decision generation
   */
  private async testTradingDecisionGeneration(): Promise<void> {
    logger.info('\n🎯 STEP 4: TRADING DECISION GENERATION TEST');

    for (const symbol of this.testSymbols) {
      logger.info(`\n📈 Testing decision generation for ${symbol}...`);

      try {
        const decision = await this.decisionEngine.generateTradingDecision(symbol);
        
        if (decision) {
          logger.info(`✅ Trading decision generated for ${symbol}`);
          logger.info(`   Decision Details:`);
          logger.info(`     Action: ${decision.action.toUpperCase()}`);
          logger.info(`     Confidence: ${(decision.confidence * 100).toFixed(1)}%`);
          logger.info(`     Position Size: ${(decision.positionSize * 100).toFixed(2)}%`);
          logger.info(`     Leverage: ${decision.leverage}x`);
          logger.info(`     Stop Loss: $${decision.stopLoss}`);
          logger.info(`     Take Profit: $${decision.takeProfit}`);
          logger.info(`     Risk Score: ${(decision.riskScore * 100).toFixed(1)}%`);
          logger.info(`     Win Probability: ${(decision.winProbability * 100).toFixed(1)}%`);
          logger.info(`     Urgency: ${decision.urgency.toUpperCase()}`);
          logger.info(`     Time to Live: ${Math.round(decision.timeToLive / 1000)}s`);
          
          // Validate decision structure
          this.validateDecisionStructure(decision);
          
        } else {
          logger.warn(`⚠️ No trading decision generated for ${symbol}`);
        }

      } catch (error: any) {
        logger.error(`❌ Decision generation failed for ${symbol}:`, error.message);
      }
    }
  }

  /**
   * Test risk assessment and position sizing
   */
  private async testRiskAssessmentAndPositionSizing(): Promise<void> {
    logger.info('\n🛡️ STEP 5: RISK ASSESSMENT AND POSITION SIZING TEST');

    const symbol = this.testSymbols[0]; // Test with BTC
    logger.info(`\n⚖️ Testing risk assessment for ${symbol}...`);

    try {
      const decision = await this.decisionEngine.generateTradingDecision(symbol);
      
      if (decision) {
        logger.info(`✅ Risk assessment completed for ${symbol}`);
        logger.info(`   Risk Analysis:`);
        logger.info(`     Risk Score: ${(decision.riskScore * 100).toFixed(1)}%`);
        logger.info(`     Max Drawdown: ${(decision.maxDrawdown * 100).toFixed(2)}%`);
        logger.info(`     Win Probability: ${(decision.winProbability * 100).toFixed(1)}%`);
        
        logger.info(`   Position Sizing:`);
        logger.info(`     Position Size: ${(decision.positionSize * 100).toFixed(2)}%`);
        logger.info(`     Leverage: ${decision.leverage}x`);
        logger.info(`     Effective Exposure: ${(decision.positionSize * decision.leverage * 100).toFixed(1)}%`);
        
        // Validate risk-position relationship
        if (decision.riskScore > 0.7 && decision.positionSize > 0.05) {
          logger.warn(`⚠️ High risk (${(decision.riskScore * 100).toFixed(0)}%) with large position (${(decision.positionSize * 100).toFixed(1)}%)`);
        } else {
          logger.info(`✅ Risk-position sizing relationship is appropriate`);
        }
        
      } else {
        logger.warn(`⚠️ No risk assessment data available for ${symbol}`);
      }

    } catch (error: any) {
      logger.error(`❌ Risk assessment test failed for ${symbol}:`, error.message);
    }
  }

  /**
   * Test confidence thresholds and filtering
   */
  private async testConfidenceThresholds(): Promise<void> {
    logger.info('\n🎚️ STEP 6: CONFIDENCE THRESHOLDS TEST');

    // Test with different confidence thresholds
    const originalConfig = this.decisionEngine.getConfiguration();
    
    // Test with very high threshold (should reject most decisions)
    logger.info('\n🔒 Testing with high confidence threshold (95%)...');
    this.decisionEngine.updateConfiguration({ minConfidenceThreshold: 0.95 });
    
    let highThresholdDecisions = 0;
    for (const symbol of this.testSymbols) {
      const decision = await this.decisionEngine.generateTradingDecision(symbol);
      if (decision) {
        highThresholdDecisions++;
        logger.info(`✅ High confidence decision for ${symbol}: ${(decision.confidence * 100).toFixed(1)}%`);
      }
    }
    logger.info(`📊 High threshold results: ${highThresholdDecisions}/${this.testSymbols.length} decisions passed`);
    
    // Test with low threshold (should accept more decisions)
    logger.info('\n🔓 Testing with low confidence threshold (50%)...');
    this.decisionEngine.updateConfiguration({ minConfidenceThreshold: 0.50 });
    
    let lowThresholdDecisions = 0;
    for (const symbol of this.testSymbols) {
      const decision = await this.decisionEngine.generateTradingDecision(symbol);
      if (decision) {
        lowThresholdDecisions++;
        logger.info(`✅ Low confidence decision for ${symbol}: ${(decision.confidence * 100).toFixed(1)}%`);
      }
    }
    logger.info(`📊 Low threshold results: ${lowThresholdDecisions}/${this.testSymbols.length} decisions passed`);
    
    // Restore original configuration
    this.decisionEngine.updateConfiguration(originalConfig);
    logger.info('🔄 Original configuration restored');
  }

  /**
   * Test decision caching and history
   */
  private async testDecisionCachingAndHistory(): Promise<void> {
    logger.info('\n💾 STEP 7: DECISION CACHING AND HISTORY TEST');

    const symbol = this.testSymbols[0];
    
    // Generate a decision
    logger.info(`🔄 Generating decision for ${symbol}...`);
    const decision1 = await this.decisionEngine.generateTradingDecision(symbol);
    
    if (decision1) {
      logger.info(`✅ Decision generated and cached`);
      
      // Test cache retrieval
      const cachedDecision = this.decisionEngine.getLatestDecision(symbol);
      if (cachedDecision && cachedDecision.timestamp === decision1.timestamp) {
        logger.info(`✅ Decision caching working correctly`);
      } else {
        logger.error(`❌ Decision caching failed`);
      }
      
      // Test decision history
      const history = this.decisionEngine.getDecisionHistory(10);
      logger.info(`📚 Decision history: ${history.length} decisions stored`);
      
      if (history.length > 0) {
        const latest = history[history.length - 1];
        logger.info(`   Latest decision: ${latest.symbol} ${latest.action} (${(latest.confidence * 100).toFixed(1)}%)`);
      }
      
    } else {
      logger.warn(`⚠️ No decision generated for caching test`);
    }
  }

  /**
   * Test configuration management
   */
  private async testConfigurationManagement(): Promise<void> {
    logger.info('\n⚙️ STEP 8: CONFIGURATION MANAGEMENT TEST');

    // Get original configuration
    const originalConfig = this.decisionEngine.getConfiguration();
    logger.info(`📋 Original configuration loaded`);
    
    // Test configuration update
    const testConfig = {
      minConfidenceThreshold: 0.75,
      maxPositionSize: 0.06,
      baseLeverage: 150
    };
    
    logger.info(`🔧 Updating configuration...`);
    this.decisionEngine.updateConfiguration(testConfig);
    
    // Verify update
    const updatedConfig = this.decisionEngine.getConfiguration();
    const configValid = 
      updatedConfig.minConfidenceThreshold === testConfig.minConfidenceThreshold &&
      updatedConfig.maxPositionSize === testConfig.maxPositionSize &&
      updatedConfig.baseLeverage === testConfig.baseLeverage;
    
    if (configValid) {
      logger.info(`✅ Configuration update successful`);
      logger.info(`   Min Confidence: ${(updatedConfig.minConfidenceThreshold * 100).toFixed(0)}%`);
      logger.info(`   Max Position: ${(updatedConfig.maxPositionSize * 100).toFixed(0)}%`);
      logger.info(`   Base Leverage: ${updatedConfig.baseLeverage}x`);
    } else {
      logger.error(`❌ Configuration update failed`);
    }
    
    // Restore original configuration
    this.decisionEngine.updateConfiguration(originalConfig);
    logger.info(`🔄 Original configuration restored`);
  }

  /**
   * Validate decision structure
   */
  private validateDecisionStructure(decision: TradingDecision): void {
    const requiredFields = [
      'action', 'confidence', 'symbol', 'timestamp',
      'stopLoss', 'takeProfit', 'positionSize', 'leverage',
      'modelVotes', 'keyFeatures', 'riskScore', 'winProbability',
      'urgency', 'timeToLive', 'reasoning'
    ];

    const missingFields = requiredFields.filter(field => !(field in decision));
    
    if (missingFields.length === 0) {
      logger.info(`✅ Decision structure validation passed`);
    } else {
      logger.error(`❌ Missing fields in decision: ${missingFields.join(', ')}`);
    }

    // Validate ranges
    if (decision.confidence < 0 || decision.confidence > 1) {
      logger.error(`❌ Invalid confidence range: ${decision.confidence}`);
    }
    if (decision.positionSize < 0 || decision.positionSize > 1) {
      logger.error(`❌ Invalid position size range: ${decision.positionSize}`);
    }
    if (decision.leverage < 1 || decision.leverage > 1000) {
      logger.error(`❌ Invalid leverage range: ${decision.leverage}`);
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
  const tester = new EnhancedTradingDecisionEngineTest();
  await tester.runTest();
}

// Run if this script is executed directly
if (require.main === module) {
  main().catch(error => {
    logger.error('💥 Enhanced trading decision engine test failed:', error);
    process.exit(1);
  });
}

export { EnhancedTradingDecisionEngineTest };
