#!/usr/bin/env node

/**
 * Enhanced Multi-Timeframe Fibonacci Trading System
 * Main execution script for institutional-grade market structure analysis
 */

const EnhancedFibonacciTrader = require('./enhanced-fibonacci-trading.js');

async function main() {
  console.log(`🚀 INITIALIZING ENHANCED FIBONACCI TRADING SYSTEM`);
  console.log(`════════════════════════════════════════════════════════════════════════════════`);
  console.log(`⚠️  TESTNET MODE - Trades will execute on Delta Exchange testnet`);
  console.log(`📊 Strategy: Multi-Timeframe Fibonacci + Market Structure Analysis`);
  console.log(`🎯 Features: Daily Structure → 4H Bias → 1H Entry + FVG + Order Blocks`);
  console.log(`⚡ Trade Types: Scalping, Day Trading, Swing Trading`);
  console.log(`════════════════════════════════════════════════════════════════════════════════`);

  try {
    // Initialize the enhanced trading system
    const trader = new EnhancedFibonacciTrader();
    
    console.log(`🔧 Initializing Delta Exchange connection...`);
    const initialized = await trader.initialize();
    
    if (!initialized) {
      console.error(`❌ Failed to initialize trading system`);
      process.exit(1);
    }

    console.log(`✅ Enhanced Fibonacci Trading System initialized successfully`);
    console.log(`🎯 Starting live trading with institutional-grade analysis...`);

    // Start the enhanced trading loop
    await trader.startEnhancedTrading();

  } catch (error) {
    console.error(`❌ Fatal error in Enhanced Fibonacci Trading System: ${error.message}`);
    console.error(error.stack);
    process.exit(1);
  }
}

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log(`\n🛑 Shutting down Enhanced Fibonacci Trading System...`);
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log(`\n🛑 Shutting down Enhanced Fibonacci Trading System...`);
  process.exit(0);
});

// Start the system
main().catch(error => {
  console.error(`❌ Unhandled error: ${error.message}`);
  process.exit(1);
});
