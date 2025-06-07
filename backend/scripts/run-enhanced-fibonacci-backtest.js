#!/usr/bin/env node

/**
 * Enhanced Fibonacci Trading Strategy Backtest Runner
 * 
 * This script runs a comprehensive backtest of the Enhanced Fibonacci Trading System
 * with multi-timeframe analysis, institutional-grade Fibonacci levels, and advanced
 * risk management.
 * 
 * Features:
 * - Multi-timeframe confluence analysis (Daily → 4H → 15M → 5M)
 * - Realistic swing detection (30-day significant swings)
 * - Institutional Fibonacci levels (23.6%, 38.2%, 50%, 61.8%, 78.6%)
 * - Dynamic trade classification (Scalping, Day Trading, Swing Trading)
 * - Advanced risk management with position sizing
 * - Comprehensive performance analytics
 */

require('dotenv').config();
const EnhancedFibonacciBacktest = require('./enhanced-fibonacci-backtest');

async function main() {
  console.log(`🔬 ENHANCED FIBONACCI TRADING STRATEGY BACKTEST`);
  console.log(`════════════════════════════════════════════════════════════════════════════════`);
  console.log(`🎯 Strategy: Multi-Timeframe Fibonacci + Market Structure Analysis`);
  console.log(`📊 Features: Daily Structure → 4H Bias → 15M Entry → 5M Scalping`);
  console.log(`⚡ Trade Types: Scalping (25x), Day Trading (15x), Swing Trading (10x)`);
  console.log(`🎛️ Risk Management: 2% per trade, 75% confluence threshold`);
  console.log(`════════════════════════════════════════════════════════════════════════════════`);

  try {
    // Initialize backtest
    const backtest = new EnhancedFibonacciBacktest();
    
    console.log(`\n🔧 Initializing backtest environment...`);
    const initialized = await backtest.initialize();
    
    if (!initialized) {
      console.error(`❌ Failed to initialize backtest environment`);
      process.exit(1);
    }

    console.log(`\n🚀 Starting comprehensive backtest...`);
    console.log(`⏱️ This may take several minutes to complete...`);
    
    // Run the backtest
    const results = await backtest.runBacktest();
    
    if (!results) {
      console.error(`❌ Backtest failed to complete`);
      process.exit(1);
    }

    console.log(`\n✅ BACKTEST COMPLETED SUCCESSFULLY`);
    console.log(`📊 Results saved and displayed above`);
    
    // Additional analysis
    console.log(`\n🔍 ADDITIONAL INSIGHTS:`);
    
    if (results.totalTrades === 0) {
      console.log(`⚠️ No trades were executed during the backtest period`);
      console.log(`💡 Consider lowering the confluence threshold or adjusting parameters`);
    } else {
      console.log(`📈 Average trade PnL: $${(results.totalPnL / results.totalTrades).toFixed(2)}`);
      console.log(`⏱️ Average trades per day: ${(results.totalTrades / 30).toFixed(1)}`);
      
      if (results.winRate > 65 && results.totalReturn > 15) {
        console.log(`🎉 OUTSTANDING PERFORMANCE! Strategy shows strong potential`);
      } else if (results.winRate > 55 && results.totalReturn > 5) {
        console.log(`👍 GOOD PERFORMANCE! Strategy shows promise with optimization`);
      } else {
        console.log(`⚠️ NEEDS IMPROVEMENT! Consider parameter adjustments`);
      }
    }

    console.log(`\n🎯 NEXT STEPS:`);
    console.log(`1. Analyze Fibonacci level performance for optimization`);
    console.log(`2. Review trade type distribution and adjust parameters`);
    console.log(`3. Consider live testing with small position sizes`);
    console.log(`4. Monitor confluence threshold effectiveness`);
    
  } catch (error) {
    console.error(`❌ Backtest error: ${error.message}`);
    console.error(error.stack);
    process.exit(1);
  }
}

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log(`\n⚠️ Backtest interrupted by user`);
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log(`\n⚠️ Backtest terminated`);
  process.exit(0);
});

// Run the backtest
main().catch(error => {
  console.error(`❌ Fatal error: ${error.message}`);
  process.exit(1);
});
