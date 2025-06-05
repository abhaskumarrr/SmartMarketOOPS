#!/usr/bin/env node

/**
 * Real Account Balance Checker
 * Fetches and displays current account balance from Delta Exchange
 */

import dotenv from 'dotenv';
import { DeltaExchangeUnified, DeltaBalance, DeltaPosition } from '../services/DeltaExchangeUnified';
import { logger } from '../utils/logger';

// Load environment variables
dotenv.config();

async function checkRealBalance(): Promise<void> {
  try {
    logger.info('🔍 CHECKING REAL ACCOUNT BALANCE ON DELTA EXCHANGE');
    logger.info('=' .repeat(80));
    
    // Initialize Delta Exchange service with credentials from .env
    const credentials = {
      apiKey: process.env.DELTA_EXCHANGE_API_KEY || '',
      apiSecret: process.env.DELTA_EXCHANGE_API_SECRET || '',
      testnet: process.env.DELTA_EXCHANGE_TESTNET === 'true' // Use testnet setting from .env
    };

    if (!credentials.apiKey || !credentials.apiSecret) {
      throw new Error('Delta Exchange API credentials not found in environment variables');
    }

    logger.info(`🔧 Using ${credentials.testnet ? 'TESTNET' : 'LIVE'} Delta Exchange environment`);

    const deltaService = new DeltaExchangeUnified(credentials);
    
    // Wait for service to initialize
    logger.info('⏳ Connecting to Delta Exchange...');
    let retries = 0;
    while (!deltaService.isReady() && retries < 10) {
      await new Promise(resolve => setTimeout(resolve, 2000));
      retries++;
    }

    if (!deltaService.isReady()) {
      throw new Error('Failed to connect to Delta Exchange');
    }

    logger.info('✅ Connected to Delta Exchange successfully');
    logger.info('');

    // Fetch account balance
    logger.info('💰 Fetching account balance...');
    const balances: DeltaBalance[] = await deltaService.getBalance();
    
    // Display all balances
    logger.info('📊 ACCOUNT BALANCES:');
    logger.info('-'.repeat(60));
    
    let totalUSDValue = 0;
    const significantBalances = balances.filter(balance => 
      parseFloat(balance.balance) > 0.01 || parseFloat(balance.available_balance) > 0.01
    );

    if (significantBalances.length === 0) {
      logger.info('   No significant balances found');
    } else {
      for (const balance of significantBalances) {
        const total = parseFloat(balance.balance);
        const available = parseFloat(balance.available_balance);
        const margin = parseFloat(balance.position_margin);
        const unrealizedPnl = parseFloat(balance.unrealized_pnl);
        
        logger.info(`   ${balance.asset_symbol}:`);
        logger.info(`     💰 Total Balance: ${total.toFixed(6)}`);
        logger.info(`     💵 Available: ${available.toFixed(6)}`);
        logger.info(`     📊 Position Margin: ${margin.toFixed(6)}`);
        logger.info(`     📈 Unrealized P&L: ${unrealizedPnl.toFixed(6)}`);
        logger.info('');
        
        // Estimate USD value (assuming USDT/USD are 1:1)
        if (balance.asset_symbol === 'USDT' || balance.asset_symbol === 'USD') {
          totalUSDValue += total;
        }
      }
    }

    // Fetch current positions (handle gracefully if none exist)
    logger.info('📈 Fetching current positions...');
    let positions: DeltaPosition[] = [];
    try {
      positions = await deltaService.getPositions();
    } catch (error) {
      logger.warn('⚠️ Could not fetch positions (likely no active positions)');
      positions = [];
    }
    
    logger.info('📊 CURRENT POSITIONS:');
    logger.info('-'.repeat(60));
    
    const activePositions = positions.filter(pos => Math.abs(pos.size) > 0);
    
    if (activePositions.length === 0) {
      logger.info('   No active positions');
    } else {
      for (const position of activePositions) {
        const side = position.size > 0 ? 'LONG' : 'SHORT';
        const size = Math.abs(position.size);
        const entryPrice = parseFloat(position.entry_price);
        const margin = parseFloat(position.margin);
        
        logger.info(`   ${position.product.symbol}:`);
        logger.info(`     📊 Side: ${side}`);
        logger.info(`     📏 Size: ${size}`);
        logger.info(`     💰 Entry Price: $${entryPrice.toFixed(2)}`);
        logger.info(`     💵 Margin Used: $${margin.toFixed(2)}`);
        logger.info('');
      }
    }

    // Calculate trading capacity
    const mainBalance = significantBalances.find(b => 
      b.asset_symbol === 'USDT' || b.asset_symbol === 'USD'
    );

    if (mainBalance) {
      const availableBalance = parseFloat(mainBalance.available_balance);
      const maxLeverage = 3; // Conservative leverage
      const buyingPower = availableBalance * maxLeverage;
      const recommendedTradeSize = availableBalance * 0.02; // 2% risk per trade
      
      logger.info('💪 TRADING CAPACITY:');
      logger.info('-'.repeat(60));
      logger.info(`   💵 Available Balance: $${availableBalance.toFixed(2)}`);
      logger.info(`   ⚡ Buying Power (3x): $${buyingPower.toFixed(2)}`);
      logger.info(`   🎯 Recommended Trade Size (2% risk): $${recommendedTradeSize.toFixed(2)}`);
      logger.info(`   📊 Estimated Trades Possible: ${Math.floor(buyingPower / recommendedTradeSize)}`);
      
      // Trading readiness assessment
      logger.info('');
      logger.info('🔍 TRADING READINESS ASSESSMENT:');
      logger.info('-'.repeat(60));
      
      if (availableBalance >= 100) {
        logger.info('   ✅ Sufficient balance for live trading');
        logger.info('   ✅ Ready to start automated trading bot');
      } else if (availableBalance >= 50) {
        logger.info('   ⚠️  Low balance - consider smaller position sizes');
        logger.info('   ⚠️  Reduce risk per trade to 1%');
      } else {
        logger.info('   ❌ Insufficient balance for safe trading');
        logger.info('   ❌ Consider depositing more funds');
      }
      
      if (activePositions.length > 0) {
        logger.info('   ⚠️  Active positions detected - bot will manage them');
      } else {
        logger.info('   ✅ No active positions - clean slate for new trades');
      }
    }

    logger.info('');
    logger.info('✅ Account balance check completed');
    
  } catch (error) {
    logger.error('❌ Failed to check account balance:', error);
    
    if (error instanceof Error) {
      if (error.message.includes('credentials')) {
        logger.error('💡 Make sure DELTA_EXCHANGE_API_KEY and DELTA_EXCHANGE_API_SECRET are set in your .env file');
      } else if (error.message.includes('authentication')) {
        logger.error('💡 Check that your API credentials are valid and have trading permissions');
      } else if (error.message.includes('network') || error.message.includes('timeout')) {
        logger.error('💡 Check your internet connection and try again');
      }
    }
    
    process.exit(1);
  }
}

// Run the balance check
if (require.main === module) {
  checkRealBalance().catch(error => {
    logger.error('❌ Script execution failed:', error);
    process.exit(1);
  });
}

export { checkRealBalance };
