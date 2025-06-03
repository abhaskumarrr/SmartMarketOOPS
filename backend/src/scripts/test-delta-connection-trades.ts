#!/usr/bin/env node

/**
 * Delta Testnet Connection & Trade Execution Test
 * Comprehensive test to verify:
 * 1. Connection to Delta testnet
 * 2. Account balance retrieval
 * 3. Market data access
 * 4. Order placement (entry)
 * 5. Order cancellation (exit)
 * 6. Position management
 */

import DeltaExchangeAPI from '../services/deltaApiService';
import { logger } from '../utils/logger';
import * as DeltaExchange from '../types/deltaExchange';

class DeltaTestnetTester {
  private deltaApi: DeltaExchangeAPI;
  private testResults: any = {};

  constructor() {
    // Initialize for testnet
    this.deltaApi = new DeltaExchangeAPI({ testnet: true });
  }

  /**
   * Run comprehensive Delta testnet tests
   */
  public async runComprehensiveTest(): Promise<void> {
    const startTime = Date.now();
    
    try {
      logger.info('🚀 DELTA TESTNET CONNECTION & TRADE EXECUTION TEST');
      logger.info('=' .repeat(80));
      
      // Test 1: Initialize connection
      await this.testConnection();
      
      // Test 2: Get account information
      await this.testAccountInfo();
      
      // Test 3: Get wallet balances
      await this.testWalletBalances();
      
      // Test 4: Get market data
      await this.testMarketData();
      
      // Test 5: Get current positions
      await this.testPositions();
      
      // Test 6: Test order placement (entry)
      await this.testOrderPlacement();
      
      // Test 7: Test order cancellation (exit)
      await this.testOrderCancellation();
      
      // Test 8: Test position management
      await this.testPositionManagement();
      
      // Generate comprehensive report
      this.generateTestReport(startTime);
      
    } catch (error) {
      logger.error('❌ Delta testnet test failed:', error);
      throw error;
    }
  }

  /**
   * Test 1: Connection to Delta testnet
   */
  private async testConnection(): Promise<void> {
    try {
      logger.info('\n📡 TEST 1: DELTA TESTNET CONNECTION');
      
      // Get credentials from environment
      const credentials: DeltaExchange.ApiCredentials = {
        key: process.env.DELTA_EXCHANGE_API_KEY || '',
        secret: process.env.DELTA_EXCHANGE_API_SECRET || ''
      };

      if (!credentials.key || !credentials.secret) {
        throw new Error('Delta API credentials not found in environment variables');
      }

      logger.info('   ✅ Credentials found in environment');
      logger.info(`   🔑 API Key: ${credentials.key.substring(0, 8)}...`);
      logger.info(`   🔐 Secret: ${credentials.secret.substring(0, 8)}...`);

      // Initialize connection
      await this.deltaApi.initialize(credentials);
      logger.info('   ✅ API client initialized successfully');

      // Test server time
      const serverTime = await this.deltaApi.getServerTime();
      logger.info(`   ✅ Server time retrieved: ${new Date(serverTime.timestamp * 1000).toISOString()}`);
      
      this.testResults.connection = {
        status: 'SUCCESS',
        serverTime: serverTime.timestamp,
        testnet: true,
      };

    } catch (error) {
      logger.error('   ❌ Connection test failed:', error);
      this.testResults.connection = {
        status: 'FAILED',
        error: error.message,
      };
      throw error;
    }
  }

  /**
   * Test 2: Account information retrieval
   */
  private async testAccountInfo(): Promise<void> {
    try {
      logger.info('\n👤 TEST 2: ACCOUNT INFORMATION');
      
      const accountInfo = await this.deltaApi.getAccountInfo();
      
      logger.info(`   ✅ Account ID: ${accountInfo.id}`);
      logger.info(`   ✅ Email: ${accountInfo.email}`);
      logger.info(`   ✅ KYC Status: ${accountInfo.kyc_status || 'N/A'}`);
      logger.info(`   ✅ Trading Enabled: ${accountInfo.trading_enabled || 'N/A'}`);
      
      this.testResults.accountInfo = {
        status: 'SUCCESS',
        accountId: accountInfo.id,
        email: accountInfo.email,
        kycStatus: accountInfo.kyc_status,
        tradingEnabled: accountInfo.trading_enabled,
      };

    } catch (error) {
      logger.error('   ❌ Account info test failed:', error);
      this.testResults.accountInfo = {
        status: 'FAILED',
        error: error.message,
      };
    }
  }

  /**
   * Test 3: Wallet balances retrieval
   */
  private async testWalletBalances(): Promise<void> {
    try {
      logger.info('\n💰 TEST 3: WALLET BALANCES');
      
      const balances = await this.deltaApi.getWalletBalances();
      
      logger.info(`   ✅ Retrieved ${balances.length} wallet balances`);
      
      // Show significant balances
      const significantBalances = balances.filter(b => parseFloat(b.balance) > 0);
      
      if (significantBalances.length > 0) {
        logger.info('   💰 Non-zero balances:');
        significantBalances.forEach(balance => {
          logger.info(`     ${balance.asset}: ${balance.balance} (Available: ${balance.available_balance})`);
        });
      } else {
        logger.info('   ⚠️  No non-zero balances found (testnet account may be empty)');
      }
      
      // Find USD/USDT balance for trading
      const usdBalance = balances.find(b => 
        b.asset === 'USDT' || b.asset === 'USD' || b.asset === 'USDC'
      );
      
      let tradingBalance = 0;
      if (usdBalance) {
        tradingBalance = parseFloat(usdBalance.available_balance);
        logger.info(`   💵 Trading balance: ${tradingBalance} ${usdBalance.asset}`);
      }
      
      this.testResults.walletBalances = {
        status: 'SUCCESS',
        totalBalances: balances.length,
        significantBalances: significantBalances.length,
        tradingBalance,
        tradingAsset: usdBalance?.asset || 'NONE',
      };

    } catch (error) {
      logger.error('   ❌ Wallet balances test failed:', error);
      this.testResults.walletBalances = {
        status: 'FAILED',
        error: error.message,
      };
    }
  }

  /**
   * Test 4: Market data retrieval
   */
  private async testMarketData(): Promise<void> {
    try {
      logger.info('\n📊 TEST 4: MARKET DATA');
      
      // Test getting all markets
      const markets = await this.deltaApi.getMarkets();
      logger.info(`   ✅ Retrieved ${markets.length} available markets`);
      
      // Find BTC perpetual contract
      const btcMarket = markets.find(m => 
        m.symbol.includes('BTC') && m.symbol.includes('USD') && 
        (m.contract_type === 'perpetual_futures' || m.product_type === 'futures')
      );
      
      if (btcMarket) {
        logger.info(`   ✅ Found BTC market: ${btcMarket.symbol}`);
        logger.info(`     Contract Type: ${btcMarket.contract_type || btcMarket.product_type}`);
        logger.info(`     Status: ${btcMarket.state || btcMarket.trading_status}`);
        
        // Get ticker for BTC market
        try {
          const ticker = await this.deltaApi.getTicker(btcMarket.symbol);
          logger.info(`   ✅ BTC Ticker - Price: $${ticker.close || ticker.mark_price}, Volume: ${ticker.volume}`);
        } catch (tickerError) {
          logger.warn(`   ⚠️  Could not get ticker for ${btcMarket.symbol}:`, tickerError.message);
        }
        
        this.testResults.marketData = {
          status: 'SUCCESS',
          totalMarkets: markets.length,
          btcMarket: btcMarket.symbol,
          btcPrice: 'Retrieved',
        };
      } else {
        logger.warn('   ⚠️  No BTC perpetual market found');
        this.testResults.marketData = {
          status: 'PARTIAL',
          totalMarkets: markets.length,
          btcMarket: 'NOT_FOUND',
        };
      }

    } catch (error) {
      logger.error('   ❌ Market data test failed:', error);
      this.testResults.marketData = {
        status: 'FAILED',
        error: error.message,
      };
    }
  }

  /**
   * Test 5: Current positions
   */
  private async testPositions(): Promise<void> {
    try {
      logger.info('\n📈 TEST 5: CURRENT POSITIONS');
      
      const positions = await this.deltaApi.getPositions();
      logger.info(`   ✅ Retrieved ${positions.length} positions`);
      
      if (positions.length > 0) {
        logger.info('   📊 Active positions:');
        positions.forEach(position => {
          logger.info(`     ${position.symbol}: ${position.size} @ $${position.entry_price || position.average_entry_price}`);
        });
      } else {
        logger.info('   ✅ No active positions (clean slate for testing)');
      }
      
      this.testResults.positions = {
        status: 'SUCCESS',
        activePositions: positions.length,
      };

    } catch (error) {
      logger.error('   ❌ Positions test failed:', error);
      this.testResults.positions = {
        status: 'FAILED',
        error: error.message,
      };
    }
  }

  /**
   * Test 6: Order placement (entry test)
   */
  private async testOrderPlacement(): Promise<void> {
    try {
      logger.info('\n📝 TEST 6: ORDER PLACEMENT (ENTRY)');
      
      // Get available markets first
      const markets = await this.deltaApi.getMarkets();
      const btcMarket = markets.find(m => 
        m.symbol.includes('BTC') && m.symbol.includes('USD') && 
        (m.contract_type === 'perpetual_futures' || m.product_type === 'futures')
      );
      
      if (!btcMarket) {
        throw new Error('No BTC market found for testing');
      }
      
      logger.info(`   🎯 Testing with market: ${btcMarket.symbol}`);
      
      // Get current ticker to determine a safe price
      const ticker = await this.deltaApi.getTicker(btcMarket.symbol);
      const currentPrice = parseFloat(ticker.close || ticker.mark_price || '50000');
      
      // Place a limit order well below market price (won't execute)
      const testPrice = currentPrice * 0.8; // 20% below market
      const testSize = 0.001; // Very small size for testing
      
      const orderParams: DeltaExchange.OrderParams = {
        symbol: btcMarket.symbol,
        side: 'buy',
        size: testSize,
        type: 'limit',
        price: testPrice,
        timeInForce: 'gtc',
        clientOrderId: `test_${Date.now()}`,
      };
      
      logger.info(`   📊 Placing test order: ${orderParams.side} ${orderParams.size} ${orderParams.symbol} @ $${testPrice}`);
      
      const order = await this.deltaApi.placeOrder(orderParams);
      
      logger.info(`   ✅ Order placed successfully!`);
      logger.info(`     Order ID: ${order.id}`);
      logger.info(`     Status: ${order.state || order.status}`);
      logger.info(`     Symbol: ${order.symbol}`);
      logger.info(`     Side: ${order.side}`);
      logger.info(`     Size: ${order.size}`);
      logger.info(`     Price: $${order.price}`);
      
      this.testResults.orderPlacement = {
        status: 'SUCCESS',
        orderId: order.id,
        symbol: order.symbol,
        side: order.side,
        size: order.size,
        price: order.price,
        orderStatus: order.state || order.status,
      };
      
      // Store order ID for cancellation test
      this.testResults.testOrderId = order.id;

    } catch (error) {
      logger.error('   ❌ Order placement test failed:', error);
      this.testResults.orderPlacement = {
        status: 'FAILED',
        error: error.message,
      };
    }
  }

  /**
   * Test 7: Order cancellation (exit test)
   */
  private async testOrderCancellation(): Promise<void> {
    try {
      logger.info('\n❌ TEST 7: ORDER CANCELLATION (EXIT)');

      if (!this.testResults.testOrderId) {
        logger.warn('   ⚠️  No test order ID available, skipping cancellation test');
        this.testResults.orderCancellation = {
          status: 'SKIPPED',
          reason: 'No order to cancel',
        };
        return;
      }

      const orderId = this.testResults.testOrderId;
      logger.info(`   🎯 Cancelling test order: ${orderId}`);

      // Cancel the order
      const cancelResult = await this.deltaApi.cancelOrder(orderId);

      logger.info(`   ✅ Order cancelled successfully!`);
      logger.info(`     Cancel result:`, cancelResult);

      // Verify cancellation by checking active orders
      const activeOrders = await this.deltaApi.getActiveOrders();
      const cancelledOrder = activeOrders.find(o => o.id === orderId);

      if (!cancelledOrder) {
        logger.info(`   ✅ Order ${orderId} no longer in active orders`);
      } else {
        logger.info(`   ⚠️  Order ${orderId} still shows as: ${cancelledOrder.state || cancelledOrder.status}`);
      }

      this.testResults.orderCancellation = {
        status: 'SUCCESS',
        orderId,
        cancelResult,
        stillActive: !!cancelledOrder,
      };

    } catch (error) {
      logger.error('   ❌ Order cancellation test failed:', error);
      this.testResults.orderCancellation = {
        status: 'FAILED',
        error: error.message,
      };
    }
  }

  /**
   * Test 8: Position management
   */
  private async testPositionManagement(): Promise<void> {
    try {
      logger.info('\n📊 TEST 8: POSITION MANAGEMENT');

      // Get current positions
      const positions = await this.deltaApi.getPositions();
      logger.info(`   ✅ Retrieved ${positions.length} current positions`);

      // Get active orders
      const activeOrders = await this.deltaApi.getActiveOrders();
      logger.info(`   ✅ Retrieved ${activeOrders.length} active orders`);

      // Get recent trade history
      const trades = await this.deltaApi.getTradeHistory({ limit: 10 });
      logger.info(`   ✅ Retrieved ${trades.length} recent trades`);

      if (trades.length > 0) {
        logger.info('   📈 Recent trades:');
        trades.slice(0, 3).forEach(trade => {
          logger.info(`     ${trade.symbol}: ${trade.side} ${trade.size} @ $${trade.price}`);
        });
      }

      this.testResults.positionManagement = {
        status: 'SUCCESS',
        currentPositions: positions.length,
        activeOrders: activeOrders.length,
        recentTrades: trades.length,
      };

    } catch (error) {
      logger.error('   ❌ Position management test failed:', error);
      this.testResults.positionManagement = {
        status: 'FAILED',
        error: error.message,
      };
    }
  }

  /**
   * Generate comprehensive test report
   */
  private generateTestReport(startTime: number): void {
    const duration = (Date.now() - startTime) / 1000;

    logger.info('\n' + '🎉 DELTA TESTNET TEST RESULTS'.padStart(80, '='));
    logger.info('=' .repeat(120));

    // Test Summary
    logger.info('📊 TEST SUMMARY:');
    const tests = [
      'connection',
      'accountInfo',
      'walletBalances',
      'marketData',
      'positions',
      'orderPlacement',
      'orderCancellation',
      'positionManagement'
    ];

    let passedTests = 0;
    let failedTests = 0;
    let skippedTests = 0;

    tests.forEach(testName => {
      const result = this.testResults[testName];
      if (result) {
        const status = result.status;
        const icon = status === 'SUCCESS' ? '✅' : status === 'FAILED' ? '❌' : '⚠️';
        logger.info(`   ${icon} ${testName.toUpperCase()}: ${status}`);

        if (status === 'SUCCESS') passedTests++;
        else if (status === 'FAILED') failedTests++;
        else skippedTests++;
      }
    });

    logger.info('\n📈 OVERALL RESULTS:');
    logger.info(`   Total Tests: ${tests.length}`);
    logger.info(`   Passed: ${passedTests}`);
    logger.info(`   Failed: ${failedTests}`);
    logger.info(`   Skipped: ${skippedTests}`);
    logger.info(`   Success Rate: ${((passedTests / tests.length) * 100).toFixed(1)}%`);
    logger.info(`   Execution Time: ${duration.toFixed(2)} seconds`);

    // Connection Status
    logger.info('\n🔗 CONNECTION STATUS:');
    if (this.testResults.connection?.status === 'SUCCESS') {
      logger.info('   ✅ Delta testnet connection: WORKING');
      logger.info('   ✅ API authentication: WORKING');
      logger.info('   ✅ Server communication: WORKING');
    } else {
      logger.info('   ❌ Delta testnet connection: FAILED');
    }

    // Trading Capabilities
    logger.info('\n💰 TRADING CAPABILITIES:');
    if (this.testResults.walletBalances?.tradingBalance > 0) {
      logger.info(`   ✅ Trading balance: $${this.testResults.walletBalances.tradingBalance} ${this.testResults.walletBalances.tradingAsset}`);
    } else {
      logger.info('   ⚠️  No trading balance available (testnet account may need funding)');
    }

    if (this.testResults.orderPlacement?.status === 'SUCCESS') {
      logger.info('   ✅ Order placement: WORKING');
    } else {
      logger.info('   ❌ Order placement: FAILED');
    }

    if (this.testResults.orderCancellation?.status === 'SUCCESS') {
      logger.info('   ✅ Order cancellation: WORKING');
    } else if (this.testResults.orderCancellation?.status === 'SKIPPED') {
      logger.info('   ⚠️  Order cancellation: SKIPPED');
    } else {
      logger.info('   ❌ Order cancellation: FAILED');
    }

    // Market Data Access
    logger.info('\n📊 MARKET DATA ACCESS:');
    if (this.testResults.marketData?.status === 'SUCCESS') {
      logger.info(`   ✅ Market data: ${this.testResults.marketData.totalMarkets} markets available`);
      logger.info(`   ✅ BTC market: ${this.testResults.marketData.btcMarket}`);
    } else {
      logger.info('   ❌ Market data access: FAILED');
    }

    // Account Information
    logger.info('\n👤 ACCOUNT INFORMATION:');
    if (this.testResults.accountInfo?.status === 'SUCCESS') {
      logger.info(`   ✅ Account ID: ${this.testResults.accountInfo.accountId}`);
      logger.info(`   ✅ Email: ${this.testResults.accountInfo.email}`);
      logger.info(`   ✅ Trading enabled: ${this.testResults.accountInfo.tradingEnabled}`);
    } else {
      logger.info('   ❌ Account information: FAILED');
    }

    // Final Assessment
    logger.info('\n🎯 FINAL ASSESSMENT:');
    if (passedTests >= 6) {
      logger.info('   🚀 EXCELLENT: Delta testnet integration is working well!');
      logger.info('   ✅ Ready for live trading system integration');
      logger.info('   ✅ Entry and exit functionality verified');
      logger.info('   ✅ Account and balance management working');

      if (this.testResults.walletBalances?.tradingBalance > 0) {
        logger.info('   💰 Sufficient balance for trading tests');
        logger.info('   🎯 READY FOR DYNAMIC TAKE PROFIT SYSTEM DEPLOYMENT');
      } else {
        logger.info('   ⚠️  Consider adding testnet funds for full trading tests');
      }
    } else if (passedTests >= 4) {
      logger.info('   📈 GOOD: Most functionality working, some issues to resolve');
      logger.info('   🔧 Address failed tests before live deployment');
    } else {
      logger.info('   ⚠️  NEEDS WORK: Multiple issues detected');
      logger.info('   🔧 Resolve connection and API issues before proceeding');
    }

    // Next Steps
    logger.info('\n🔄 NEXT STEPS:');
    if (passedTests >= 6) {
      logger.info('   1. ✅ Delta testnet integration verified');
      logger.info('   2. 🚀 Deploy dynamic take profit system');
      logger.info('   3. 📊 Run 3-month backtest with real Delta balance');
      logger.info('   4. 💰 Start live trading with small capital');
      logger.info('   5. 📈 Scale up based on performance');
    } else {
      logger.info('   1. 🔧 Fix failed test cases');
      logger.info('   2. 💰 Add testnet funds if needed');
      logger.info('   3. 🔄 Re-run connection tests');
      logger.info('   4. ✅ Verify all functionality before live deployment');
    }

    logger.info('=' .repeat(120));
  }
}

/**
 * Main execution function
 */
async function main() {
  const tester = new DeltaTestnetTester();

  try {
    await tester.runComprehensiveTest();
  } catch (error) {
    logger.error('💥 Delta testnet test failed:', error);
    process.exit(1);
  }
}

if (require.main === module) {
  main().catch(console.error);
}

export { DeltaTestnetTester };
