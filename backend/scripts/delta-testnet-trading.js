#!/usr/bin/env node
/**
 * DELTA EXCHANGE TESTNET TRADING SYSTEM
 * 
 * Clean, standalone implementation of our Ultimate Trading Strategy
 * Running ONLY on Delta Exchange testnet for safe validation
 * 
 * STRATEGY: Daily OHLC Zone Trading
 * - Previous Day High (PDH) resistance
 * - Previous Day Low (PDL) support
 * - Previous Day Close (PDC) pivot
 * - 75%+ confluence required for trades
 * - 2.5% risk per trade, 3:1 reward ratio
 */

const { DeltaExchangeUnified } = require('../dist/services/DeltaExchangeUnified');
const { logger } = require('../dist/utils/logger');

class DeltaTestnetTradingSystem {
  constructor() {
    require('dotenv').config();
    
    // TESTNET ONLY - Safe for testing
    this.deltaService = new DeltaExchangeUnified({
      apiKey: process.env.DELTA_EXCHANGE_API_KEY,
      apiSecret: process.env.DELTA_EXCHANGE_API_SECRET,
      testnet: true // TESTNET ONLY
    });

    // Trading configuration
    this.config = {
      // Testnet symbols (Delta Exchange product IDs)
      symbols: [
        { symbol: 'BTCUSD', productId: parseInt(process.env.DELTA_BTCUSD_PRODUCT_ID || 84) },   // BTC perpetual
        { symbol: 'ETHUSD', productId: parseInt(process.env.DELTA_ETHUSD_PRODUCT_ID || 1699) },  // ETH perpetual
        { symbol: 'SOLUSD', productId: parseInt(process.env.DELTA_SOLUSD_PRODUCT_ID || 92572) },  // SOL perpetual
        { symbol: 'ADAUSD', productId: parseInt(process.env.DELTA_ADAUSD_PRODUCT_ID || 101760) }   // ADA perpetual
      ],
      
      // Risk management
      riskPerTrade: 2.5,        // 2.5% risk per trade
      maxConcurrentPositions: 2, // Max 2 positions
      stopLossPercent: 1.2,     // 1.2% stop loss
      takeProfitRatio: 3.0,     // 3:1 reward ratio
      
      // OHLC Zone strategy
      zoneBuffer: 0.15,         // 0.15% zone buffer
      minZoneStrength: 75,      // 75% minimum zone strength
      confluenceThreshold: 0.75, // 75% minimum confluence
      
      // Trading limits
      maxTradesPerDay: 3,       // Max 3 trades per day
      tradingHours: {
        start: 0,   // 00:00 UTC
        end: 24     // 24:00 UTC (24/7)
      }
    };

    // System state
    this.isRunning = false;
    this.activePositions = new Map();
    this.dailyLevels = new Map();
    this.performance = {
      totalTrades: 0,
      winningTrades: 0,
      dailyTrades: 0,
      totalPnL: 0,
      winRate: 0,
      lastTradeDate: null
    };
    
    this.portfolio = {
      initialBalance: 0,
      currentBalance: 0,
      availableBalance: 0
    };
  }

  /**
   * Start Delta Exchange testnet trading
   */
  async startTestnetTrading() {
    try {
      this.isRunning = true;
      
      logger.info('🚀 STARTING DELTA EXCHANGE TESTNET TRADING SYSTEM');
      logger.info('═'.repeat(80));
      logger.info('⚠️  TESTNET MODE - Safe for testing with virtual funds');
      logger.info('📊 Strategy: Daily OHLC Zone Trading');
      logger.info('🎯 Target: 75%+ confluence, 3:1 risk/reward');
      logger.info('═'.repeat(80));
      
      // Initialize system
      await this.initializeSystem();
      
      // Main trading loop
      while (this.isRunning) {
        try {
          // Update portfolio balance
          await this.updatePortfolioBalance();
          
          // Reset daily counters if new day
          this.resetDailyCountersIfNeeded();
          
          // Update daily OHLC levels
          await this.updateDailyOHLCLevels();
          
          // Check trading limits
          if (this.performance.dailyTrades >= this.config.maxTradesPerDay) {
            logger.info(`⏸️ Daily trade limit reached (${this.performance.dailyTrades}/${this.config.maxTradesPerDay})`);
            await this.sleep(300000); // Wait 5 minutes
            continue;
          }
          
          // Analyze each symbol for trading opportunities
          for (const symbolConfig of this.config.symbols) {
            if (!this.isRunning) break;
            
            // Skip if already have position
            if (this.activePositions.has(symbolConfig.symbol)) {
              continue;
            }
            
            await this.analyzeSymbolForTrade(symbolConfig);
            await this.sleep(5000); // 5 second delay between symbols
          }
          
          // Manage active positions
          await this.manageActivePositions();
          
          // Display current status
          this.displayTradingStatus();
          
          // Wait before next cycle
          await this.sleep(30000); // 30 second cycle
          
        } catch (error) {
          logger.error('❌ Error in trading loop:', error);
          await this.sleep(30000);
        }
      }
      
    } catch (error) {
      logger.error('❌ Failed to start testnet trading:', error);
      throw error;
    }
  }

  /**
   * Initialize trading system
   */
  async initializeSystem() {
    try {
      logger.info('🔧 Initializing Delta Exchange testnet connection...');
      
      // Get account balance
      await this.updatePortfolioBalance();
      
      logger.info(`💰 Testnet Balance: $${this.portfolio.currentBalance.toFixed(2)}`);
      logger.info(`📊 Available for Trading: $${this.portfolio.availableBalance.toFixed(2)}`);
      
      // Initialize daily levels for each symbol
      for (const symbolConfig of this.config.symbols) {
        await this.calculateInitialOHLCLevels(symbolConfig);
      }
      
      logger.info('✅ System initialized successfully');
      
    } catch (error) {
      logger.error('❌ Failed to initialize system:', error);
      throw error;
    }
  }

  /**
   * Update portfolio balance from Delta Exchange
   */
  async updatePortfolioBalance() {
    try {
      const balance = await this.deltaService.getBalance();
      
      // Use USDT balance for trading
      const usdtBalance = balance.find(b => b.asset === 'USDT') || { balance: 0, available_balance: 0 };
      
      this.portfolio.currentBalance = parseFloat(usdtBalance.balance || 0);
      this.portfolio.availableBalance = parseFloat(usdtBalance.available_balance || 0);
      
      if (this.portfolio.initialBalance === 0) {
        this.portfolio.initialBalance = this.portfolio.currentBalance;
      }
      
    } catch (error) {
      logger.error('❌ Failed to update portfolio balance:', error);
      // Use fallback values for demo
      this.portfolio.currentBalance = 10000;
      this.portfolio.availableBalance = 10000;
      this.portfolio.initialBalance = 10000;
    }
  }

  /**
   * Calculate initial OHLC levels for symbol
   */
  async calculateInitialOHLCLevels(symbolConfig) {
    try {
      // Get recent candles for OHLC calculation
      const candles = await this.deltaService.getCandles(symbolConfig.productId, '1d', 2);
      
      if (candles && candles.length >= 2) {
        const previousDay = candles[candles.length - 2];
        
        const PDH = parseFloat(previousDay.high);
        const PDL = parseFloat(previousDay.low);
        const PDC = parseFloat(previousDay.close);
        const PDO = parseFloat(previousDay.open);
        const PP = (PDH + PDL + PDC) / 3;
        
        this.dailyLevels.set(symbolConfig.symbol, {
          PDH, PDL, PDC, PDO, PP,
          lastUpdate: Date.now()
        });
        
        logger.info(`📊 ${symbolConfig.symbol} OHLC Levels:`);
        logger.info(`   PDH: $${PDH.toFixed(2)} | PDL: $${PDL.toFixed(2)} | PDC: $${PDC.toFixed(2)}`);
      }
      
    } catch (error) {
      logger.error(`❌ Failed to calculate OHLC levels for ${symbolConfig.symbol}:`, error);
    }
  }

  /**
   * Update daily OHLC levels
   */
  async updateDailyOHLCLevels() {
    const now = Date.now();
    
    for (const symbolConfig of this.config.symbols) {
      const levels = this.dailyLevels.get(symbolConfig.symbol);
      
      // Update once per day
      if (!levels || (now - levels.lastUpdate) > 24 * 60 * 60 * 1000) {
        await this.calculateInitialOHLCLevels(symbolConfig);
      }
    }
  }

  /**
   * Analyze symbol for trading opportunity
   */
  async analyzeSymbolForTrade(symbolConfig) {
    try {
      // Get current market price
      const marketData = await this.deltaService.getMarketData(symbolConfig.symbol);
      const currentPrice = parseFloat(marketData.last_price || marketData.mark_price);
      
      // Get daily levels
      const levels = this.dailyLevels.get(symbolConfig.symbol);
      if (!levels) return;
      
      // Perform OHLC zone analysis
      const zoneAnalysis = this.performOHLCZoneAnalysis(currentPrice, levels);
      
      if (zoneAnalysis.inZone && zoneAnalysis.strength >= this.config.minZoneStrength) {
        
        // Calculate confluence score
        const confluenceScore = this.calculateConfluenceScore(zoneAnalysis);
        
        if (confluenceScore >= this.config.confluenceThreshold) {
          
          // Execute trade
          const trade = await this.executeTrade(symbolConfig, currentPrice, zoneAnalysis, confluenceScore);
          
          if (trade) {
            logger.info(`✅ TRADE EXECUTED: ${symbolConfig.symbol} ${trade.side} @ $${currentPrice.toFixed(2)}`);
            logger.info(`   🎯 Zone: ${zoneAnalysis.zoneName} (${zoneAnalysis.strength}% strength)`);
            logger.info(`   📊 Confluence: ${(confluenceScore * 100).toFixed(0)}%`);
            logger.info(`   💰 Risk: ${this.config.riskPerTrade}% | R:R = 1:${this.config.takeProfitRatio}`);
          }
        }
      }
      
    } catch (error) {
      logger.error(`❌ Failed to analyze ${symbolConfig.symbol}:`, error);
    }
  }

  /**
   * Perform OHLC zone analysis
   */
  performOHLCZoneAnalysis(currentPrice, levels) {
    const { PDH, PDL, PDC, PP } = levels;
    const zoneBuffer = this.config.zoneBuffer / 100;
    
    // Check each zone for interaction
    const zones = [
      { name: 'PDH_Resistance', level: PDH, type: 'resistance', strength: 90 },
      { name: 'PDL_Support', level: PDL, type: 'support', strength: 90 },
      { name: 'PDC_Pivot', level: PDC, type: 'pivot', strength: 80 },
      { name: 'Daily_Pivot', level: PP, type: 'pivot', strength: 75 }
    ];
    
    for (const zone of zones) {
      const distance = Math.abs(currentPrice - zone.level) / zone.level;
      
      if (distance <= zoneBuffer) {
        return {
          inZone: true,
          zoneName: zone.name,
          zoneType: zone.type,
          zoneLevel: zone.level,
          strength: zone.strength,
          distance: distance * 100,
          signal: this.generateZoneSignal(currentPrice, zone)
        };
      }
    }
    
    return { inZone: false };
  }

  /**
   * Generate trading signal from zone
   */
  generateZoneSignal(currentPrice, zone) {
    const stopLossPercent = this.config.stopLossPercent / 100;
    const takeProfitRatio = this.config.takeProfitRatio;
    
    let signal = 'wait';
    let stopLoss = 0;
    let takeProfit = 0;
    
    if (zone.type === 'resistance') {
      signal = 'sell';
      stopLoss = zone.level * (1 + stopLossPercent);
      takeProfit = currentPrice * (1 - (stopLossPercent * takeProfitRatio));
    } else if (zone.type === 'support') {
      signal = 'buy';
      stopLoss = zone.level * (1 - stopLossPercent);
      takeProfit = currentPrice * (1 + (stopLossPercent * takeProfitRatio));
    }
    
    return {
      action: signal,
      stopLoss: stopLoss,
      takeProfit: takeProfit,
      confidence: zone.strength / 100
    };
  }

  /**
   * Calculate confluence score
   */
  calculateConfluenceScore(zoneAnalysis) {
    let score = 0.4; // Base score
    
    // Zone strength (60% weight)
    score += (zoneAnalysis.strength / 100) * 0.6;
    
    // Distance bonus (closer = better)
    const distanceBonus = Math.max(0, (0.15 - zoneAnalysis.distance) / 0.15) * 0.2;
    score += distanceBonus;
    
    // Zone type bonus
    if (zoneAnalysis.zoneName.includes('PDH') || zoneAnalysis.zoneName.includes('PDL')) {
      score += 0.1; // 10% bonus for PDH/PDL
    }
    
    return Math.min(0.95, score);
  }

  /**
   * Sleep utility
   */
  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}
