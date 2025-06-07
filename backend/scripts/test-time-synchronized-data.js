/**
 * TEST TIME-SYNCHRONIZED DATA FETCHING
 * 
 * This script tests the corrected approach to data synchronization:
 * - All timeframes cover the SAME time period
 * - Proper bar counts calculated for each timeframe
 * - Validates data alignment across timeframes
 */

const IntelligentMarketDataManager = require('./intelligent-market-data-manager');
const MomentumTrainAnalyzer = require('./momentum-train-analyzer');
const CandleFormationAnalyzer = require('./candle-formation-analyzer');

class TimeSynchronizedDataTest {
  constructor() {
    this.dataManager = new IntelligentMarketDataManager();
    this.momentumAnalyzer = null; // Will be initialized after data manager
    this.candleAnalyzer = null; // Will be initialized after data manager
  }

  async runTest() {
    console.log(`🧪 TESTING TIME-SYNCHRONIZED DATA FETCHING`);
    console.log(`════════════════════════════════════════════════════════════════════════════════`);
    console.log(`🔧 FIXING CRITICAL DATA ALIGNMENT ISSUE:`);
    console.log(`   ❌ OLD: 100 bars per timeframe (misaligned time periods)`);
    console.log(`   ✅ NEW: Time-synchronized bars for same period`);
    console.log(`════════════════════════════════════════════════════════════════════════════════`);

    try {
      // Initialize the data manager
      console.log(`🔧 Initializing time-synchronized data manager...`);
      await this.dataManager.initialize();

      // Initialize momentum train analyzer
      this.momentumAnalyzer = new MomentumTrainAnalyzer(this.dataManager);

      // Initialize candle formation analyzer
      this.candleAnalyzer = new CandleFormationAnalyzer(this.dataManager);

      // Test data alignment
      await this.testDataAlignment();

      // Display time coverage analysis
      await this.analyzeTimeCoverage();

      // Test momentum train analysis
      await this.testMomentumTrainAnalysis();

      // Test candle formation analysis
      await this.testCandleFormationAnalysis();

    } catch (error) {
      console.error(`❌ Test error: ${error.message}`);
    }
  }

  async testDataAlignment() {
    console.log(`\n📊 TESTING DATA ALIGNMENT:`);

    const symbol = 'BTCUSD';
    const timeframes = ['1d', '4h', '1h', '15m', '5m']; // Added missing 1H timeframe
    
    for (const timeframe of timeframes) {
      const data = this.dataManager.getHistoricalData(symbol, timeframe);
      
      if (data && data.length > 0) {
        const firstCandle = data[0];
        const lastCandle = data[data.length - 1];
        const firstTime = new Date(firstCandle.time);
        const lastTime = new Date(lastCandle.time);
        const timeDiff = (lastTime - firstTime) / (1000 * 60 * 60 * 24); // Days
        
        console.log(`   ${timeframe}: ${data.length} bars | ${timeDiff.toFixed(1)} days | ${firstTime.toISOString().split('T')[0]} → ${lastTime.toISOString().split('T')[0]}`);
      } else {
        console.log(`   ${timeframe}: No data available`);
      }
    }
  }

  async analyzeTimeCoverage() {
    console.log(`\n🕐 TIME COVERAGE ANALYSIS:`);

    const symbol = 'BTCUSD';
    const timeframes = ['1d', '4h', '1h', '15m', '5m']; // Added missing 1H timeframe
    
    // Calculate expected vs actual coverage
    const analysisTimeframeDays = this.dataManager.config.analysisTimeframeDays;
    
    console.log(`📅 Target Analysis Period: ${analysisTimeframeDays} days`);
    console.log(`════════════════════════════════════════════════════════════════════════════════`);
    
    for (const timeframe of timeframes) {
      const data = this.dataManager.getHistoricalData(symbol, timeframe);
      const expectedBars = this.dataManager.config.timeframeBars[timeframe];
      
      if (data && data.length > 0) {
        const actualBars = data.length;
        const firstCandle = data[0];
        const lastCandle = data[data.length - 1];
        const actualDays = (lastCandle.time - firstCandle.time) / (1000 * 60 * 60 * 24);
        
        console.log(`📊 ${timeframe.toUpperCase()} TIMEFRAME:`);
        console.log(`   Expected Bars: ${expectedBars}`);
        console.log(`   Actual Bars: ${actualBars}`);
        console.log(`   Actual Days: ${actualDays.toFixed(1)}`);
        console.log(`   Coverage: ${((actualDays / analysisTimeframeDays) * 100).toFixed(1)}%`);
        console.log(`   Status: ${Math.abs(actualDays - analysisTimeframeDays) < 2 ? '✅ ALIGNED' : '⚠️ MISALIGNED'}`);
        console.log(``);
      }
    }
  }

  /**
   * Test momentum train analysis with professional 4-tier hierarchy
   */
  async testMomentumTrainAnalysis() {
    console.log(`\n🚂 TESTING MOMENTUM TRAIN ANALYSIS:`);
    console.log(`════════════════════════════════════════════════════════════════════════════════`);
    console.log(`🎯 PROFESSIONAL 4-TIER TIMEFRAME HIERARCHY:`);
    console.log(`   4H - TREND DIRECTION: Where is the train going?`);
    console.log(`   1H - TREND CONFIRMATION: Is the train accelerating?`);
    console.log(`   15M - ENTRY TIMING: When to jump on the train?`);
    console.log(`   5M - PRECISION ENTRY: Exact entry point`);
    console.log(`════════════════════════════════════════════════════════════════════════════════`);

    const symbol = 'BTCUSD';
    const currentPrice = this.dataManager.getCurrentPrice(symbol) || 103000;

    if (this.momentumAnalyzer) {
      const momentumSignal = await this.momentumAnalyzer.analyzeMomentumTrain(symbol, currentPrice);

      if (momentumSignal) {
        console.log(`\n🎯 MOMENTUM TRAIN SIGNAL SUMMARY:`);
        console.log(`   Action: ${momentumSignal.action.toUpperCase()}`);
        console.log(`   Confidence: ${(momentumSignal.confidence * 100).toFixed(1)}%`);
        console.log(`   Direction: ${momentumSignal.direction}`);
        console.log(`   Momentum: ${momentumSignal.momentum}`);
        console.log(`   Timing: ${momentumSignal.timing}`);
        console.log(`   Precision: ${momentumSignal.precision}`);
        console.log(`   Current Price: $${momentumSignal.currentPrice.toFixed(2)}`);

        if (momentumSignal.action !== 'wait') {
          console.log(`\n🚂 MOMENTUM TRAIN DETECTED! Ready to ${momentumSignal.action.replace('_', ' ').toUpperCase()}!`);
        } else {
          console.log(`\n⏳ Waiting for momentum train... Current conditions not optimal for entry.`);
        }
      } else {
        console.log(`⚠️ Momentum train analysis failed - insufficient data`);
      }
    } else {
      console.log(`❌ Momentum analyzer not initialized`);
    }
  }

  /**
   * Test candle formation analysis with real-time insights
   */
  async testCandleFormationAnalysis() {
    console.log(`\n🕯️ TESTING CANDLE FORMATION ANALYSIS:`);
    console.log(`════════════════════════════════════════════════════════════════════════════════`);
    console.log(`🔍 CANDLE ANATOMY & INTRA-CANDLE BEHAVIOR:`);
    console.log(`   • Body: Open vs Close battle (buying/selling pressure)`);
    console.log(`   • Upper Wick: Rejection at higher prices (selling pressure)`);
    console.log(`   • Lower Wick: Rejection at lower prices (buying pressure)`);
    console.log(`   • Color: Green (bullish) vs Red (bearish)`);
    console.log(`════════════════════════════════════════════════════════════════════════════════`);
    console.log(`🕯️ TIMEFRAME RELATIONSHIPS:`);
    console.log(`   • 4H Candle = 16 × 15M candles = 48 × 5M candles`);
    console.log(`   • 1H Candle = 4 × 15M candles = 12 × 5M candles`);
    console.log(`   • 15M Candle = 3 × 5M candles`);
    console.log(`   Shorter timeframes show HOW longer candles are forming!`);
    console.log(`════════════════════════════════════════════════════════════════════════════════`);

    const symbol = 'BTCUSD';
    const currentPrice = this.dataManager.getCurrentPrice(symbol) || 103000;

    if (this.candleAnalyzer) {
      const formationAnalysis = await this.candleAnalyzer.analyzeCurrentCandleFormation(symbol, currentPrice);

      if (formationAnalysis) {
        console.log(`\n🎯 CANDLE FORMATION SUMMARY:`);
        console.log(`   Overall Signal: ${formationAnalysis.intraCandleAnalysis.overallSignal}`);
        console.log(`   Next Move Prediction: ${formationAnalysis.intraCandleAnalysis.nextMovePrediction}`);
        console.log(`   Current Price: $${formationAnalysis.currentPrice.toFixed(2)}`);

        // Display key insights
        console.log(`\n🔍 KEY INSIGHTS:`);
        console.log(`   4H Formation: ${formationAnalysis.intraCandleAnalysis.fourHourInsight}`);
        console.log(`   1H Formation: ${formationAnalysis.intraCandleAnalysis.oneHourInsight}`);
        console.log(`   15M Formation: ${formationAnalysis.intraCandleAnalysis.fifteenMinInsight}`);

        // Show most significant formation
        const formations = formationAnalysis.formations;
        const significantTF = this.findMostSignificantFormation(formations);

        if (significantTF) {
          console.log(`\n🎯 MOST SIGNIFICANT FORMATION (${significantTF.timeframe.toUpperCase()}):`);
          console.log(`   Type: ${significantTF.bodyType}`);
          console.log(`   Signal: ${significantTF.candleType?.signal || 'N/A'}`);
          console.log(`   Strength: ${significantTF.candleType?.strength || 'N/A'}`);
          console.log(`   Prediction: ${significantTF.nextCandlePrediction}`);
        }

      } else {
        console.log(`⚠️ Candle formation analysis failed - insufficient data`);
      }
    } else {
      console.log(`❌ Candle analyzer not initialized`);
    }
  }

  /**
   * Find the most significant candle formation
   */
  findMostSignificantFormation(formations) {
    if (!formations) return null;

    // Priority: 4h > 1h > 15m (longer timeframes more significant)
    const priority = ['4h', '1h', '15m'];

    for (const tf of priority) {
      const formation = formations[tf];
      if (formation && formation.candleType && formation.candleType.strength === 'high') {
        return formation;
      }
    }

    // If no high-strength formation, return first available
    return Object.values(formations)[0] || null;
  }
}

// Run the test
async function runTimeSynchronizedTest() {
  const test = new TimeSynchronizedDataTest();
  await test.runTest();
}

if (require.main === module) {
  runTimeSynchronizedTest().catch(console.error);
}

module.exports = TimeSynchronizedDataTest;
