#!/usr/bin/env node
/**
 * Enhanced 75% Win Rate Trading Backtest
 * Target: 3-5 PROFITABLE trades daily with 75%+ win rate
 * Implementation: Advanced ML ensemble + Enhanced filtering + Market regime detection
 */
interface Enhanced75Config {
    symbol: string;
    startDate: string;
    endDate: string;
    initialCapital: number;
    leverage: number;
    riskPerTrade: number;
    targetTradesPerDay: number;
    targetWinRate: number;
    mlAccuracy: number;
}
interface Enhanced75Trade {
    id: string;
    side: 'LONG' | 'SHORT';
    entryPrice: number;
    exitPrice: number;
    size: number;
    pnl: number;
    exitReason: string;
    mlConfidence: number;
    ensembleScore: number;
    signalScore: number;
    qualityScore: number;
    marketRegime: string;
    holdTimeMinutes: number;
    timestamp: number;
}
interface MLEnsemblePrediction {
    model1_confidence: number;
    model2_confidence: number;
    model3_confidence: number;
    ensemble_confidence: number;
    consensus_side: 'LONG' | 'SHORT';
    agreement_score: number;
}
interface MarketRegime {
    regime: 'trending_bullish' | 'trending_bearish' | 'breakout_bullish' | 'breakout_bearish' | 'ranging' | 'volatile';
    confidence: number;
    volatility: number;
    trend_strength: number;
    volume_profile: number;
}
declare class Enhanced75PercentBacktester {
    private config;
    private currentBalance;
    private trades;
    private maxDrawdown;
    private peakBalance;
    private dailyTrades;
    constructor(config: Enhanced75Config);
    runBacktest(): Promise<void>;
    private generateEnhancedFrequencyETHData;
    private getEnhancedTrendFactor;
    private generatePremiumOpportunities;
    private generateMLEnsemblePrediction;
    private simulateMLModel;
    private detectMarketRegime;
    private generatePremiumTradingSignal;
    private calculatePremiumQualityScore;
    private passesEnhancedFiltering;
    private executeEnhancedTrade;
    private calculateEnhancedHoldTime;
    private exitEnhancedTrade;
    private getCurrentCorrelationRisk;
    private displayEnhancedResults;
    private analyzeRegimePerformance;
}
