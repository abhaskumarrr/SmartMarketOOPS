#!/usr/bin/env node
/**
 * Intelligent AI-Driven Backtesting Script
 * Tests the integrated AI trading system with existing ML models
 */
declare class IntelligentBacktestRunner {
    /**
     * Run intelligent AI-driven backtesting
     */
    runIntelligentBacktest(): Promise<void>;
    private initializeInfrastructure;
    private createIntelligentConfig;
    private loadMarketData;
    private testIntelligentSystem;
    private publishIntelligentSignal;
    private displayIntelligentResults;
    private getIntelligentRating;
    cleanup(): Promise<void>;
}
export { IntelligentBacktestRunner };
