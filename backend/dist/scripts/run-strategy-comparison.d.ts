#!/usr/bin/env node
/**
 * Strategy Comparison Script
 * Compares original MA Crossover vs Enhanced Trend Strategy
 */
declare class StrategyComparison {
    /**
     * Run strategy comparison
     */
    runComparison(): Promise<void>;
    private initializeInfrastructure;
    private createBacktestConfig;
    private loadMarketData;
    private testStrategy;
    private compareStrategies;
    private calculateImprovementScore;
    cleanup(): Promise<void>;
}
export { StrategyComparison };
