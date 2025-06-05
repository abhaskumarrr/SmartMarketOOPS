#!/usr/bin/env node
/**
 * ETH Intelligent Trading Bot Backtest
 * Execute backtest with $10 capital, 200x leverage on ETH using 1 year data
 */
declare function runETHBacktest(): Promise<void>;
export { runETHBacktest };
