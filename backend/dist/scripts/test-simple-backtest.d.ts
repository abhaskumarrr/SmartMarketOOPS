#!/usr/bin/env node
/**
 * Simple Test Backtest
 * Basic test to see if the system works
 */
declare const realBalance = 2500;
declare const tradingCapital: number;
declare const leverage = 200;
declare const riskPerTrade = 5;
declare const assets: string[];
declare const results: any[];
declare const totalPnL: any;
declare const totalSignals: any;
declare const totalTrades: any;
declare const totalFinalBalance: any;
declare const overallReturn: number;
declare const balanceImpact: number;
declare const newRealBalance: number;
