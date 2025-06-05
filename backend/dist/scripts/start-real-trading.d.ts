#!/usr/bin/env node
/**
 * REAL TRADING ENGINE LAUNCHER
 *
 * ⚠️  WARNING: This script places REAL ORDERS on Delta Exchange with REAL MONEY!
 *
 * This is NOT a simulation - all trades will be executed live on your Delta Exchange account.
 * Make sure you understand the risks before running this script.
 */
declare function startRealTrading(): Promise<void>;
export { startRealTrading };
