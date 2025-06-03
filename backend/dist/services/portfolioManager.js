"use strict";
/**
 * Portfolio Manager for Backtesting
 * Manages positions, trades, and portfolio performance
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.PortfolioManager = void 0;
const logger_1 = require("../utils/logger");
const events_1 = require("../types/events");
class PortfolioManager {
    constructor(config) {
        this.positions = new Map();
        this.trades = [];
        this.portfolioHistory = [];
        this.maxDrawdown = 0;
        this.totalCommission = 0;
        this.config = config;
        this.cash = config.initialCapital;
        this.highWaterMark = config.initialCapital;
        logger_1.logger.info('💼 Portfolio Manager initialized', {
            initialCapital: config.initialCapital,
            leverage: config.leverage,
            riskPerTrade: config.riskPerTrade,
            commission: config.commission,
        });
    }
    /**
     * Execute a trading signal
     */
    executeTrade(signal, currentPrice, timestamp) {
        try {
            logger_1.logger.debug(`🔄 Attempting to execute trade:`, {
                symbol: signal.symbol,
                type: signal.type,
                price: currentPrice,
                quantity: signal.quantity,
                confidence: signal.confidence,
                cash: this.cash,
            });
            if (signal.type === 'BUY') {
                return this.openLongPosition(signal, currentPrice, timestamp);
            }
            else if (signal.type === 'SELL') {
                return this.openShortPosition(signal, currentPrice, timestamp);
            }
            logger_1.logger.debug(`⚠️ Signal type ${signal.type} not handled`);
            return null;
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to execute trade:', error);
            return null;
        }
    }
    /**
     * Open a long position
     */
    openLongPosition(signal, currentPrice, timestamp) {
        const positionKey = `${signal.symbol}_LONG`;
        // Check if we already have a long position
        if (this.positions.has(positionKey)) {
            logger_1.logger.debug(`📊 Already have long position for ${signal.symbol}`);
            return null;
        }
        // Close any existing short position first
        const shortPositionKey = `${signal.symbol}_SHORT`;
        if (this.positions.has(shortPositionKey)) {
            this.closePosition(shortPositionKey, currentPrice, timestamp, 'Position reversal');
        }
        // Calculate position size and cost
        const quantity = this.calculatePositionSize(signal, currentPrice);
        const cost = quantity * currentPrice;
        const commission = cost * (this.config.commission / 100);
        const totalCost = cost + commission;
        logger_1.logger.debug(`📊 Position calculation:`, {
            signalQuantity: signal.quantity,
            calculatedQuantity: quantity,
            cost,
            commission,
            totalCost,
            leverage: this.config.leverage,
        });
        // Check if we have enough cash (considering leverage)
        const requiredCash = totalCost / this.config.leverage;
        logger_1.logger.debug(`💰 Cash check:`, {
            requiredCash,
            availableCash: this.cash,
            sufficient: requiredCash <= this.cash,
        });
        if (requiredCash > this.cash) {
            logger_1.logger.info(`💰 INSUFFICIENT CASH: Required: $${requiredCash.toFixed(2)}, Available: $${this.cash.toFixed(2)}`);
            return null;
        }
        // Create position
        const position = {
            symbol: signal.symbol,
            side: 'LONG',
            size: quantity,
            entryPrice: currentPrice,
            entryTime: timestamp,
            currentPrice: currentPrice,
            unrealizedPnl: 0,
            leverage: this.config.leverage,
        };
        // Update cash and positions
        this.cash -= requiredCash;
        this.totalCommission += commission;
        this.positions.set(positionKey, position);
        logger_1.logger.info(`📈 Opened LONG position`, {
            symbol: signal.symbol,
            quantity,
            price: currentPrice,
            cost: totalCost,
            commission,
            remainingCash: this.cash,
        });
        // Return trade record (entry only, will be completed when closed)
        return {
            id: (0, events_1.createEventId)(),
            symbol: signal.symbol,
            side: 'LONG',
            entryPrice: currentPrice,
            exitPrice: 0, // Will be set when position is closed
            quantity,
            entryTime: timestamp,
            exitTime: 0, // Will be set when position is closed
            pnl: 0, // Will be calculated when position is closed
            pnlPercent: 0,
            commission,
            strategy: signal.strategy,
            reason: signal.reason,
            duration: 0,
        };
    }
    /**
     * Open a short position
     */
    openShortPosition(signal, currentPrice, timestamp) {
        const positionKey = `${signal.symbol}_SHORT`;
        // Check if we already have a short position
        if (this.positions.has(positionKey)) {
            logger_1.logger.debug(`📊 Already have short position for ${signal.symbol}`);
            return null;
        }
        // Close any existing long position first
        const longPositionKey = `${signal.symbol}_LONG`;
        if (this.positions.has(longPositionKey)) {
            this.closePosition(longPositionKey, currentPrice, timestamp, 'Position reversal');
        }
        // Calculate position size and cost
        const quantity = this.calculatePositionSize(signal, currentPrice);
        const cost = quantity * currentPrice;
        const commission = cost * (this.config.commission / 100);
        const totalCost = cost + commission;
        // Check if we have enough cash (considering leverage)
        const requiredCash = totalCost / this.config.leverage;
        if (requiredCash > this.cash) {
            logger_1.logger.info(`💰 INSUFFICIENT CASH (SHORT): Required: $${requiredCash.toFixed(2)}, Available: $${this.cash.toFixed(2)}`);
            return null;
        }
        // Create position
        const position = {
            symbol: signal.symbol,
            side: 'SHORT',
            size: quantity,
            entryPrice: currentPrice,
            entryTime: timestamp,
            currentPrice: currentPrice,
            unrealizedPnl: 0,
            leverage: this.config.leverage,
        };
        // Update cash and positions
        this.cash -= requiredCash;
        this.totalCommission += commission;
        this.positions.set(positionKey, position);
        logger_1.logger.info(`📉 Opened SHORT position`, {
            symbol: signal.symbol,
            quantity,
            price: currentPrice,
            cost: totalCost,
            commission,
            remainingCash: this.cash,
        });
        // Return trade record (entry only, will be completed when closed)
        return {
            id: (0, events_1.createEventId)(),
            symbol: signal.symbol,
            side: 'SHORT',
            entryPrice: currentPrice,
            exitPrice: 0,
            quantity,
            entryTime: timestamp,
            exitTime: 0,
            pnl: 0,
            pnlPercent: 0,
            commission,
            strategy: signal.strategy,
            reason: signal.reason,
            duration: 0,
        };
    }
    /**
     * Close a position
     */
    closePosition(positionKey, exitPrice, timestamp, reason) {
        const position = this.positions.get(positionKey);
        if (!position) {
            return null;
        }
        // Calculate P&L
        const pnl = this.calculatePnL(position, exitPrice);
        const pnlPercent = (pnl / (position.size * position.entryPrice)) * 100;
        // Calculate exit commission
        const exitCost = position.size * exitPrice;
        const exitCommission = exitCost * (this.config.commission / 100);
        const totalCommission = exitCommission; // Entry commission already deducted
        // Net P&L after commission
        const netPnl = pnl - totalCommission;
        // Update cash
        const returnedCash = (position.size * position.entryPrice) / this.config.leverage + netPnl;
        this.cash += returnedCash;
        this.totalCommission += exitCommission;
        // Create completed trade record
        const trade = {
            id: (0, events_1.createEventId)(),
            symbol: position.symbol,
            side: position.side,
            entryPrice: position.entryPrice,
            exitPrice,
            quantity: position.size,
            entryTime: position.entryTime,
            exitTime: timestamp,
            pnl: netPnl,
            pnlPercent,
            commission: totalCommission,
            strategy: 'Unknown', // Will be updated by caller
            reason,
            duration: timestamp - position.entryTime,
        };
        // Remove position
        this.positions.delete(positionKey);
        this.trades.push(trade);
        logger_1.logger.info(`💰 Closed ${position.side} position`, {
            symbol: position.symbol,
            entryPrice: position.entryPrice,
            exitPrice,
            pnl: netPnl,
            pnlPercent: pnlPercent.toFixed(2),
            duration: this.formatDuration(trade.duration),
            reason,
        });
        return trade;
    }
    /**
     * Update all positions with current market prices
     */
    updatePositions(symbol, currentPrice, timestamp) {
        for (const [key, position] of this.positions) {
            if (position.symbol === symbol) {
                position.currentPrice = currentPrice;
                position.unrealizedPnl = this.calculatePnL(position, currentPrice);
            }
        }
    }
    /**
     * Check for stop loss and take profit triggers
     */
    checkStopLossAndTakeProfit(symbol, currentPrice, timestamp, signal) {
        const closedTrades = [];
        for (const [key, position] of this.positions) {
            if (position.symbol !== symbol)
                continue;
            let shouldClose = false;
            let reason = '';
            // Check stop loss and take profit if signal provided
            if (signal?.stopLoss && signal?.takeProfit) {
                if (position.side === 'LONG') {
                    if (currentPrice <= signal.stopLoss) {
                        shouldClose = true;
                        reason = 'Stop loss triggered';
                    }
                    else if (currentPrice >= signal.takeProfit) {
                        shouldClose = true;
                        reason = 'Take profit triggered';
                    }
                }
                else { // SHORT
                    if (currentPrice >= signal.stopLoss) {
                        shouldClose = true;
                        reason = 'Stop loss triggered';
                    }
                    else if (currentPrice <= signal.takeProfit) {
                        shouldClose = true;
                        reason = 'Take profit triggered';
                    }
                }
            }
            if (shouldClose) {
                const trade = this.closePosition(key, currentPrice, timestamp, reason);
                if (trade) {
                    closedTrades.push(trade);
                }
            }
        }
        return closedTrades;
    }
    /**
     * Create portfolio snapshot
     */
    createSnapshot(timestamp) {
        const totalValue = this.getTotalPortfolioValue();
        const totalPnl = totalValue - this.config.initialCapital;
        const totalPnlPercent = (totalPnl / this.config.initialCapital) * 100;
        // Calculate drawdown
        if (totalValue > this.highWaterMark) {
            this.highWaterMark = totalValue;
        }
        const currentDrawdown = (this.highWaterMark - totalValue) / this.highWaterMark * 100;
        if (currentDrawdown > this.maxDrawdown) {
            this.maxDrawdown = currentDrawdown;
        }
        const snapshot = {
            timestamp,
            totalValue,
            cash: this.cash,
            positions: Array.from(this.positions.values()),
            totalPnl,
            totalPnlPercent,
            drawdown: currentDrawdown,
            maxDrawdown: this.maxDrawdown,
            leverage: this.config.leverage,
        };
        this.portfolioHistory.push(snapshot);
        return snapshot;
    }
    /**
     * Calculate position size based on signal and risk management
     */
    calculatePositionSize(signal, currentPrice) {
        // Use the quantity from the signal if provided, otherwise calculate
        if (signal.quantity && signal.quantity > 0) {
            return signal.quantity;
        }
        // Calculate based on risk per trade
        const riskAmount = this.config.initialCapital * (this.config.riskPerTrade / 100);
        const stopLossDistance = signal.stopLoss ? Math.abs(currentPrice - signal.stopLoss) : currentPrice * 0.02;
        let positionSize = riskAmount / stopLossDistance;
        positionSize *= this.config.leverage;
        return Math.max(positionSize, 0.001);
    }
    /**
     * Calculate P&L for a position
     */
    calculatePnL(position, currentPrice) {
        if (position.side === 'LONG') {
            return (currentPrice - position.entryPrice) * position.size;
        }
        else {
            return (position.entryPrice - currentPrice) * position.size;
        }
    }
    /**
     * Get total portfolio value
     */
    getTotalPortfolioValue() {
        let totalValue = this.cash;
        for (const position of this.positions.values()) {
            const positionValue = position.size * position.currentPrice;
            const unrealizedPnl = this.calculatePnL(position, position.currentPrice);
            totalValue += (positionValue / this.config.leverage) + unrealizedPnl;
        }
        return totalValue;
    }
    /**
     * Format duration in human readable format
     */
    formatDuration(durationMs) {
        const hours = Math.floor(durationMs / (1000 * 60 * 60));
        const minutes = Math.floor((durationMs % (1000 * 60 * 60)) / (1000 * 60));
        return `${hours}h ${minutes}m`;
    }
    // Getters
    getCash() { return this.cash; }
    getPositions() { return Array.from(this.positions.values()); }
    getTrades() { return [...this.trades]; }
    getPortfolioHistory() { return [...this.portfolioHistory]; }
    getMaxDrawdown() { return this.maxDrawdown; }
    getTotalCommission() { return this.totalCommission; }
}
exports.PortfolioManager = PortfolioManager;
//# sourceMappingURL=portfolioManager.js.map