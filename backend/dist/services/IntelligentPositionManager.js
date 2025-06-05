"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.IntelligentPositionManager = void 0;
const logger_1 = require("../utils/logger");
class IntelligentPositionManager {
    constructor() {
        this.marketData = new Map();
        this.positionHistory = new Map();
        logger_1.logger.info('🧠 Initializing Intelligent Position Manager');
    }
    /**
     * Analyze position health across multiple timeframes
     */
    analyzePositionHealth(position, marketData) {
        const technical = this.calculateTechnicalIndicators(marketData);
        const regime = this.detectMarketRegime(marketData);
        // Multi-timeframe trend analysis
        const shortTrend = this.analyzeTrend(marketData.slice(-15)); // 15 periods
        const mediumTrend = this.analyzeTrend(marketData.slice(-50)); // 50 periods
        const longTrend = this.analyzeTrend(marketData.slice(-200)); // 200 periods
        // Calculate trend alignment score
        const trendAlignment = this.calculateTrendAlignment(position, shortTrend, mediumTrend, longTrend);
        // Position duration factor
        const positionAge = Date.now() - position.entryTime;
        const ageFactor = this.calculateAgeFactor(positionAge, regime);
        // PnL momentum
        const pnlMomentum = this.calculatePnLMomentum(position, marketData);
        // Overall health score
        const healthScore = this.calculateHealthScore({
            trendAlignment,
            ageFactor,
            pnlMomentum,
            volatility: technical.volatility,
            rsi: technical.rsi
        });
        return {
            score: healthScore,
            trend_alignment: trendAlignment,
            risk_level: this.assessRiskLevel(healthScore, technical.volatility),
            recommended_action: this.getRecommendedAction(healthScore, trendAlignment, pnlMomentum),
            confidence: this.calculateConfidence(technical, regime)
        };
    }
    /**
     * Calculate dynamic stop loss based on ATR and market conditions
     */
    calculateDynamicStopLoss(position, marketData) {
        const atr = this.calculateATR(marketData, 14);
        const regime = this.detectMarketRegime(marketData);
        // Adaptive multiplier based on market regime
        let atrMultiplier = 2.0; // Base multiplier
        switch (regime.type) {
            case 'trending':
                atrMultiplier = 1.5; // Tighter stops in trending markets
                break;
            case 'ranging':
                atrMultiplier = 2.5; // Wider stops in ranging markets
                break;
            case 'volatile':
                atrMultiplier = 3.0; // Much wider stops in volatile markets
                break;
            case 'low_volatility':
                atrMultiplier = 1.2; // Very tight stops in low volatility
                break;
        }
        // Adjust based on position performance
        const pnlPercent = (position.unrealizedPnl / (position.entryPrice * Math.abs(position.size))) * 100;
        if (pnlPercent > 2) {
            // Trailing stop for profitable positions
            atrMultiplier *= 0.8;
        }
        else if (pnlPercent < -1) {
            // Wider stop for losing positions (give room to recover)
            atrMultiplier *= 1.2;
        }
        const stopDistance = atr * atrMultiplier;
        if (position.size > 0) {
            return position.currentPrice - stopDistance;
        }
        else {
            return position.currentPrice + stopDistance;
        }
    }
    /**
     * Calculate dynamic take profit levels
     */
    calculateDynamicTakeProfit(position, marketData) {
        const technical = this.calculateTechnicalIndicators(marketData);
        const regime = this.detectMarketRegime(marketData);
        // Base take profit levels
        const baseTpLevels = [1.5, 3.0, 5.0]; // 1.5%, 3%, 5%
        // Adjust based on market regime
        let multiplier = 1.0;
        switch (regime.type) {
            case 'trending':
                multiplier = 1.5; // Higher targets in trending markets
                break;
            case 'ranging':
                multiplier = 0.8; // Lower targets in ranging markets
                break;
            case 'volatile':
                multiplier = 1.2; // Moderate targets in volatile markets
                break;
        }
        // Calculate actual TP levels
        const tpLevels = baseTpLevels.map(level => {
            const adjustedLevel = level * multiplier;
            if (position.size > 0) {
                return position.entryPrice * (1 + adjustedLevel / 100);
            }
            else {
                return position.entryPrice * (1 - adjustedLevel / 100);
            }
        });
        return tpLevels;
    }
    /**
     * Determine if position should be held, reduced, or closed
     */
    getPositionAction(position, marketData) {
        const health = this.analyzePositionHealth(position, marketData);
        const technical = this.calculateTechnicalIndicators(marketData);
        // Critical exit conditions
        if (health.score < 20) {
            return { action: 'close', reason: 'Poor position health score' };
        }
        if (Math.abs(health.trend_alignment) < 0.3 && technical.volatility > 0.8) {
            return { action: 'reduce', percentage: 50, reason: 'Trend misalignment in volatile market' };
        }
        // Profit taking conditions
        const pnlPercent = (position.unrealizedPnl / (position.entryPrice * Math.abs(position.size))) * 100;
        if (pnlPercent > 5 && health.trend_alignment > 0.7) {
            return { action: 'reduce', percentage: 25, reason: 'Partial profit taking at strong trend' };
        }
        if (pnlPercent > 10) {
            return { action: 'reduce', percentage: 50, reason: 'Significant profit protection' };
        }
        // Position addition conditions
        if (health.score > 80 && health.trend_alignment > 0.8 && pnlPercent > 1) {
            return { action: 'add', percentage: 25, reason: 'Strong trend continuation signal' };
        }
        return { action: 'hold', reason: 'Position within acceptable parameters' };
    }
    // Helper methods
    calculateATR(data, period) {
        if (data.length < period + 1)
            return 0;
        let atrSum = 0;
        for (let i = data.length - period; i < data.length; i++) {
            const high = data[i].high;
            const low = data[i].low;
            const prevClose = i > 0 ? data[i - 1].close : data[i].open;
            const tr = Math.max(high - low, Math.abs(high - prevClose), Math.abs(low - prevClose));
            atrSum += tr;
        }
        return atrSum / period;
    }
    calculateTechnicalIndicators(data) {
        const atr = this.calculateATR(data, 14);
        const rsi = this.calculateRSI(data, 14);
        const trend = this.analyzeTrend(data.slice(-50));
        // Simple support/resistance calculation
        const recentData = data.slice(-20);
        const support = Math.min(...recentData.map(d => d.low));
        const resistance = Math.max(...recentData.map(d => d.high));
        // Volatility calculation
        const returns = data.slice(-20).map((d, i) => i > 0 ? (d.close - data[data.length - 20 + i - 1].close) / data[data.length - 20 + i - 1].close : 0).slice(1);
        const volatility = Math.sqrt(returns.reduce((sum, r) => sum + r * r, 0) / returns.length);
        return { atr, rsi, trend, support, resistance, volatility };
    }
    calculateRSI(data, period) {
        if (data.length < period + 1)
            return 50;
        let gains = 0;
        let losses = 0;
        for (let i = data.length - period; i < data.length; i++) {
            const change = data[i].close - data[i - 1].close;
            if (change > 0) {
                gains += change;
            }
            else {
                losses += Math.abs(change);
            }
        }
        const avgGain = gains / period;
        const avgLoss = losses / period;
        if (avgLoss === 0)
            return 100;
        const rs = avgGain / avgLoss;
        return 100 - (100 / (1 + rs));
    }
    analyzeTrend(data) {
        if (data.length < 10)
            return 'sideways';
        const firstPrice = data[0].close;
        const lastPrice = data[data.length - 1].close;
        const change = (lastPrice - firstPrice) / firstPrice;
        if (change > 0.02)
            return 'bullish';
        if (change < -0.02)
            return 'bearish';
        return 'sideways';
    }
    detectMarketRegime(data) {
        const volatility = this.calculateTechnicalIndicators(data).volatility;
        const trend = this.analyzeTrend(data);
        if (volatility > 0.05) {
            return { type: 'volatile', strength: volatility * 10, duration: 60 };
        }
        else if (volatility < 0.01) {
            return { type: 'low_volatility', strength: 1 - volatility * 100, duration: 120 };
        }
        else if (trend !== 'sideways') {
            return { type: 'trending', strength: 0.7, duration: 240 };
        }
        else {
            return { type: 'ranging', strength: 0.5, duration: 180 };
        }
    }
    calculateTrendAlignment(position, short, medium, long) {
        const positionDirection = position.size > 0 ? 1 : -1;
        let alignment = 0;
        // Short-term alignment (40% weight)
        if ((short === 'bullish' && positionDirection > 0) || (short === 'bearish' && positionDirection < 0)) {
            alignment += 0.4;
        }
        else if (short === 'sideways') {
            alignment += 0.2;
        }
        // Medium-term alignment (35% weight)
        if ((medium === 'bullish' && positionDirection > 0) || (medium === 'bearish' && positionDirection < 0)) {
            alignment += 0.35;
        }
        else if (medium === 'sideways') {
            alignment += 0.175;
        }
        // Long-term alignment (25% weight)
        if ((long === 'bullish' && positionDirection > 0) || (long === 'bearish' && positionDirection < 0)) {
            alignment += 0.25;
        }
        else if (long === 'sideways') {
            alignment += 0.125;
        }
        return alignment * 2 - 1; // Convert to -1 to 1 range
    }
    calculateAgeFactor(ageMs, regime) {
        const ageMinutes = ageMs / (1000 * 60);
        // Optimal holding time based on market regime
        let optimalTime = 60; // Default 1 hour
        switch (regime.type) {
            case 'trending':
                optimalTime = 240; // 4 hours for trending markets
                break;
            case 'ranging':
                optimalTime = 30; // 30 minutes for ranging markets
                break;
            case 'volatile':
                optimalTime = 15; // 15 minutes for volatile markets
                break;
        }
        // Calculate factor (1.0 at optimal time, decreases as time deviates)
        const timeFactor = Math.exp(-Math.pow((ageMinutes - optimalTime) / optimalTime, 2));
        return timeFactor;
    }
    calculatePnLMomentum(position, data) {
        // Simple momentum calculation based on recent price movement
        if (data.length < 5)
            return 0;
        const recentData = data.slice(-5);
        const priceChange = (recentData[recentData.length - 1].close - recentData[0].close) / recentData[0].close;
        const positionDirection = position.size > 0 ? 1 : -1;
        return priceChange * positionDirection;
    }
    calculateHealthScore(factors) {
        const { trendAlignment, ageFactor, pnlMomentum, volatility, rsi } = factors;
        let score = 50; // Base score
        // Trend alignment (30% weight)
        score += trendAlignment * 30;
        // Age factor (20% weight)
        score += ageFactor * 20;
        // PnL momentum (25% weight)
        score += pnlMomentum * 100 * 0.25;
        // RSI factor (15% weight)
        const rsiFactor = Math.abs(rsi - 50) / 50; // 0 to 1, where 1 is extreme
        score += (1 - rsiFactor) * 15;
        // Volatility factor (10% weight)
        score += Math.max(0, (1 - volatility * 10)) * 10;
        return Math.max(0, Math.min(100, score));
    }
    assessRiskLevel(score, volatility) {
        if (score > 70 && volatility < 0.03)
            return 'low';
        if (score > 50 && volatility < 0.05)
            return 'medium';
        return 'high';
    }
    getRecommendedAction(score, alignment, momentum) {
        if (score < 30)
            return 'close';
        if (score < 50 || alignment < -0.5)
            return 'reduce';
        if (score > 80 && alignment > 0.7 && momentum > 0.02)
            return 'add';
        return 'hold';
    }
    calculateConfidence(technical, regime) {
        let confidence = 0.5; // Base confidence
        // Higher confidence in trending markets
        if (regime.type === 'trending')
            confidence += 0.2;
        // Higher confidence with clear RSI signals
        if (technical.rsi > 70 || technical.rsi < 30)
            confidence += 0.15;
        // Lower confidence in volatile markets
        if (regime.type === 'volatile')
            confidence -= 0.2;
        return Math.max(0.1, Math.min(0.9, confidence));
    }
}
exports.IntelligentPositionManager = IntelligentPositionManager;
//# sourceMappingURL=IntelligentPositionManager.js.map