"use strict";
/**
 * Technical Analysis Utilities
 * Implements common technical indicators for trading strategies
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.technicalAnalysis = exports.TechnicalAnalysisService = void 0;
class TechnicalAnalysisService {
    /**
     * Calculate Simple Moving Average
     */
    calculateSMA(prices, period) {
        const sma = [];
        for (let i = 0; i < prices.length; i++) {
            if (i < period - 1) {
                sma.push(NaN);
            }
            else {
                const sum = prices.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
                sma.push(sum / period);
            }
        }
        return sma;
    }
    /**
     * Calculate Exponential Moving Average
     */
    calculateEMA(prices, period) {
        const ema = [];
        const multiplier = 2 / (period + 1);
        for (let i = 0; i < prices.length; i++) {
            if (i === 0) {
                ema.push(prices[0]);
            }
            else {
                ema.push((prices[i] * multiplier) + (ema[i - 1] * (1 - multiplier)));
            }
        }
        return ema;
    }
    /**
     * Calculate Relative Strength Index
     */
    calculateRSI(prices, period = 14) {
        const rsi = [];
        const gains = [];
        const losses = [];
        // Calculate price changes
        for (let i = 1; i < prices.length; i++) {
            const change = prices[i] - prices[i - 1];
            gains.push(change > 0 ? change : 0);
            losses.push(change < 0 ? Math.abs(change) : 0);
        }
        // Calculate RSI
        for (let i = 0; i < gains.length; i++) {
            if (i < period - 1) {
                rsi.push(NaN);
            }
            else {
                const avgGain = gains.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0) / period;
                const avgLoss = losses.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0) / period;
                if (avgLoss === 0) {
                    rsi.push(100);
                }
                else {
                    const rs = avgGain / avgLoss;
                    rsi.push(100 - (100 / (1 + rs)));
                }
            }
        }
        // Add NaN for first price (no change calculated)
        return [NaN, ...rsi];
    }
    /**
     * Calculate MACD (Moving Average Convergence Divergence)
     */
    calculateMACD(prices, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) {
        const fastEMA = this.calculateEMA(prices, fastPeriod);
        const slowEMA = this.calculateEMA(prices, slowPeriod);
        // Calculate MACD line
        const macd = fastEMA.map((fast, i) => fast - slowEMA[i]);
        // Calculate signal line (EMA of MACD)
        const signal = this.calculateEMA(macd.filter(val => !isNaN(val)), signalPeriod);
        // Pad signal array to match MACD length
        const paddedSignal = [...Array(macd.length - signal.length).fill(NaN), ...signal];
        // Calculate histogram
        const histogram = macd.map((macdVal, i) => macdVal - paddedSignal[i]);
        return { macd, signal: paddedSignal, histogram };
    }
    /**
     * Calculate Bollinger Bands
     */
    calculateBollingerBands(prices, period = 20, stdDev = 2) {
        const middle = this.calculateSMA(prices, period);
        const upper = [];
        const lower = [];
        for (let i = 0; i < prices.length; i++) {
            if (i < period - 1) {
                upper.push(NaN);
                lower.push(NaN);
            }
            else {
                const slice = prices.slice(i - period + 1, i + 1);
                const mean = middle[i];
                const variance = slice.reduce((sum, price) => sum + Math.pow(price - mean, 2), 0) / period;
                const standardDeviation = Math.sqrt(variance);
                upper.push(mean + (stdDev * standardDeviation));
                lower.push(mean - (stdDev * standardDeviation));
            }
        }
        return { upper, middle, lower };
    }
    /**
     * Calculate all indicators for a price series
     */
    calculateAllIndicators(prices, volumes) {
        const sma20 = this.calculateSMA(prices, 20);
        const sma50 = this.calculateSMA(prices, 50);
        const ema12 = this.calculateEMA(prices, 12);
        const ema26 = this.calculateEMA(prices, 26);
        const rsi = this.calculateRSI(prices, 14);
        const macd = this.calculateMACD(prices, 12, 26, 9);
        const bollinger = this.calculateBollingerBands(prices, 20, 2);
        let volumeSMA = [];
        if (volumes) {
            volumeSMA = this.calculateSMA(volumes, 20);
        }
        const indicators = [];
        for (let i = 0; i < prices.length; i++) {
            indicators.push({
                sma_20: sma20[i],
                sma_50: sma50[i],
                ema_12: ema12[i],
                ema_26: ema26[i],
                rsi: rsi[i],
                macd: macd.macd[i],
                macd_signal: macd.signal[i],
                macd_histogram: macd.histogram[i],
                bollinger_upper: bollinger.upper[i],
                bollinger_middle: bollinger.middle[i],
                bollinger_lower: bollinger.lower[i],
                volume_sma: volumes ? volumeSMA[i] : undefined,
            });
        }
        return indicators;
    }
    /**
     * Calculate price momentum
     */
    calculateMomentum(prices, period = 10) {
        const momentum = [];
        for (let i = 0; i < prices.length; i++) {
            if (i < period) {
                momentum.push(NaN);
            }
            else {
                momentum.push(((prices[i] - prices[i - period]) / prices[i - period]) * 100);
            }
        }
        return momentum;
    }
    /**
     * Calculate Average True Range (ATR)
     */
    calculateATR(highs, lows, closes, period = 14) {
        const trueRanges = [];
        for (let i = 1; i < highs.length; i++) {
            const tr1 = highs[i] - lows[i];
            const tr2 = Math.abs(highs[i] - closes[i - 1]);
            const tr3 = Math.abs(lows[i] - closes[i - 1]);
            trueRanges.push(Math.max(tr1, tr2, tr3));
        }
        // Calculate ATR as SMA of True Ranges
        const atr = [NaN, ...this.calculateSMA(trueRanges, period)];
        return atr;
    }
    /**
     * Calculate Stochastic Oscillator
     */
    calculateStochastic(highs, lows, closes, kPeriod = 14, dPeriod = 3) {
        const k = [];
        for (let i = 0; i < closes.length; i++) {
            if (i < kPeriod - 1) {
                k.push(NaN);
            }
            else {
                const periodHigh = Math.max(...highs.slice(i - kPeriod + 1, i + 1));
                const periodLow = Math.min(...lows.slice(i - kPeriod + 1, i + 1));
                const currentClose = closes[i];
                if (periodHigh === periodLow) {
                    k.push(50); // Avoid division by zero
                }
                else {
                    k.push(((currentClose - periodLow) / (periodHigh - periodLow)) * 100);
                }
            }
        }
        // Calculate %D as SMA of %K
        const d = this.calculateSMA(k.filter(val => !isNaN(val)), dPeriod);
        const paddedD = [...Array(k.length - d.length).fill(NaN), ...d];
        return { k, d: paddedD };
    }
    /**
     * Detect crossover signals
     */
    detectCrossover(series1, series2, lookback = 1) {
        const signals = [];
        for (let i = 0; i < series1.length; i++) {
            if (i < lookback || isNaN(series1[i]) || isNaN(series2[i]) ||
                isNaN(series1[i - lookback]) || isNaN(series2[i - lookback])) {
                signals.push('none');
            }
            else {
                const currentAbove = series1[i] > series2[i];
                const previousAbove = series1[i - lookback] > series2[i - lookback];
                if (!previousAbove && currentAbove) {
                    signals.push('bullish');
                }
                else if (previousAbove && !currentAbove) {
                    signals.push('bearish');
                }
                else {
                    signals.push('none');
                }
            }
        }
        return signals;
    }
    /**
     * Calculate support and resistance levels
     */
    calculateSupportResistance(highs, lows, period = 20) {
        const support = [];
        const resistance = [];
        for (let i = 0; i < highs.length; i++) {
            if (i < period) {
                support.push(NaN);
                resistance.push(NaN);
            }
            else {
                const periodLows = lows.slice(i - period, i + 1);
                const periodHighs = highs.slice(i - period, i + 1);
                support.push(Math.min(...periodLows));
                resistance.push(Math.max(...periodHighs));
            }
        }
        return { support, resistance };
    }
}
exports.TechnicalAnalysisService = TechnicalAnalysisService;
// Export singleton instance
exports.technicalAnalysis = new TechnicalAnalysisService();
//# sourceMappingURL=technicalAnalysis.js.map