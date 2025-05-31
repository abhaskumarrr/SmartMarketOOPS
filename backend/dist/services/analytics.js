"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.analyticsEmitter = void 0;
exports.onTradeExecuted = onTradeExecuted;
exports.recordMetric = recordMetric;
exports.calculateSharpe = calculateSharpe;
const events_1 = require("events");
const prismaClient_1 = __importDefault(require("../utils/prismaClient"));
const analyticsEmitter = new events_1.EventEmitter();
exports.analyticsEmitter = analyticsEmitter;
function calculateSharpe(returns, riskFreeRate = 0) {
    if (!returns.length)
        return 0;
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const std = Math.sqrt(returns.map(x => Math.pow(x - mean, 2)).reduce((a, b) => a + b, 0) / returns.length);
    return std ? (mean - riskFreeRate) / std : 0;
}
async function recordMetric(name, value, tags = {}) {
    await prismaClient_1.default.metric.create({
        data: {
            name,
            value,
            recordedAt: new Date(),
            tags: JSON.stringify(tags),
        },
    });
    analyticsEmitter.emit('metric', { name, value, tags });
}
async function onTradeExecuted(trade) {
    // Example: Compute PnL
    const pnl = trade.exitPrice - trade.entryPrice;
    await recordMetric('PnL', pnl, { userId: trade.userId, symbol: trade.symbol });
    // Compute win rate, Sharpe, drawdown, etc. (implement as needed)
    // ...
}
//# sourceMappingURL=analytics.js.map