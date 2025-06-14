import { EventEmitter } from 'events';
import prismaReadOnly from '../config/prisma-readonly';

const analyticsEmitter = new EventEmitter();

interface Trade {
  userId: string;
  symbol: string;
  entryPrice: number;
  exitPrice: number;
  quantity: number;
}

interface MetricTags {
  [key: string]: string | number | boolean;
}

interface MetricEvent {
  name: string;
  value: number;
  tags: MetricTags;
}

function calculateSharpe(returns: number[], riskFreeRate: number = 0): number {
  if (!returns.length) return 0;
  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const std = Math.sqrt(returns.map(x => Math.pow(x - mean, 2)).reduce((a, b) => a + b, 0) / returns.length);
  return std ? (mean - riskFreeRate) / std : 0;
}

async function recordMetric(name: string, value: number, tags: MetricTags = {}): Promise<void> {
  await prismaReadOnly.metric.create({
    data: {
      name,
      value,
      recordedAt: new Date(),
      tags: JSON.stringify(tags),
    },
  });
  analyticsEmitter.emit('metric', { name, value, tags });
}

async function onTradeExecuted(trade: Trade): Promise<void> {
  // Example: Compute PnL
  const pnl = trade.exitPrice - trade.entryPrice;
  await recordMetric('PnL', pnl, { userId: trade.userId, symbol: trade.symbol });
  // Compute win rate, Sharpe, drawdown, etc. (implement as needed)
  // ...
}

export { onTradeExecuted, recordMetric, analyticsEmitter, calculateSharpe, MetricEvent, Trade, MetricTags }; 