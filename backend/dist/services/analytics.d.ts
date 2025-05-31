import { EventEmitter } from 'events';
declare const analyticsEmitter: EventEmitter<[never]>;
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
declare function calculateSharpe(returns: number[], riskFreeRate?: number): number;
declare function recordMetric(name: string, value: number, tags?: MetricTags): Promise<void>;
declare function onTradeExecuted(trade: Trade): Promise<void>;
export { onTradeExecuted, recordMetric, analyticsEmitter, calculateSharpe, MetricEvent, Trade, MetricTags };
