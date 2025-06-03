"use strict";
/**
 * Event Schema and Types for Event-Driven Architecture
 * Defines all event types and schemas for Redis Streams
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.ProcessingStatus = exports.EventPriority = exports.CONSUMER_GROUPS = exports.STREAM_NAMES = void 0;
exports.createEventId = createEventId;
exports.createCorrelationId = createCorrelationId;
exports.isMarketDataEvent = isMarketDataEvent;
exports.isTradingSignalEvent = isTradingSignalEvent;
exports.isOrderEvent = isOrderEvent;
exports.isRiskEvent = isRiskEvent;
exports.isSystemEvent = isSystemEvent;
exports.isBotEvent = isBotEvent;
// Event stream names
exports.STREAM_NAMES = {
    MARKET_DATA: 'market-data-stream',
    TRADING_SIGNALS: 'trading-signals-stream',
    ML_PREDICTIONS: 'ml-predictions-stream',
    ORDERS: 'orders-stream',
    POSITIONS: 'positions-stream',
    RISK_MANAGEMENT: 'risk-management-stream',
    PORTFOLIO: 'portfolio-stream',
    SYSTEM: 'system-stream',
    PERFORMANCE: 'performance-stream',
    BOTS: 'bots-stream',
};
// Consumer group names
exports.CONSUMER_GROUPS = {
    SIGNAL_PROCESSOR: 'signal-processor-group',
    ORDER_EXECUTOR: 'order-executor-group',
    RISK_MANAGER: 'risk-manager-group',
    PORTFOLIO_MANAGER: 'portfolio-manager-group',
    ANALYTICS: 'analytics-group',
    MONITORING: 'monitoring-group',
    NOTIFICATION: 'notification-group',
};
// Event priorities
var EventPriority;
(function (EventPriority) {
    EventPriority[EventPriority["LOW"] = 1] = "LOW";
    EventPriority[EventPriority["NORMAL"] = 2] = "NORMAL";
    EventPriority[EventPriority["HIGH"] = 3] = "HIGH";
    EventPriority[EventPriority["CRITICAL"] = 4] = "CRITICAL";
})(EventPriority || (exports.EventPriority = EventPriority = {}));
// Event processing status
var ProcessingStatus;
(function (ProcessingStatus) {
    ProcessingStatus["PENDING"] = "PENDING";
    ProcessingStatus["PROCESSING"] = "PROCESSING";
    ProcessingStatus["COMPLETED"] = "COMPLETED";
    ProcessingStatus["FAILED"] = "FAILED";
    ProcessingStatus["RETRYING"] = "RETRYING";
    ProcessingStatus["DEAD_LETTER"] = "DEAD_LETTER";
})(ProcessingStatus || (exports.ProcessingStatus = ProcessingStatus = {}));
// Utility functions
function createEventId() {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}
function createCorrelationId() {
    return `corr-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}
function isMarketDataEvent(event) {
    return event.type.startsWith('MARKET_DATA_');
}
function isTradingSignalEvent(event) {
    return event.type.startsWith('SIGNAL_');
}
function isOrderEvent(event) {
    return event.type.startsWith('ORDER_');
}
function isRiskEvent(event) {
    return event.type.startsWith('RISK_');
}
function isSystemEvent(event) {
    return event.type.startsWith('SYSTEM_');
}
function isBotEvent(event) {
    return event.type.startsWith('BOT_');
}
//# sourceMappingURL=events.js.map