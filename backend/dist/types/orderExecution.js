"use strict";
/**
 * Order Execution Types
 * Interfaces for the order execution service
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.ExecutionSource = exports.TimeInForce = exports.OrderSide = exports.OrderType = exports.OrderExecutionStatus = void 0;
// The DeltaExchange import isn't needed for now, removing it
// import { DeltaExchange } from './deltaExchange';
/**
 * Order execution status
 */
var OrderExecutionStatus;
(function (OrderExecutionStatus) {
    OrderExecutionStatus["PENDING"] = "PENDING";
    OrderExecutionStatus["SUBMITTED"] = "SUBMITTED";
    OrderExecutionStatus["PARTIALLY_FILLED"] = "PARTIALLY_FILLED";
    OrderExecutionStatus["FILLED"] = "FILLED";
    OrderExecutionStatus["CANCELLED"] = "CANCELLED";
    OrderExecutionStatus["REJECTED"] = "REJECTED";
    OrderExecutionStatus["EXPIRED"] = "EXPIRED"; // Order has expired
})(OrderExecutionStatus || (exports.OrderExecutionStatus = OrderExecutionStatus = {}));
/**
 * Order type
 */
var OrderType;
(function (OrderType) {
    OrderType["MARKET"] = "MARKET";
    OrderType["LIMIT"] = "LIMIT";
    OrderType["STOP"] = "STOP";
    OrderType["STOP_LIMIT"] = "STOP_LIMIT";
    OrderType["TRAILING_STOP"] = "TRAILING_STOP";
    OrderType["OCO"] = "OCO"; // One-cancels-the-other order
})(OrderType || (exports.OrderType = OrderType = {}));
/**
 * Order side
 */
var OrderSide;
(function (OrderSide) {
    OrderSide["BUY"] = "BUY";
    OrderSide["SELL"] = "SELL"; // Sell order
})(OrderSide || (exports.OrderSide = OrderSide = {}));
/**
 * Time in force
 */
var TimeInForce;
(function (TimeInForce) {
    TimeInForce["GTC"] = "GTC";
    TimeInForce["IOC"] = "IOC";
    TimeInForce["FOK"] = "FOK"; // Fill or kill
})(TimeInForce || (exports.TimeInForce = TimeInForce = {}));
/**
 * Execution source
 */
var ExecutionSource;
(function (ExecutionSource) {
    ExecutionSource["MANUAL"] = "MANUAL";
    ExecutionSource["STRATEGY"] = "STRATEGY";
    ExecutionSource["BOT"] = "BOT";
    ExecutionSource["SIGNAL"] = "SIGNAL";
    ExecutionSource["SYSTEM"] = "SYSTEM"; // Order from system operations
})(ExecutionSource || (exports.ExecutionSource = ExecutionSource = {}));
//# sourceMappingURL=orderExecution.js.map