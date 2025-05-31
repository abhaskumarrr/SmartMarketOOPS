"use strict";
/**
 * Risk Management Types
 * Definitions for the risk management system
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.RiskAlertType = exports.TakeProfitType = exports.StopLossType = exports.PositionSizingMethod = exports.RiskLevel = void 0;
/**
 * Risk level enum for portfolio and positions
 */
var RiskLevel;
(function (RiskLevel) {
    RiskLevel["VERY_LOW"] = "VERY_LOW";
    RiskLevel["LOW"] = "LOW";
    RiskLevel["MODERATE"] = "MODERATE";
    RiskLevel["HIGH"] = "HIGH";
    RiskLevel["VERY_HIGH"] = "VERY_HIGH"; // 80-100% of max allowed risk
})(RiskLevel || (exports.RiskLevel = RiskLevel = {}));
/**
 * Position sizing methods
 */
var PositionSizingMethod;
(function (PositionSizingMethod) {
    PositionSizingMethod["FIXED_FRACTIONAL"] = "FIXED_FRACTIONAL";
    PositionSizingMethod["KELLY_CRITERION"] = "KELLY_CRITERION";
    PositionSizingMethod["FIXED_RATIO"] = "FIXED_RATIO";
    PositionSizingMethod["FIXED_AMOUNT"] = "FIXED_AMOUNT";
    PositionSizingMethod["VOLATILITY_BASED"] = "VOLATILITY_BASED";
    PositionSizingMethod["CUSTOM"] = "CUSTOM"; // Custom sizing algorithm
})(PositionSizingMethod || (exports.PositionSizingMethod = PositionSizingMethod = {}));
/**
 * Stop loss types
 */
var StopLossType;
(function (StopLossType) {
    StopLossType["FIXED"] = "FIXED";
    StopLossType["PERCENTAGE"] = "PERCENTAGE";
    StopLossType["ATR_MULTIPLE"] = "ATR_MULTIPLE";
    StopLossType["SUPPORT_RESISTANCE"] = "SUPPORT_RESISTANCE";
    StopLossType["VOLATILITY_BASED"] = "VOLATILITY_BASED";
    StopLossType["TRAILING"] = "TRAILING";
    StopLossType["TIME_BASED"] = "TIME_BASED";
    StopLossType["MARTINGALE"] = "MARTINGALE";
    StopLossType["PYRAMID"] = "PYRAMID"; // Decrease position on loss
})(StopLossType || (exports.StopLossType = StopLossType = {}));
/**
 * Take profit types
 */
var TakeProfitType;
(function (TakeProfitType) {
    TakeProfitType["FIXED"] = "FIXED";
    TakeProfitType["PERCENTAGE"] = "PERCENTAGE";
    TakeProfitType["RISK_REWARD"] = "RISK_REWARD";
    TakeProfitType["TRAILING"] = "TRAILING";
    TakeProfitType["PARTIAL"] = "PARTIAL";
    TakeProfitType["RESISTANCE_LEVEL"] = "RESISTANCE_LEVEL";
    TakeProfitType["PARABOLIC"] = "PARABOLIC";
    TakeProfitType["VOLATILITY_BASED"] = "VOLATILITY_BASED";
    TakeProfitType["PROFIT_TARGET"] = "PROFIT_TARGET"; // Fixed profit target
})(TakeProfitType || (exports.TakeProfitType = TakeProfitType = {}));
/**
 * Risk alert types
 */
var RiskAlertType;
(function (RiskAlertType) {
    RiskAlertType["MARGIN_CALL"] = "MARGIN_CALL";
    RiskAlertType["HIGH_EXPOSURE"] = "HIGH_EXPOSURE";
    RiskAlertType["DRAWDOWN_WARNING"] = "DRAWDOWN_WARNING";
    RiskAlertType["CONCENTRATION_RISK"] = "CONCENTRATION_RISK";
    RiskAlertType["VOLATILITY_SPIKE"] = "VOLATILITY_SPIKE";
    RiskAlertType["CIRCUIT_BREAKER"] = "CIRCUIT_BREAKER";
    RiskAlertType["CORRELATION_RISK"] = "CORRELATION_RISK";
    RiskAlertType["DAILY_LOSS_WARNING"] = "DAILY_LOSS_WARNING";
    RiskAlertType["POSITION_SIZE_WARNING"] = "POSITION_SIZE_WARNING";
    RiskAlertType["TRADE_FREQUENCY_WARNING"] = "TRADE_FREQUENCY_WARNING";
    RiskAlertType["STOP_DISTANCE_WARNING"] = "STOP_DISTANCE_WARNING";
    RiskAlertType["WEEKEND_RISK"] = "WEEKEND_RISK";
    RiskAlertType["API_CONNECTION_WARNING"] = "API_CONNECTION_WARNING";
    RiskAlertType["EXTERNAL_EVENT_RISK"] = "EXTERNAL_EVENT_RISK";
    RiskAlertType["LIQUIDITY_RISK"] = "LIQUIDITY_RISK"; // Low liquidity warning
})(RiskAlertType || (exports.RiskAlertType = RiskAlertType = {}));
//# sourceMappingURL=risk.js.map