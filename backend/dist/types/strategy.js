"use strict";
/**
 * Trading Strategy Types
 * Definitions for the strategy execution system
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.ExitRuleType = exports.EntryRuleType = exports.StrategyExecutionStatus = exports.StrategyTimeHorizon = exports.StrategyType = void 0;
/**
 * Strategy types represent different trading approaches
 */
var StrategyType;
(function (StrategyType) {
    StrategyType["TREND_FOLLOWING"] = "TREND_FOLLOWING";
    StrategyType["MEAN_REVERSION"] = "MEAN_REVERSION";
    StrategyType["BREAKOUT"] = "BREAKOUT";
    StrategyType["MOMENTUM"] = "MOMENTUM";
    StrategyType["ARBITRAGE"] = "ARBITRAGE";
    StrategyType["GRID"] = "GRID";
    StrategyType["MARTINGALE"] = "MARTINGALE";
    StrategyType["ML_PREDICTION"] = "ML_PREDICTION";
    StrategyType["CUSTOM"] = "CUSTOM"; // Custom strategy
})(StrategyType || (exports.StrategyType = StrategyType = {}));
/**
 * Strategy time horizons
 */
var StrategyTimeHorizon;
(function (StrategyTimeHorizon) {
    StrategyTimeHorizon["SCALPING"] = "SCALPING";
    StrategyTimeHorizon["INTRADAY"] = "INTRADAY";
    StrategyTimeHorizon["SWING"] = "SWING";
    StrategyTimeHorizon["POSITION"] = "POSITION";
    StrategyTimeHorizon["LONG_TERM"] = "LONG_TERM"; // Months to years
})(StrategyTimeHorizon || (exports.StrategyTimeHorizon = StrategyTimeHorizon = {}));
/**
 * Strategy execution status
 */
var StrategyExecutionStatus;
(function (StrategyExecutionStatus) {
    StrategyExecutionStatus["ACTIVE"] = "ACTIVE";
    StrategyExecutionStatus["PAUSED"] = "PAUSED";
    StrategyExecutionStatus["STOPPED"] = "STOPPED";
    StrategyExecutionStatus["BACKTEST"] = "BACKTEST";
    StrategyExecutionStatus["SIMULATION"] = "SIMULATION";
    StrategyExecutionStatus["ERROR"] = "ERROR"; // Strategy encountered an error
})(StrategyExecutionStatus || (exports.StrategyExecutionStatus = StrategyExecutionStatus = {}));
/**
 * Entry rule types
 */
var EntryRuleType;
(function (EntryRuleType) {
    EntryRuleType["SIGNAL_BASED"] = "SIGNAL_BASED";
    EntryRuleType["PRICE_BREAKOUT"] = "PRICE_BREAKOUT";
    EntryRuleType["INDICATOR_CROSS"] = "INDICATOR_CROSS";
    EntryRuleType["PATTERN_MATCH"] = "PATTERN_MATCH";
    EntryRuleType["TIME_BASED"] = "TIME_BASED";
    EntryRuleType["VOLUME_SPIKE"] = "VOLUME_SPIKE";
    EntryRuleType["ML_PREDICTION"] = "ML_PREDICTION";
    EntryRuleType["CUSTOM"] = "CUSTOM"; // Custom rule
})(EntryRuleType || (exports.EntryRuleType = EntryRuleType = {}));
/**
 * Exit rule types
 */
var ExitRuleType;
(function (ExitRuleType) {
    ExitRuleType["SIGNAL_BASED"] = "SIGNAL_BASED";
    ExitRuleType["STOP_LOSS"] = "STOP_LOSS";
    ExitRuleType["TAKE_PROFIT"] = "TAKE_PROFIT";
    ExitRuleType["TRAILING_STOP"] = "TRAILING_STOP";
    ExitRuleType["TIME_BASED"] = "TIME_BASED";
    ExitRuleType["INDICATOR_BASED"] = "INDICATOR_BASED";
    ExitRuleType["ML_PREDICTION"] = "ML_PREDICTION";
    ExitRuleType["CUSTOM"] = "CUSTOM"; // Custom rule
})(ExitRuleType || (exports.ExitRuleType = ExitRuleType = {}));
//# sourceMappingURL=strategy.js.map