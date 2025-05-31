"use strict";
/**
 * Trading Signal Types
 * Definitions for the signal generation system
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.SignalTimeframe = exports.SignalStrength = exports.SignalDirection = exports.SignalType = void 0;
/**
 * Signal types represent different kinds of trading signals
 */
var SignalType;
(function (SignalType) {
    SignalType["ENTRY"] = "ENTRY";
    SignalType["EXIT"] = "EXIT";
    SignalType["INCREASE"] = "INCREASE";
    SignalType["DECREASE"] = "DECREASE";
    SignalType["HOLD"] = "HOLD"; // Signal to maintain current position
})(SignalType || (exports.SignalType = SignalType = {}));
/**
 * Signal direction represents the market direction
 */
var SignalDirection;
(function (SignalDirection) {
    SignalDirection["LONG"] = "LONG";
    SignalDirection["SHORT"] = "SHORT";
    SignalDirection["NEUTRAL"] = "NEUTRAL"; // Neutral signal
})(SignalDirection || (exports.SignalDirection = SignalDirection = {}));
/**
 * Signal strength represents the confidence level
 */
var SignalStrength;
(function (SignalStrength) {
    SignalStrength["VERY_WEAK"] = "VERY_WEAK";
    SignalStrength["WEAK"] = "WEAK";
    SignalStrength["MODERATE"] = "MODERATE";
    SignalStrength["STRONG"] = "STRONG";
    SignalStrength["VERY_STRONG"] = "VERY_STRONG"; // 80-100% confidence
})(SignalStrength || (exports.SignalStrength = SignalStrength = {}));
/**
 * Signal timeframe represents the expected duration
 */
var SignalTimeframe;
(function (SignalTimeframe) {
    SignalTimeframe["VERY_SHORT"] = "VERY_SHORT";
    SignalTimeframe["SHORT"] = "SHORT";
    SignalTimeframe["MEDIUM"] = "MEDIUM";
    SignalTimeframe["LONG"] = "LONG";
    SignalTimeframe["VERY_LONG"] = "VERY_LONG"; // Weeks or longer
})(SignalTimeframe || (exports.SignalTimeframe = SignalTimeframe = {}));
//# sourceMappingURL=signals.js.map