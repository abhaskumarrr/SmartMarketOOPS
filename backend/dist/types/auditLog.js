"use strict";
/**
 * Audit Log Types
 * Type definitions for the audit logging and decision logging system
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.AuditEventStatus = exports.AuditEventType = exports.AuditTrailStatus = exports.AuditTrailType = exports.DecisionImportance = exports.DecisionOutcome = exports.DecisionActionType = exports.DecisionSource = void 0;
/**
 * Decision source types
 */
var DecisionSource;
(function (DecisionSource) {
    DecisionSource["SIGNAL"] = "SIGNAL";
    DecisionSource["STRATEGY"] = "STRATEGY";
    DecisionSource["BOT"] = "BOT";
    DecisionSource["USER"] = "USER";
    DecisionSource["SYSTEM"] = "SYSTEM";
    DecisionSource["RISK_MANAGEMENT"] = "RISK_MANAGEMENT";
    DecisionSource["ML_MODEL"] = "ML_MODEL";
    DecisionSource["CIRCUIT_BREAKER"] = "CIRCUIT_BREAKER";
})(DecisionSource || (exports.DecisionSource = DecisionSource = {}));
/**
 * Decision action types
 */
var DecisionActionType;
(function (DecisionActionType) {
    DecisionActionType["ENTRY"] = "ENTRY";
    DecisionActionType["EXIT"] = "EXIT";
    DecisionActionType["ADJUSTMENT"] = "ADJUSTMENT";
    DecisionActionType["CANCELLATION"] = "CANCELLATION";
    DecisionActionType["RISK_OVERRIDE"] = "RISK_OVERRIDE";
    DecisionActionType["LEVERAGE_CHANGE"] = "LEVERAGE_CHANGE";
    DecisionActionType["POSITION_CLOSE"] = "POSITION_CLOSE";
    DecisionActionType["STOP_LOSS_MOVE"] = "STOP_LOSS_MOVE";
    DecisionActionType["TAKE_PROFIT_MOVE"] = "TAKE_PROFIT_MOVE";
    DecisionActionType["ORDER_REJECTION"] = "ORDER_REJECTION";
    DecisionActionType["STRATEGY_ACTIVATION"] = "STRATEGY_ACTIVATION";
    DecisionActionType["STRATEGY_DEACTIVATION"] = "STRATEGY_DEACTIVATION";
    DecisionActionType["BOT_ACTIVATION"] = "BOT_ACTIVATION";
    DecisionActionType["BOT_DEACTIVATION"] = "BOT_DEACTIVATION";
    DecisionActionType["SIGNAL_GENERATION"] = "SIGNAL_GENERATION";
    DecisionActionType["SIGNAL_VALIDATION"] = "SIGNAL_VALIDATION";
})(DecisionActionType || (exports.DecisionActionType = DecisionActionType = {}));
/**
 * Decision outcome types
 */
var DecisionOutcome;
(function (DecisionOutcome) {
    DecisionOutcome["SUCCESS"] = "SUCCESS";
    DecisionOutcome["FAILURE"] = "FAILURE";
    DecisionOutcome["PARTIAL"] = "PARTIAL";
    DecisionOutcome["PENDING"] = "PENDING";
    DecisionOutcome["CANCELLED"] = "CANCELLED";
    DecisionOutcome["UNKNOWN"] = "UNKNOWN";
})(DecisionOutcome || (exports.DecisionOutcome = DecisionOutcome = {}));
/**
 * Decision importance levels
 */
var DecisionImportance;
(function (DecisionImportance) {
    DecisionImportance["HIGH"] = "HIGH";
    DecisionImportance["NORMAL"] = "NORMAL";
    DecisionImportance["LOW"] = "LOW";
})(DecisionImportance || (exports.DecisionImportance = DecisionImportance = {}));
/**
 * Audit trail types
 */
var AuditTrailType;
(function (AuditTrailType) {
    AuditTrailType["ORDER"] = "ORDER";
    AuditTrailType["SIGNAL"] = "SIGNAL";
    AuditTrailType["STRATEGY"] = "STRATEGY";
    AuditTrailType["BOT"] = "BOT";
    AuditTrailType["RISK"] = "RISK";
    AuditTrailType["AUTHENTICATION"] = "AUTHENTICATION";
    AuditTrailType["USER_MANAGEMENT"] = "USER_MANAGEMENT";
    AuditTrailType["SYSTEM"] = "SYSTEM";
    AuditTrailType["ML_MODEL"] = "ML_MODEL";
    AuditTrailType["PERFORMANCE"] = "PERFORMANCE";
})(AuditTrailType || (exports.AuditTrailType = AuditTrailType = {}));
/**
 * Audit trail status
 */
var AuditTrailStatus;
(function (AuditTrailStatus) {
    AuditTrailStatus["ACTIVE"] = "ACTIVE";
    AuditTrailStatus["COMPLETED"] = "COMPLETED";
    AuditTrailStatus["CANCELLED"] = "CANCELLED";
    AuditTrailStatus["FAILED"] = "FAILED";
})(AuditTrailStatus || (exports.AuditTrailStatus = AuditTrailStatus = {}));
/**
 * Audit event types
 */
var AuditEventType;
(function (AuditEventType) {
    AuditEventType["CREATE"] = "CREATE";
    AuditEventType["READ"] = "READ";
    AuditEventType["UPDATE"] = "UPDATE";
    AuditEventType["DELETE"] = "DELETE";
    AuditEventType["LOGIN"] = "LOGIN";
    AuditEventType["LOGOUT"] = "LOGOUT";
    AuditEventType["AUTHENTICATION"] = "AUTHENTICATION";
    AuditEventType["AUTHORIZATION"] = "AUTHORIZATION";
    AuditEventType["EXPORT"] = "EXPORT";
    AuditEventType["IMPORT"] = "IMPORT";
    AuditEventType["VALIDATION"] = "VALIDATION";
    AuditEventType["CALCULATION"] = "CALCULATION";
    AuditEventType["EXECUTION"] = "EXECUTION";
    AuditEventType["API_CALL"] = "API_CALL";
    AuditEventType["ERROR"] = "ERROR";
    AuditEventType["WARNING"] = "WARNING";
    AuditEventType["INFO"] = "INFO";
})(AuditEventType || (exports.AuditEventType = AuditEventType = {}));
/**
 * Audit event status
 */
var AuditEventStatus;
(function (AuditEventStatus) {
    AuditEventStatus["SUCCESS"] = "SUCCESS";
    AuditEventStatus["FAILURE"] = "FAILURE";
    AuditEventStatus["WARNING"] = "WARNING";
    AuditEventStatus["INFO"] = "INFO";
})(AuditEventStatus || (exports.AuditEventStatus = AuditEventStatus = {}));
//# sourceMappingURL=auditLog.js.map