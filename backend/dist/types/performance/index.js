"use strict";
/**
 * Performance Testing Types
 * Interfaces for performance testing and optimization
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.ABTestStatus = exports.ABTestType = exports.OptimizationImpact = exports.OptimizationCategory = exports.TestStatus = exports.PerformanceTestType = void 0;
/**
 * Performance Test Type
 */
var PerformanceTestType;
(function (PerformanceTestType) {
    PerformanceTestType["API_LATENCY"] = "API_LATENCY";
    PerformanceTestType["ML_PREDICTION_THROUGHPUT"] = "ML_PREDICTION_THROUGHPUT";
    PerformanceTestType["SIGNAL_GENERATION"] = "SIGNAL_GENERATION";
    PerformanceTestType["STRATEGY_EXECUTION"] = "STRATEGY_EXECUTION";
    PerformanceTestType["END_TO_END"] = "END_TO_END";
    PerformanceTestType["LOAD_TEST"] = "LOAD_TEST";
    PerformanceTestType["STRESS_TEST"] = "STRESS_TEST";
})(PerformanceTestType || (exports.PerformanceTestType = PerformanceTestType = {}));
/**
 * Test Status
 */
var TestStatus;
(function (TestStatus) {
    TestStatus["RUNNING"] = "RUNNING";
    TestStatus["COMPLETED"] = "COMPLETED";
    TestStatus["FAILED"] = "FAILED";
    TestStatus["CANCELLED"] = "CANCELLED";
})(TestStatus || (exports.TestStatus = TestStatus = {}));
/**
 * Optimization Category
 */
var OptimizationCategory;
(function (OptimizationCategory) {
    OptimizationCategory["CACHING"] = "CACHING";
    OptimizationCategory["DATABASE"] = "DATABASE";
    OptimizationCategory["ML_MODEL"] = "ML_MODEL";
    OptimizationCategory["API_ENDPOINT"] = "API_ENDPOINT";
    OptimizationCategory["CONCURRENCY"] = "CONCURRENCY";
    OptimizationCategory["MEMORY_USAGE"] = "MEMORY_USAGE";
    OptimizationCategory["CODE_OPTIMIZATION"] = "CODE_OPTIMIZATION";
    OptimizationCategory["CONFIGURATION"] = "CONFIGURATION";
})(OptimizationCategory || (exports.OptimizationCategory = OptimizationCategory = {}));
/**
 * Optimization Impact
 */
var OptimizationImpact;
(function (OptimizationImpact) {
    OptimizationImpact["LOW"] = "LOW";
    OptimizationImpact["MEDIUM"] = "MEDIUM";
    OptimizationImpact["HIGH"] = "HIGH";
    OptimizationImpact["CRITICAL"] = "CRITICAL";
})(OptimizationImpact || (exports.OptimizationImpact = OptimizationImpact = {}));
/**
 * A/B Test Type
 */
var ABTestType;
(function (ABTestType) {
    ABTestType["ML_MODEL"] = "ML_MODEL";
    ABTestType["STRATEGY"] = "STRATEGY";
    ABTestType["SIGNAL_GENERATION"] = "SIGNAL_GENERATION";
    ABTestType["API_CONFIGURATION"] = "API_CONFIGURATION";
})(ABTestType || (exports.ABTestType = ABTestType = {}));
/**
 * A/B Test Status
 */
var ABTestStatus;
(function (ABTestStatus) {
    ABTestStatus["DRAFT"] = "DRAFT";
    ABTestStatus["RUNNING"] = "RUNNING";
    ABTestStatus["COMPLETED"] = "COMPLETED";
    ABTestStatus["CANCELLED"] = "CANCELLED";
})(ABTestStatus || (exports.ABTestStatus = ABTestStatus = {}));
//# sourceMappingURL=index.js.map