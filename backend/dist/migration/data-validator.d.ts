/**
 * Data Validator for QuestDB Migration
 * Validates data integrity during PostgreSQL to QuestDB migration
 */
import { MetricData, TradingSignalData, MLPredictionData, PerformanceMetricData } from '../services/questdbService';
export interface ValidationResult {
    valid: boolean;
    errors: string[];
    warnings: string[];
}
export interface ValidationStats {
    totalRecords: number;
    validRecords: number;
    invalidRecords: number;
    warningRecords: number;
    validationErrors: Record<string, number>;
    validationWarnings: Record<string, number>;
}
export declare class DataValidator {
    private stats;
    /**
     * Validate a metric record
     */
    validateMetric(metric: MetricData): ValidationResult;
    /**
     * Validate a trading signal record
     */
    validateTradingSignal(signal: TradingSignalData): ValidationResult;
    /**
     * Validate an ML prediction record
     */
    validateMLPrediction(prediction: MLPredictionData): ValidationResult;
    /**
     * Validate a performance metric record
     */
    validatePerformanceMetric(metric: PerformanceMetricData): ValidationResult;
    /**
     * Get validation statistics
     */
    getStats(): ValidationStats;
    /**
     * Reset validation statistics
     */
    resetStats(): void;
    /**
     * Log validation summary
     */
    logValidationSummary(): void;
    private normalizeTimestamp;
    private updateStats;
}
