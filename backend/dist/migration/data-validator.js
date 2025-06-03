"use strict";
/**
 * Data Validator for QuestDB Migration
 * Validates data integrity during PostgreSQL to QuestDB migration
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.DataValidator = void 0;
const logger_1 = require("../utils/logger");
class DataValidator {
    constructor() {
        this.stats = {
            totalRecords: 0,
            validRecords: 0,
            invalidRecords: 0,
            warningRecords: 0,
            validationErrors: {},
            validationWarnings: {},
        };
    }
    /**
     * Validate a metric record
     */
    validateMetric(metric) {
        const errors = [];
        const warnings = [];
        this.stats.totalRecords++;
        // Required field validation
        if (!metric.name || typeof metric.name !== 'string') {
            errors.push('Metric name is required and must be a string');
        }
        if (metric.value === undefined || metric.value === null || typeof metric.value !== 'number') {
            errors.push('Metric value is required and must be a number');
        }
        if (!metric.timestamp) {
            errors.push('Timestamp is required');
        }
        // Data type validation
        if (typeof metric.value === 'number') {
            if (!isFinite(metric.value)) {
                errors.push('Metric value must be a finite number');
            }
            if (metric.value < -1e15 || metric.value > 1e15) {
                warnings.push('Metric value is extremely large, may cause precision issues');
            }
        }
        // Timestamp validation
        if (metric.timestamp) {
            const timestamp = this.normalizeTimestamp(metric.timestamp);
            if (!timestamp || isNaN(timestamp.getTime())) {
                errors.push('Invalid timestamp format');
            }
            else {
                const now = new Date();
                const oneYearAgo = new Date(now.getFullYear() - 1, now.getMonth(), now.getDate());
                const oneYearFromNow = new Date(now.getFullYear() + 1, now.getMonth(), now.getDate());
                if (timestamp < oneYearAgo) {
                    warnings.push('Timestamp is more than 1 year old');
                }
                if (timestamp > oneYearFromNow) {
                    warnings.push('Timestamp is more than 1 year in the future');
                }
            }
        }
        // Tags validation
        if (metric.tags) {
            if (typeof metric.tags !== 'object') {
                errors.push('Tags must be an object');
            }
            else {
                for (const [key, value] of Object.entries(metric.tags)) {
                    if (typeof key !== 'string' || key.length === 0) {
                        errors.push('Tag keys must be non-empty strings');
                    }
                    if (typeof value !== 'string' && typeof value !== 'number') {
                        errors.push('Tag values must be strings or numbers');
                    }
                    if (typeof value === 'string' && value.length > 1000) {
                        warnings.push('Tag value is very long, may impact performance');
                    }
                }
            }
        }
        // Metric name validation
        if (metric.name) {
            if (metric.name.length > 255) {
                errors.push('Metric name is too long (max 255 characters)');
            }
            if (!/^[a-zA-Z0-9_.-]+$/.test(metric.name)) {
                warnings.push('Metric name contains special characters, may cause issues');
            }
        }
        this.updateStats(errors, warnings);
        return { valid: errors.length === 0, errors, warnings };
    }
    /**
     * Validate a trading signal record
     */
    validateTradingSignal(signal) {
        const errors = [];
        const warnings = [];
        this.stats.totalRecords++;
        // Required field validation
        const requiredFields = ['id', 'symbol', 'type', 'direction', 'strength', 'timeframe', 'source', 'price', 'confidenceScore', 'expectedReturn', 'expectedRisk', 'riskRewardRatio'];
        for (const field of requiredFields) {
            if (signal[field] === undefined || signal[field] === null) {
                errors.push(`${field} is required`);
            }
        }
        // Enum validation
        const validTypes = ['ENTRY', 'EXIT', 'INCREASE', 'DECREASE', 'HOLD'];
        if (signal.type && !validTypes.includes(signal.type)) {
            errors.push(`Invalid signal type: ${signal.type}`);
        }
        const validDirections = ['LONG', 'SHORT', 'NEUTRAL'];
        if (signal.direction && !validDirections.includes(signal.direction)) {
            errors.push(`Invalid signal direction: ${signal.direction}`);
        }
        const validStrengths = ['VERY_WEAK', 'WEAK', 'MODERATE', 'STRONG', 'VERY_STRONG'];
        if (signal.strength && !validStrengths.includes(signal.strength)) {
            errors.push(`Invalid signal strength: ${signal.strength}`);
        }
        // Numeric validation
        if (typeof signal.price === 'number') {
            if (signal.price <= 0) {
                errors.push('Price must be positive');
            }
            if (!isFinite(signal.price)) {
                errors.push('Price must be a finite number');
            }
        }
        if (typeof signal.confidenceScore === 'number') {
            if (signal.confidenceScore < 0 || signal.confidenceScore > 100) {
                errors.push('Confidence score must be between 0 and 100');
            }
        }
        if (typeof signal.expectedReturn === 'number' && !isFinite(signal.expectedReturn)) {
            errors.push('Expected return must be a finite number');
        }
        if (typeof signal.expectedRisk === 'number' && !isFinite(signal.expectedRisk)) {
            errors.push('Expected risk must be a finite number');
        }
        if (typeof signal.riskRewardRatio === 'number' && !isFinite(signal.riskRewardRatio)) {
            errors.push('Risk reward ratio must be a finite number');
        }
        // Optional field validation
        if (signal.targetPrice !== undefined && (typeof signal.targetPrice !== 'number' || signal.targetPrice <= 0)) {
            errors.push('Target price must be a positive number');
        }
        if (signal.stopLoss !== undefined && (typeof signal.stopLoss !== 'number' || signal.stopLoss <= 0)) {
            errors.push('Stop loss must be a positive number');
        }
        // Business logic validation
        if (signal.targetPrice && signal.price) {
            if (signal.direction === 'LONG' && signal.targetPrice <= signal.price) {
                warnings.push('Target price should be higher than entry price for LONG signals');
            }
            if (signal.direction === 'SHORT' && signal.targetPrice >= signal.price) {
                warnings.push('Target price should be lower than entry price for SHORT signals');
            }
        }
        if (signal.stopLoss && signal.price) {
            if (signal.direction === 'LONG' && signal.stopLoss >= signal.price) {
                warnings.push('Stop loss should be lower than entry price for LONG signals');
            }
            if (signal.direction === 'SHORT' && signal.stopLoss <= signal.price) {
                warnings.push('Stop loss should be higher than entry price for SHORT signals');
            }
        }
        // Timestamp validation
        if (!signal.timestamp) {
            errors.push('Timestamp is required');
        }
        else {
            const timestamp = this.normalizeTimestamp(signal.timestamp);
            if (!timestamp || isNaN(timestamp.getTime())) {
                errors.push('Invalid timestamp format');
            }
        }
        this.updateStats(errors, warnings);
        return { valid: errors.length === 0, errors, warnings };
    }
    /**
     * Validate an ML prediction record
     */
    validateMLPrediction(prediction) {
        const errors = [];
        const warnings = [];
        this.stats.totalRecords++;
        // Required field validation
        const requiredFields = ['id', 'modelId', 'symbol', 'timeframe', 'predictionType', 'values', 'confidenceScores'];
        for (const field of requiredFields) {
            if (prediction[field] === undefined || prediction[field] === null) {
                errors.push(`${field} is required`);
            }
        }
        // Enum validation
        const validPredictionTypes = ['PRICE', 'DIRECTION', 'PROBABILITY'];
        if (prediction.predictionType && !validPredictionTypes.includes(prediction.predictionType)) {
            errors.push(`Invalid prediction type: ${prediction.predictionType}`);
        }
        // JSON validation
        if (prediction.values) {
            try {
                const values = JSON.parse(prediction.values);
                if (!Array.isArray(values)) {
                    errors.push('Values must be a JSON array');
                }
                else {
                    if (values.length === 0) {
                        warnings.push('Values array is empty');
                    }
                    if (values.length > 1000) {
                        warnings.push('Values array is very large, may impact performance');
                    }
                    for (const value of values) {
                        if (typeof value !== 'number' || !isFinite(value)) {
                            errors.push('All values must be finite numbers');
                            break;
                        }
                    }
                }
            }
            catch (error) {
                errors.push('Values must be valid JSON');
            }
        }
        if (prediction.confidenceScores) {
            try {
                const scores = JSON.parse(prediction.confidenceScores);
                if (!Array.isArray(scores)) {
                    errors.push('Confidence scores must be a JSON array');
                }
                else {
                    for (const score of scores) {
                        if (typeof score !== 'number' || score < 0 || score > 1) {
                            errors.push('All confidence scores must be numbers between 0 and 1');
                            break;
                        }
                    }
                }
            }
            catch (error) {
                errors.push('Confidence scores must be valid JSON');
            }
        }
        // Timestamp validation
        if (!prediction.timestamp) {
            errors.push('Timestamp is required');
        }
        this.updateStats(errors, warnings);
        return { valid: errors.length === 0, errors, warnings };
    }
    /**
     * Validate a performance metric record
     */
    validatePerformanceMetric(metric) {
        const errors = [];
        const warnings = [];
        this.stats.totalRecords++;
        // Required field validation
        const requiredFields = ['system', 'component', 'metric', 'unit', 'value'];
        for (const field of requiredFields) {
            if (metric[field] === undefined || metric[field] === null) {
                errors.push(`${field} is required`);
            }
        }
        // Enum validation
        const validSystems = ['API', 'ML', 'TRADING', 'DATABASE', 'FRONTEND', 'WEBSOCKET'];
        if (metric.system && !validSystems.includes(metric.system)) {
            errors.push(`Invalid system: ${metric.system}`);
        }
        // Numeric validation
        if (typeof metric.value === 'number') {
            if (!isFinite(metric.value)) {
                errors.push('Metric value must be a finite number');
            }
            if (metric.value < 0 && !['error_rate', 'latency', 'response_time'].includes(metric.metric)) {
                warnings.push('Negative metric value may be unexpected');
            }
        }
        // Timestamp validation
        if (!metric.timestamp) {
            errors.push('Timestamp is required');
        }
        this.updateStats(errors, warnings);
        return { valid: errors.length === 0, errors, warnings };
    }
    /**
     * Get validation statistics
     */
    getStats() {
        return { ...this.stats };
    }
    /**
     * Reset validation statistics
     */
    resetStats() {
        this.stats = {
            totalRecords: 0,
            validRecords: 0,
            invalidRecords: 0,
            warningRecords: 0,
            validationErrors: {},
            validationWarnings: {},
        };
    }
    /**
     * Log validation summary
     */
    logValidationSummary() {
        logger_1.logger.info('📊 Validation Summary:', {
            totalRecords: this.stats.totalRecords,
            validRecords: this.stats.validRecords,
            invalidRecords: this.stats.invalidRecords,
            warningRecords: this.stats.warningRecords,
            validationRate: this.stats.totalRecords > 0 ? (this.stats.validRecords / this.stats.totalRecords * 100).toFixed(2) + '%' : '0%',
        });
        if (Object.keys(this.stats.validationErrors).length > 0) {
            logger_1.logger.warn('🚨 Top validation errors:', this.stats.validationErrors);
        }
        if (Object.keys(this.stats.validationWarnings).length > 0) {
            logger_1.logger.warn('⚠️ Top validation warnings:', this.stats.validationWarnings);
        }
    }
    // Private helper methods
    normalizeTimestamp(timestamp) {
        try {
            if (timestamp instanceof Date) {
                return timestamp;
            }
            if (typeof timestamp === 'string') {
                return new Date(timestamp);
            }
            if (typeof timestamp === 'number') {
                // Handle both milliseconds and seconds
                const date = timestamp > 1e10 ? new Date(timestamp) : new Date(timestamp * 1000);
                return date;
            }
            return null;
        }
        catch (error) {
            return null;
        }
    }
    updateStats(errors, warnings) {
        if (errors.length === 0) {
            this.stats.validRecords++;
        }
        else {
            this.stats.invalidRecords++;
            // Count error types
            for (const error of errors) {
                this.stats.validationErrors[error] = (this.stats.validationErrors[error] || 0) + 1;
            }
        }
        if (warnings.length > 0) {
            this.stats.warningRecords++;
            // Count warning types
            for (const warning of warnings) {
                this.stats.validationWarnings[warning] = (this.stats.validationWarnings[warning] || 0) + 1;
            }
        }
    }
}
exports.DataValidator = DataValidator;
//# sourceMappingURL=data-validator.js.map