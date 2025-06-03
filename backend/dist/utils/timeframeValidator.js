"use strict";
/**
 * Timeframe Validation Utility
 * Ensures proper timeframe relationships and temporal consistency
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.timeframeValidator = exports.TimeframeValidator = void 0;
const logger_1 = require("./logger");
class TimeframeValidator {
    /**
     * Validate timeframe relationships and data consistency
     */
    static validateTimeframes(data, targetTimeframes) {
        const errors = [];
        const warnings = [];
        const validatedRelationships = [];
        logger_1.logger.info('🔍 Validating timeframe relationships...', { targetTimeframes });
        // Validate basic timeframe relationships
        const relationshipValidation = this.validateRelationships(targetTimeframes);
        errors.push(...relationshipValidation.errors);
        warnings.push(...relationshipValidation.warnings);
        validatedRelationships.push(...relationshipValidation.relationships);
        // Validate temporal consistency
        const temporalValidation = this.validateTemporalConsistency(data, targetTimeframes);
        errors.push(...temporalValidation.errors);
        warnings.push(...temporalValidation.warnings);
        // Validate data alignment
        const alignmentValidation = this.validateDataAlignment(data, targetTimeframes);
        errors.push(...alignmentValidation.errors);
        warnings.push(...alignmentValidation.warnings);
        const isValid = errors.length === 0;
        const temporalScore = Math.max(0, 100 - (errors.length * 20) - (warnings.length * 5));
        const result = {
            isValid,
            errors,
            warnings,
            relationships: validatedRelationships,
            temporalConsistency: {
                score: temporalScore,
                issues: [...errors, ...warnings],
            },
        };
        this.logValidationResults(result);
        return result;
    }
    /**
     * Validate timeframe mathematical relationships
     */
    static validateRelationships(timeframes) {
        const errors = [];
        const warnings = [];
        const relationships = [];
        // Check each expected relationship
        this.EXPECTED_RELATIONSHIPS.forEach(expected => {
            if (timeframes.includes(expected.from) && timeframes.includes(expected.to)) {
                const fromMinutes = this.TIMEFRAME_MINUTES[expected.from];
                const toMinutes = this.TIMEFRAME_MINUTES[expected.to];
                const actualMultiplier = toMinutes / fromMinutes;
                if (actualMultiplier === expected.multiplier) {
                    relationships.push(expected);
                }
                else {
                    errors.push(`Invalid relationship: ${expected.from} to ${expected.to}. ` +
                        `Expected multiplier: ${expected.multiplier}, Actual: ${actualMultiplier}`);
                }
            }
        });
        // Validate timeframe hierarchy
        const sortedTimeframes = timeframes.sort((a, b) => this.TIMEFRAME_MINUTES[a] - this.TIMEFRAME_MINUTES[b]);
        for (let i = 1; i < sortedTimeframes.length; i++) {
            const lower = sortedTimeframes[i - 1];
            const higher = sortedTimeframes[i];
            const lowerMinutes = this.TIMEFRAME_MINUTES[lower];
            const higherMinutes = this.TIMEFRAME_MINUTES[higher];
            if (higherMinutes % lowerMinutes !== 0) {
                warnings.push(`Timeframe ${higher} (${higherMinutes}m) is not a clean multiple of ${lower} (${lowerMinutes}m)`);
            }
        }
        return { errors, warnings, relationships };
    }
    /**
     * Validate temporal consistency across timeframes
     */
    static validateTemporalConsistency(data, timeframes) {
        const errors = [];
        const warnings = [];
        timeframes.forEach(timeframe => {
            const timeframeData = data[timeframe];
            if (!timeframeData || timeframeData.length === 0) {
                warnings.push(`No data available for timeframe ${timeframe}`);
                return;
            }
            // Check for chronological order
            for (let i = 1; i < timeframeData.length; i++) {
                if (timeframeData[i].timestamp <= timeframeData[i - 1].timestamp) {
                    errors.push(`Temporal inconsistency in ${timeframe}: timestamps not in chronological order`);
                    break;
                }
            }
            // Check for proper timeframe intervals
            const expectedInterval = this.TIMEFRAME_MINUTES[timeframe] * 60 * 1000; // Convert to milliseconds
            let intervalIssues = 0;
            for (let i = 1; i < Math.min(timeframeData.length, 100); i++) {
                const actualInterval = timeframeData[i].timestamp - timeframeData[i - 1].timestamp;
                const tolerance = expectedInterval * 0.1; // 10% tolerance
                if (Math.abs(actualInterval - expectedInterval) > tolerance) {
                    intervalIssues++;
                }
            }
            if (intervalIssues > timeframeData.length * 0.1) {
                warnings.push(`${timeframe} has ${intervalIssues} interval inconsistencies (${(intervalIssues / timeframeData.length * 100).toFixed(1)}%)`);
            }
        });
        return { errors, warnings };
    }
    /**
     * Validate data alignment between timeframes
     */
    static validateDataAlignment(data, timeframes) {
        const errors = [];
        const warnings = [];
        // Sort timeframes by duration
        const sortedTimeframes = timeframes.sort((a, b) => this.TIMEFRAME_MINUTES[a] - this.TIMEFRAME_MINUTES[b]);
        // Check alignment between consecutive timeframes
        for (let i = 1; i < sortedTimeframes.length; i++) {
            const lowerTF = sortedTimeframes[i - 1];
            const higherTF = sortedTimeframes[i];
            const lowerData = data[lowerTF];
            const higherData = data[higherTF];
            if (!lowerData || !higherData)
                continue;
            const lowerMinutes = this.TIMEFRAME_MINUTES[lowerTF];
            const higherMinutes = this.TIMEFRAME_MINUTES[higherTF];
            const expectedRatio = higherMinutes / lowerMinutes;
            // Check if higher timeframe data can be properly constructed from lower timeframe
            let alignmentIssues = 0;
            const sampleSize = Math.min(higherData.length, 50);
            for (let j = 0; j < sampleSize; j++) {
                const higherCandle = higherData[j];
                const higherPeriodStart = Math.floor(higherCandle.timestamp / (higherMinutes * 60 * 1000)) * (higherMinutes * 60 * 1000);
                // Find corresponding lower timeframe candles
                const correspondingLowerCandles = lowerData.filter(candle => {
                    const candlePeriodStart = Math.floor(candle.timestamp / (higherMinutes * 60 * 1000)) * (higherMinutes * 60 * 1000);
                    return candlePeriodStart === higherPeriodStart;
                });
                if (correspondingLowerCandles.length === 0) {
                    alignmentIssues++;
                }
                else if (Math.abs(correspondingLowerCandles.length - expectedRatio) > expectedRatio * 0.2) {
                    // Allow 20% tolerance for missing data
                    alignmentIssues++;
                }
            }
            if (alignmentIssues > sampleSize * 0.1) {
                warnings.push(`Alignment issues between ${lowerTF} and ${higherTF}: ${alignmentIssues}/${sampleSize} periods misaligned`);
            }
        }
        return { errors, warnings };
    }
    /**
     * Validate look-ahead bias
     */
    static validateLookAheadBias(data, currentTimestamp) {
        let biasDetected = false;
        Object.entries(data).forEach(([timeframe, timeframeData]) => {
            if (timeframeData) {
                timeframeData.forEach(candle => {
                    if (candle.timestamp > currentTimestamp) {
                        logger_1.logger.warn(`Look-ahead bias detected in ${timeframe}:`, {
                            candleTimestamp: new Date(candle.timestamp).toISOString(),
                            currentTimestamp: new Date(currentTimestamp).toISOString(),
                        });
                        biasDetected = true;
                    }
                });
            }
        });
        return biasDetected;
    }
    /**
     * Get timeframe priority for hierarchical decision making
     */
    static getTimeframePriority(timeframe) {
        const priorities = {
            '1m': 1,
            '3m': 2,
            '5m': 3,
            '15m': 4,
            '1h': 5,
            '4h': 6,
            '1d': 7,
        };
        return priorities[timeframe] || 0;
    }
    /**
     * Check if timeframe A can be aggregated to timeframe B
     */
    static canAggregate(from, to) {
        const fromMinutes = this.TIMEFRAME_MINUTES[from];
        const toMinutes = this.TIMEFRAME_MINUTES[to];
        return toMinutes > fromMinutes && toMinutes % fromMinutes === 0;
    }
    /**
     * Get aggregation ratio between timeframes
     */
    static getAggregationRatio(from, to) {
        if (!this.canAggregate(from, to)) {
            return 0;
        }
        return this.TIMEFRAME_MINUTES[to] / this.TIMEFRAME_MINUTES[from];
    }
    /**
     * Log validation results
     */
    static logValidationResults(result) {
        if (result.isValid) {
            logger_1.logger.info('✅ Timeframe validation passed', {
                relationships: result.relationships.length,
                temporalScore: result.temporalConsistency.score,
            });
        }
        else {
            logger_1.logger.error('❌ Timeframe validation failed', {
                errors: result.errors.length,
                warnings: result.warnings.length,
                temporalScore: result.temporalConsistency.score,
            });
        }
        if (result.errors.length > 0) {
            logger_1.logger.error('Validation errors:', result.errors);
        }
        if (result.warnings.length > 0) {
            logger_1.logger.warn('Validation warnings:', result.warnings);
        }
    }
    /**
     * Generate timeframe relationship documentation
     */
    static generateDocumentation() {
        let doc = '# Timeframe Relationships\n\n';
        doc += '## Supported Timeframes\n';
        Object.entries(this.TIMEFRAME_MINUTES).forEach(([tf, minutes]) => {
            doc += `- ${tf}: ${minutes} minutes\n`;
        });
        doc += '\n## Expected Relationships\n';
        this.EXPECTED_RELATIONSHIPS.forEach(rel => {
            doc += `- ${rel.from} → ${rel.to}: ${rel.multiplier}x (${rel.description})\n`;
        });
        doc += '\n## Validation Rules\n';
        doc += '1. All timeframes must be in chronological order\n';
        doc += '2. Higher timeframes must be clean multiples of lower timeframes\n';
        doc += '3. No look-ahead bias allowed\n';
        doc += '4. Data alignment must be consistent across timeframes\n';
        doc += '5. Temporal intervals must match expected timeframe durations\n';
        return doc;
    }
}
exports.TimeframeValidator = TimeframeValidator;
TimeframeValidator.TIMEFRAME_MINUTES = {
    '1m': 1,
    '3m': 3,
    '5m': 5,
    '15m': 15,
    '1h': 60,
    '4h': 240,
    '1d': 1440,
};
TimeframeValidator.EXPECTED_RELATIONSHIPS = [
    { from: '1m', to: '3m', multiplier: 3, description: '1 hour = 60 one-minute candles' },
    { from: '1m', to: '5m', multiplier: 5, description: '1 hour = 60 one-minute candles' },
    { from: '1m', to: '15m', multiplier: 15, description: '1 hour = 60 one-minute candles' },
    { from: '1m', to: '1h', multiplier: 60, description: '1 hour = 60 one-minute candles' },
    { from: '1m', to: '4h', multiplier: 240, description: '1 hour = 60 one-minute candles' },
    { from: '1m', to: '1d', multiplier: 1440, description: '1 hour = 60 one-minute candles' },
    { from: '3m', to: '15m', multiplier: 5, description: '15 minutes = 5 three-minute candles' },
    { from: '3m', to: '1h', multiplier: 20, description: '1 hour = 20 three-minute candles' },
    { from: '5m', to: '15m', multiplier: 3, description: '15 minutes = 3 five-minute candles' },
    { from: '5m', to: '1h', multiplier: 12, description: '1 hour = 12 five-minute candles' },
    { from: '15m', to: '1h', multiplier: 4, description: '1 hour = 4 fifteen-minute candles' },
    { from: '1h', to: '4h', multiplier: 4, description: '4 hours = 4 one-hour candles' },
    { from: '4h', to: '1d', multiplier: 6, description: '1 day = 6 four-hour candles' },
];
// Export utility functions
exports.timeframeValidator = TimeframeValidator;
//# sourceMappingURL=timeframeValidator.js.map