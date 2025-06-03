/**
 * Timeframe Validation Utility
 * Ensures proper timeframe relationships and temporal consistency
 */
import { Timeframe } from '../services/multiTimeframeDataProvider';
import { EnhancedMarketData } from '../types/marketData';
export interface TimeframeRelationship {
    from: Timeframe;
    to: Timeframe;
    multiplier: number;
    description: string;
}
export interface ValidationResult {
    isValid: boolean;
    errors: string[];
    warnings: string[];
    relationships: TimeframeRelationship[];
    temporalConsistency: {
        score: number;
        issues: string[];
    };
}
export declare class TimeframeValidator {
    private static readonly TIMEFRAME_MINUTES;
    private static readonly EXPECTED_RELATIONSHIPS;
    /**
     * Validate timeframe relationships and data consistency
     */
    static validateTimeframes(data: {
        [key in Timeframe]?: EnhancedMarketData[];
    }, targetTimeframes: Timeframe[]): ValidationResult;
    /**
     * Validate timeframe mathematical relationships
     */
    private static validateRelationships;
    /**
     * Validate temporal consistency across timeframes
     */
    private static validateTemporalConsistency;
    /**
     * Validate data alignment between timeframes
     */
    private static validateDataAlignment;
    /**
     * Validate look-ahead bias
     */
    static validateLookAheadBias(data: {
        [key in Timeframe]?: EnhancedMarketData[];
    }, currentTimestamp: number): boolean;
    /**
     * Get timeframe priority for hierarchical decision making
     */
    static getTimeframePriority(timeframe: Timeframe): number;
    /**
     * Check if timeframe A can be aggregated to timeframe B
     */
    static canAggregate(from: Timeframe, to: Timeframe): boolean;
    /**
     * Get aggregation ratio between timeframes
     */
    static getAggregationRatio(from: Timeframe, to: Timeframe): number;
    /**
     * Log validation results
     */
    private static logValidationResults;
    /**
     * Generate timeframe relationship documentation
     */
    static generateDocumentation(): string;
}
export declare const timeframeValidator: typeof TimeframeValidator;
