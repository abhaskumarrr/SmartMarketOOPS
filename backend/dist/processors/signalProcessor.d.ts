/**
 * Trading Signal Event Processor
 * Processes trading signals and triggers order execution
 */
import { EventProcessor } from '../services/eventProcessingPipeline';
import { TradingEvent, TradingSignalEvent, EventProcessingResult } from '../types/events';
export interface SignalValidationRule {
    name: string;
    validate: (signal: TradingSignalEvent) => Promise<boolean>;
    reason?: string;
}
export declare class SignalProcessor implements EventProcessor {
    private name;
    private validationRules;
    private processedSignals;
    private signalStats;
    constructor();
    getName(): string;
    canProcess(event: TradingEvent): boolean;
    process(event: TradingEvent): Promise<EventProcessingResult>;
    /**
     * Process newly generated signal
     */
    private processGeneratedSignal;
    /**
     * Process validated signal (ready for execution)
     */
    private processValidatedSignal;
    /**
     * Process executed signal
     */
    private processExecutedSignal;
    /**
     * Process expired signal
     */
    private processExpiredSignal;
    /**
     * Process error signal
     */
    private processErrorSignal;
    /**
     * Initialize validation rules
     */
    private initializeValidationRules;
    /**
     * Validate signal against all rules
     */
    private validateSignal;
    /**
     * Add custom validation rule
     */
    addValidationRule(rule: SignalValidationRule): void;
    /**
     * Remove validation rule
     */
    removeValidationRule(name: string): boolean;
    /**
     * Get processor statistics
     */
    getStats(): {
        name: string;
        signalStats: typeof this.signalStats;
        validationRules: number;
        processedSignalsCount: number;
    };
    /**
     * Clear processed signals cache
     */
    clearProcessedSignals(): void;
}
