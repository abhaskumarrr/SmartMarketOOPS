/**
 * Event Processing Pipeline
 * Core event-driven processing pipeline for trading system
 */
import { EventEmitter } from 'events';
import { TradingEvent, EventProcessingResult } from '../types/events';
export interface EventProcessor {
    process(event: TradingEvent): Promise<EventProcessingResult>;
    canProcess(event: TradingEvent): boolean;
    getName(): string;
}
export interface PipelineConfig {
    consumerName: string;
    batchSize: number;
    blockTime: number;
    retryAttempts: number;
    retryDelay: number;
    deadLetterThreshold: number;
    processingTimeout: number;
    enableCircuitBreaker: boolean;
    circuitBreakerThreshold: number;
    circuitBreakerTimeout: number;
}
export interface PipelineStats {
    processedEvents: number;
    failedEvents: number;
    retryEvents: number;
    deadLetterEvents: number;
    averageProcessingTime: number;
    throughput: number;
    circuitBreakerOpen: boolean;
    lastProcessedAt: number;
    uptime: number;
}
export declare class EventProcessingPipeline extends EventEmitter {
    private processors;
    private isRunning;
    private config;
    private stats;
    private startTime;
    private processingTimes;
    private circuitBreakerOpen;
    private circuitBreakerOpenTime;
    private consecutiveFailures;
    constructor(config?: Partial<PipelineConfig>);
    /**
     * Register an event processor
     */
    registerProcessor(processor: EventProcessor): void;
    /**
     * Start the event processing pipeline
     */
    start(): Promise<void>;
    /**
     * Stop the event processing pipeline
     */
    stop(): Promise<void>;
    /**
     * Process events from a specific stream
     */
    private processStream;
    /**
     * Process a single event
     */
    private processEvent;
    /**
     * Find appropriate processor for an event
     */
    private findProcessor;
    /**
     * Handle failed event processing
     */
    private handleFailedEvent;
    /**
     * Send event to dead letter queue
     */
    private sendToDeadLetterQueue;
    /**
     * Circuit breaker methods
     */
    private isCircuitBreakerOpen;
    private openCircuitBreaker;
    private closeCircuitBreaker;
    /**
     * Update processing time statistics
     */
    private updateProcessingTime;
    /**
     * Start stats update loop
     */
    private startStatsUpdate;
    /**
     * Utility methods
     */
    private sleep;
    private createTimeoutPromise;
    /**
     * Get pipeline statistics
     */
    getStats(): PipelineStats;
    /**
     * Get pipeline configuration
     */
    getConfig(): PipelineConfig;
    /**
     * Health check
     */
    healthCheck(): Promise<boolean>;
}
export declare const eventProcessingPipeline: EventProcessingPipeline;
