#!/usr/bin/env node
/**
 * AI Position Manager Runner
 * Manages your Delta Exchange positions with AI-powered dynamic take profit
 */
declare class AIPositionManagerRunner {
    private deltaApi;
    private aiManager;
    private isRunning;
    constructor();
    /**
     * Start AI position management
     */
    start(): Promise<void>;
    /**
     * Test connection to Delta Exchange
     */
    private testConnection;
    /**
     * Keep the system running
     */
    private keepRunning;
    /**
     * Display current status
     */
    private displayStatus;
    /**
     * Stop AI position management
     */
    stop(): void;
}
export { AIPositionManagerRunner };
