#!/usr/bin/env node
/**
 * Simulate Position Management
 * Demonstrates AI position management with simulated Delta position
 */
declare class PositionSimulator {
    private aiManager;
    private simulatedPosition;
    private currentPrice;
    private priceDirection;
    private iteration;
    constructor();
    /**
     * Create mock Delta API for simulation
     */
    private createMockDeltaApi;
    /**
     * Run position management simulation
     */
    runSimulation(): Promise<void>;
    /**
     * Simulate realistic market movement
     */
    private simulateMarketMovement;
    /**
     * Update position P&L
     */
    private updatePositionPnL;
    /**
     * Log current state
     */
    private logCurrentState;
    /**
     * Display final results
     */
    private displayFinalResults;
    /**
     * Sleep utility
     */
    private sleep;
}
export { PositionSimulator };
