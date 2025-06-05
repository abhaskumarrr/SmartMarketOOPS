/**
 * Delta Exchange Bot Manager
 * Manages multiple trading bots for Delta Exchange India testnet
 */
import { EventEmitter } from 'events';
import { BotConfig, BotStatus } from './DeltaTradingBot';
export interface BotManagerStatus {
    totalBots: number;
    runningBots: number;
    pausedBots: number;
    stoppedBots: number;
    errorBots: number;
    totalTrades: number;
    totalPnL: number;
}
export declare class DeltaBotManager extends EventEmitter {
    private deltaService;
    private bots;
    private isInitialized;
    constructor();
    /**
     * Initialize the bot manager with Delta Exchange credentials
     */
    initialize(): Promise<void>;
    /**
     * Wait for Delta Exchange service to be ready
     */
    private waitForDeltaService;
    /**
     * Create a new trading bot
     */
    createBot(config: BotConfig): Promise<string>;
    /**
     * Start a trading bot
     */
    startBot(botId: string): Promise<void>;
    /**
     * Stop a trading bot
     */
    stopBot(botId: string): Promise<void>;
    /**
     * Pause a trading bot
     */
    pauseBot(botId: string): void;
    /**
     * Resume a trading bot
     */
    resumeBot(botId: string): void;
    /**
     * Remove a trading bot
     */
    removeBot(botId: string): Promise<void>;
    /**
     * Get bot status
     */
    getBotStatus(botId: string): BotStatus;
    /**
     * Get all bot statuses
     */
    getAllBotStatuses(): BotStatus[];
    /**
     * Get bot manager status
     */
    getManagerStatus(): BotManagerStatus;
    /**
     * Update bot configuration
     */
    updateBotConfig(botId: string, newConfig: Partial<BotConfig>): void;
    /**
     * Emergency stop all bots
     */
    emergencyStopAll(): Promise<void>;
    /**
     * Get bot performance metrics
     */
    getBotPerformance(botId: string): any;
    /**
     * Get all bots performance summary
     */
    getAllBotsPerformance(): any;
    /**
     * Validate bot configuration
     */
    private validateBotConfig;
    /**
     * Set up event listeners for a bot
     */
    private setupBotEventListeners;
    /**
     * Cleanup all resources
     */
    cleanup(): Promise<void>;
}
