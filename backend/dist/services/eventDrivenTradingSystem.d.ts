/**
 * Event-Driven Trading System
 * Main orchestrator for event-driven trading architecture
 */
import { EventEmitter } from 'events';
import { MarketDataEvent, TradingSignalEvent, SystemEvent, BotEvent } from '../types/events';
export interface TradingSystemConfig {
    enableMarketDataProcessing: boolean;
    enableSignalProcessing: boolean;
    enableOrderProcessing: boolean;
    enableRiskManagement: boolean;
    enablePortfolioManagement: boolean;
    enableSystemMonitoring: boolean;
    marketDataSources: string[];
    signalGenerationModels: string[];
    riskManagementRules: string[];
}
export interface TradingSystemStats {
    uptime: number;
    eventsProcessed: number;
    eventsPerSecond: number;
    activeStreams: number;
    activeProcessors: number;
    systemHealth: 'HEALTHY' | 'DEGRADED' | 'UNHEALTHY';
    lastEventTime: number;
    errors: number;
}
export declare class EventDrivenTradingSystem extends EventEmitter {
    private static instance;
    private isRunning;
    private startTime;
    private config;
    private stats;
    private processors;
    private constructor();
    static getInstance(config?: Partial<TradingSystemConfig>): EventDrivenTradingSystem;
    /**
     * Initialize and start the event-driven trading system
     */
    start(): Promise<void>;
    /**
     * Stop the event-driven trading system
     */
    stop(): Promise<void>;
    /**
     * Register event processors
     */
    private registerProcessors;
    /**
     * Set up event listeners for monitoring
     */
    private setupEventListeners;
    /**
     * Start system monitoring
     */
    private startSystemMonitoring;
    /**
     * Perform system health check
     */
    private performHealthCheck;
    /**
     * Update system statistics
     */
    private updateSystemStats;
    /**
     * Publish market data event
     */
    publishMarketDataEvent(symbol: string, price: number, volume: number, exchange?: string, additionalData?: Partial<MarketDataEvent['data']>): Promise<string>;
    /**
     * Publish trading signal event
     */
    publishTradingSignalEvent(signalData: Partial<TradingSignalEvent['data']>, correlationId?: string): Promise<string>;
    /**
     * Publish system event
     */
    publishSystemEvent(type: SystemEvent['type'], data: Partial<SystemEvent['data']>): Promise<string>;
    /**
     * Publish bot event
     */
    publishBotEvent(type: BotEvent['type'], botData: Partial<BotEvent['data']>, userId?: string): Promise<string>;
    /**
     * Get system statistics
     */
    getStats(): TradingSystemStats;
    /**
     * Get system configuration
     */
    getConfig(): TradingSystemConfig;
    /**
     * Update system configuration
     */
    updateConfig(newConfig: Partial<TradingSystemConfig>): void;
    /**
     * Get processor statistics
     */
    getProcessorStats(): Record<string, any>;
    /**
     * Check if system is running
     */
    isSystemRunning(): boolean;
    /**
     * Get system health status
     */
    getHealthStatus(): 'HEALTHY' | 'DEGRADED' | 'UNHEALTHY';
}
export declare const eventDrivenTradingSystem: EventDrivenTradingSystem;
