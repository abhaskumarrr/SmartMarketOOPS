/**
 * Analysis-Execution Bridge
 * Real-time coordination layer between analysis and execution engines
 * Built with FastAPI-style architecture, WebSocket support, and robust error handling
 */
export interface BridgeConfig {
    port: number;
    host: string;
    enableWebSocket: boolean;
    enableRateLimit: boolean;
    maxRequestsPerMinute: number;
    corsOrigins: string[];
    enableHelmet: boolean;
}
export interface TradingSignal {
    id: string;
    symbol: string;
    action: 'buy' | 'sell' | 'hold' | 'close';
    confidence: number;
    timestamp: number;
    source: 'ml_decision' | 'risk_management' | 'position_manager' | 'manual';
    metadata: Record<string, any>;
}
export interface ExecutionResult {
    signalId: string;
    success: boolean;
    executedAt: number;
    orderId?: string;
    executedPrice?: number;
    executedQuantity?: number;
    error?: string;
    latency: number;
}
export interface BridgeStatus {
    isRunning: boolean;
    connectedClients: number;
    totalSignals: number;
    successfulExecutions: number;
    failedExecutions: number;
    averageLatency: number;
    uptime: number;
    lastError?: string;
}
export interface WebSocketMessage {
    type: 'signal' | 'execution' | 'status' | 'error' | 'heartbeat';
    data: any;
    timestamp: number;
    id: string;
}
export declare class BridgeError extends Error {
    statusCode: number;
    code: string;
    constructor(message: string, statusCode?: number, code?: string);
}
export declare class ValidationError extends BridgeError {
    constructor(message: string);
}
export declare class ExecutionError extends BridgeError {
    constructor(message: string);
}
export declare class AnalysisExecutionBridge {
    private app;
    private server;
    private wss;
    private decisionEngine;
    private positionManager;
    private riskManager;
    private dataIntegration;
    private tradingBot;
    private isRunning;
    private connectedClients;
    private signalQueue;
    private executionResults;
    private startTime;
    private metrics;
    private config;
    constructor(customConfig?: Partial<BridgeConfig>);
    /**
     * Initialize the Analysis-Execution Bridge
     */
    initialize(): Promise<void>;
    /**
     * Start the bridge server
     */
    start(): Promise<void>;
    /**
     * Stop the bridge server
     */
    stop(): Promise<void>;
    /**
     * Get bridge status
     */
    getStatus(): BridgeStatus;
    /**
     * Send trading signal through the bridge
     */
    sendTradingSignal(signal: Omit<TradingSignal, 'id' | 'timestamp'>): Promise<string>;
    /**
     * Get execution result for a signal
     */
    getExecutionResult(signalId: string): ExecutionResult | null;
    /**
     * Cleanup resources
     */
    cleanup(): Promise<void>;
    /**
     * Setup Express middleware
     */
    private setupMiddleware;
    /**
     * Setup API routes
     */
    private setupRoutes;
    /**
     * Setup WebSocket handlers
     */
    private setupWebSocketHandlers;
    /**
     * Setup error handlers
     */
    private setupErrorHandlers;
    /**
     * Start signal processing loop
     */
    private startSignalProcessing;
    /**
     * Process individual trading signal
     */
    private processSignal;
    /**
     * Execute regular trading signal
     */
    private executeSignal;
    /**
     * Execute entry signal (buy/sell)
     */
    private executeEntry;
    /**
     * Execute exit signal
     */
    private executeExit;
    /**
     * Execute emergency close all positions
     */
    private executeEmergencyClose;
    /**
     * Handle WebSocket message
     */
    private handleWebSocketMessage;
    /**
     * Send message to specific WebSocket client
     */
    private sendToClient;
    /**
     * Broadcast message to all connected WebSocket clients
     */
    private broadcastToClients;
}
