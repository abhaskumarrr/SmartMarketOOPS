"use strict";
/**
 * Analysis-Execution Bridge
 * Real-time coordination layer between analysis and execution engines
 * Built with FastAPI-style architecture, WebSocket support, and robust error handling
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.AnalysisExecutionBridge = exports.ExecutionError = exports.ValidationError = exports.BridgeError = void 0;
const express_1 = __importDefault(require("express"));
const http_1 = require("http");
const ws_1 = require("ws");
const cors_1 = __importDefault(require("cors"));
const helmet_1 = __importDefault(require("helmet"));
const express_rate_limit_1 = __importDefault(require("express-rate-limit"));
const EnhancedTradingDecisionEngine_1 = require("./EnhancedTradingDecisionEngine");
const MLPositionManager_1 = require("./MLPositionManager");
const EnhancedRiskManagementSystem_1 = require("./EnhancedRiskManagementSystem");
const DataCollectorIntegration_1 = require("./DataCollectorIntegration");
const DeltaTradingBot_1 = require("./DeltaTradingBot");
const logger_1 = require("../utils/logger");
// Custom error classes
class BridgeError extends Error {
    constructor(message, statusCode = 500, code = 'BRIDGE_ERROR') {
        super(message);
        this.statusCode = statusCode;
        this.code = code;
        this.name = 'BridgeError';
    }
}
exports.BridgeError = BridgeError;
class ValidationError extends BridgeError {
    constructor(message) {
        super(message, 400, 'VALIDATION_ERROR');
        this.name = 'ValidationError';
    }
}
exports.ValidationError = ValidationError;
class ExecutionError extends BridgeError {
    constructor(message) {
        super(message, 500, 'EXECUTION_ERROR');
        this.name = 'ExecutionError';
    }
}
exports.ExecutionError = ExecutionError;
class AnalysisExecutionBridge {
    constructor(customConfig) {
        // Bridge state
        this.isRunning = false;
        this.connectedClients = new Set();
        this.signalQueue = [];
        this.executionResults = new Map();
        this.startTime = 0;
        // Performance metrics
        this.metrics = {
            totalSignals: 0,
            successfulExecutions: 0,
            failedExecutions: 0,
            totalLatency: 0,
            executionCount: 0
        };
        // Configuration optimized for high-frequency trading
        this.config = {
            port: 8000,
            host: '0.0.0.0',
            enableWebSocket: true,
            enableRateLimit: true,
            maxRequestsPerMinute: 1000, // High rate limit for trading
            corsOrigins: ['http://localhost:3000', 'http://localhost:3001'],
            enableHelmet: true
        };
        if (customConfig) {
            this.config = { ...this.config, ...customConfig };
        }
        // Initialize Express app
        this.app = (0, express_1.default)();
        this.server = (0, http_1.createServer)(this.app);
        // Initialize WebSocket server
        this.wss = new ws_1.WebSocketServer({ server: this.server });
        // Initialize trading components
        this.decisionEngine = new EnhancedTradingDecisionEngine_1.EnhancedTradingDecisionEngine();
        this.positionManager = new MLPositionManager_1.MLPositionManager();
        this.riskManager = new EnhancedRiskManagementSystem_1.EnhancedRiskManagementSystem();
        this.dataIntegration = new DataCollectorIntegration_1.DataCollectorIntegration();
        this.tradingBot = new DeltaTradingBot_1.DeltaTradingBot();
    }
    /**
     * Initialize the Analysis-Execution Bridge
     */
    async initialize() {
        try {
            logger_1.logger.info('🌉 Initializing Analysis-Execution Bridge...');
            // Initialize trading components
            await this.decisionEngine.initialize();
            await this.positionManager.initialize();
            await this.riskManager.initialize();
            await this.dataIntegration.initialize();
            // Setup Express middleware
            this.setupMiddleware();
            // Setup API routes
            this.setupRoutes();
            // Setup WebSocket handlers
            this.setupWebSocketHandlers();
            // Setup error handlers
            this.setupErrorHandlers();
            logger_1.logger.info('✅ Analysis-Execution Bridge initialized successfully');
            logger_1.logger.info(`🔧 Configuration: Port ${this.config.port}, WebSocket ${this.config.enableWebSocket ? 'ON' : 'OFF'}, Rate Limit ${this.config.maxRequestsPerMinute}/min`);
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to initialize Analysis-Execution Bridge:', error.message);
            throw error;
        }
    }
    /**
     * Start the bridge server
     */
    async start() {
        try {
            if (this.isRunning) {
                logger_1.logger.warn('⚠️ Bridge is already running');
                return;
            }
            this.startTime = Date.now();
            // Start the server
            await new Promise((resolve, reject) => {
                this.server.listen(this.config.port, this.config.host, () => {
                    this.isRunning = true;
                    logger_1.logger.info(`🚀 Analysis-Execution Bridge started on ${this.config.host}:${this.config.port}`);
                    resolve();
                });
                this.server.on('error', (error) => {
                    logger_1.logger.error('❌ Server error:', error);
                    reject(error);
                });
            });
            // Start signal processing
            this.startSignalProcessing();
            logger_1.logger.info('🌉 Analysis-Execution Bridge is now operational');
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to start Analysis-Execution Bridge:', error.message);
            throw error;
        }
    }
    /**
     * Stop the bridge server
     */
    async stop() {
        try {
            if (!this.isRunning) {
                logger_1.logger.warn('⚠️ Bridge is not running');
                return;
            }
            logger_1.logger.info('🛑 Stopping Analysis-Execution Bridge...');
            // Close WebSocket connections
            this.connectedClients.forEach(client => {
                if (client.readyState === ws_1.WebSocket.OPEN) {
                    client.close(1000, 'Server shutdown');
                }
            });
            this.connectedClients.clear();
            // Close WebSocket server
            this.wss.close();
            // Close HTTP server
            await new Promise((resolve) => {
                this.server.close(() => {
                    this.isRunning = false;
                    logger_1.logger.info('✅ Analysis-Execution Bridge stopped');
                    resolve();
                });
            });
        }
        catch (error) {
            logger_1.logger.error('❌ Error stopping Analysis-Execution Bridge:', error.message);
            throw error;
        }
    }
    /**
     * Get bridge status
     */
    getStatus() {
        const uptime = this.isRunning ? Date.now() - this.startTime : 0;
        const averageLatency = this.metrics.executionCount > 0 ?
            this.metrics.totalLatency / this.metrics.executionCount : 0;
        return {
            isRunning: this.isRunning,
            connectedClients: this.connectedClients.size,
            totalSignals: this.metrics.totalSignals,
            successfulExecutions: this.metrics.successfulExecutions,
            failedExecutions: this.metrics.failedExecutions,
            averageLatency,
            uptime
        };
    }
    /**
     * Send trading signal through the bridge
     */
    async sendTradingSignal(signal) {
        try {
            const tradingSignal = {
                ...signal,
                id: `signal_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                timestamp: Date.now()
            };
            // Add to signal queue
            this.signalQueue.push(tradingSignal);
            this.metrics.totalSignals++;
            // Broadcast to WebSocket clients
            this.broadcastToClients({
                type: 'signal',
                data: tradingSignal,
                timestamp: Date.now(),
                id: tradingSignal.id
            });
            logger_1.logger.info(`📡 Trading signal sent: ${tradingSignal.action.toUpperCase()} ${tradingSignal.symbol} (${(tradingSignal.confidence * 100).toFixed(1)}%)`);
            return tradingSignal.id;
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to send trading signal:', error.message);
            throw new BridgeError(`Failed to send trading signal: ${error.message}`);
        }
    }
    /**
     * Get execution result for a signal
     */
    getExecutionResult(signalId) {
        return this.executionResults.get(signalId) || null;
    }
    /**
     * Cleanup resources
     */
    async cleanup() {
        try {
            logger_1.logger.info('🧹 Cleaning up Analysis-Execution Bridge...');
            await this.stop();
            // Cleanup trading components
            await this.positionManager.cleanup();
            await this.riskManager.cleanup();
            await this.dataIntegration.cleanup();
            logger_1.logger.info('✅ Analysis-Execution Bridge cleanup completed');
        }
        catch (error) {
            logger_1.logger.error('❌ Error during Analysis-Execution Bridge cleanup:', error.message);
        }
    }
    // Private methods for bridge implementation
    /**
     * Setup Express middleware
     */
    setupMiddleware() {
        // Security middleware
        if (this.config.enableHelmet) {
            this.app.use((0, helmet_1.default)({
                contentSecurityPolicy: false, // Disable for WebSocket support
                crossOriginEmbedderPolicy: false
            }));
        }
        // CORS middleware
        this.app.use((0, cors_1.default)({
            origin: this.config.corsOrigins,
            credentials: true,
            methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
            allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
        }));
        // Rate limiting
        if (this.config.enableRateLimit) {
            const limiter = (0, express_rate_limit_1.default)({
                windowMs: 60 * 1000, // 1 minute
                max: this.config.maxRequestsPerMinute,
                message: {
                    error: 'Too many requests',
                    retryAfter: 60
                },
                standardHeaders: true,
                legacyHeaders: false
            });
            this.app.use(limiter);
        }
        // Body parsing middleware
        this.app.use(express_1.default.json({ limit: '10mb' }));
        this.app.use(express_1.default.urlencoded({ extended: true, limit: '10mb' }));
        // Request logging middleware
        this.app.use((req, res, next) => {
            const start = Date.now();
            res.on('finish', () => {
                const duration = Date.now() - start;
                logger_1.logger.debug(`${req.method} ${req.path} - ${res.statusCode} (${duration}ms)`);
            });
            next();
        });
    }
    /**
     * Setup API routes
     */
    setupRoutes() {
        // Health check endpoint
        this.app.get('/health', (req, res) => {
            const status = this.getStatus();
            res.json({
                status: 'healthy',
                timestamp: Date.now(),
                bridge: status
            });
        });
        // Get bridge status
        this.app.get('/api/status', (req, res) => {
            try {
                const status = this.getStatus();
                res.json(status);
            }
            catch (error) {
                res.status(500).json({ error: error.message });
            }
        });
        // Generate trading decision
        this.app.post('/api/decisions/:symbol', async (req, res, next) => {
            try {
                const { symbol } = req.params;
                if (!symbol) {
                    throw new ValidationError('Symbol is required');
                }
                const decision = await this.decisionEngine.generateTradingDecision(symbol);
                if (!decision) {
                    res.status(204).json({ message: 'No trading decision generated' });
                    return;
                }
                // Send signal through bridge
                const signalId = await this.sendTradingSignal({
                    symbol: decision.symbol,
                    action: decision.action,
                    confidence: decision.confidence,
                    source: 'ml_decision',
                    metadata: {
                        positionSize: decision.positionSize,
                        leverage: decision.leverage,
                        stopLoss: decision.stopLoss,
                        takeProfit: decision.takeProfit,
                        riskScore: decision.riskScore
                    }
                });
                res.json({
                    decision,
                    signalId,
                    timestamp: Date.now()
                });
            }
            catch (error) {
                next(error);
            }
        });
        // Get positions
        this.app.get('/api/positions', async (req, res, next) => {
            try {
                const positions = this.positionManager.getActivePositions();
                const metrics = this.positionManager.getPerformanceMetrics();
                res.json({
                    positions,
                    metrics,
                    timestamp: Date.now()
                });
            }
            catch (error) {
                next(error);
            }
        });
        // Get risk metrics
        this.app.get('/api/risk', async (req, res, next) => {
            try {
                const riskMetrics = this.riskManager.getRiskMetrics();
                const failsafeMechanisms = this.riskManager.getFailsafeMechanisms();
                res.json({
                    metrics: riskMetrics,
                    failsafeMechanisms,
                    timestamp: Date.now()
                });
            }
            catch (error) {
                next(error);
            }
        });
        // Manual trading signal
        this.app.post('/api/signals', async (req, res, next) => {
            try {
                const { symbol, action, confidence, metadata } = req.body;
                if (!symbol || !action) {
                    throw new ValidationError('Symbol and action are required');
                }
                if (confidence !== undefined && (confidence < 0 || confidence > 1)) {
                    throw new ValidationError('Confidence must be between 0 and 1');
                }
                const signalId = await this.sendTradingSignal({
                    symbol,
                    action,
                    confidence: confidence || 0.5,
                    source: 'manual',
                    metadata: metadata || {}
                });
                res.json({
                    signalId,
                    timestamp: Date.now()
                });
            }
            catch (error) {
                next(error);
            }
        });
        // Get execution result
        this.app.get('/api/executions/:signalId', (req, res, next) => {
            try {
                const { signalId } = req.params;
                const result = this.getExecutionResult(signalId);
                if (!result) {
                    res.status(404).json({ error: 'Execution result not found' });
                    return;
                }
                res.json(result);
            }
            catch (error) {
                next(error);
            }
        });
        // Emergency stop
        this.app.post('/api/emergency-stop', async (req, res, next) => {
            try {
                logger_1.logger.warn('🚨 Emergency stop triggered via API');
                // Trigger emergency stop in risk management
                await this.riskManager.checkCircuitBreakers();
                // Send emergency signal
                const signalId = await this.sendTradingSignal({
                    symbol: 'ALL',
                    action: 'close',
                    confidence: 1.0,
                    source: 'manual',
                    metadata: { emergency: true }
                });
                res.json({
                    message: 'Emergency stop triggered',
                    signalId,
                    timestamp: Date.now()
                });
            }
            catch (error) {
                next(error);
            }
        });
    }
    /**
     * Setup WebSocket handlers
     */
    setupWebSocketHandlers() {
        if (!this.config.enableWebSocket) {
            return;
        }
        this.wss.on('connection', (ws, req) => {
            const clientId = `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
            logger_1.logger.info(`🔌 WebSocket client connected: ${clientId}`);
            this.connectedClients.add(ws);
            // Send welcome message
            this.sendToClient(ws, {
                type: 'status',
                data: {
                    message: 'Connected to Analysis-Execution Bridge',
                    clientId,
                    status: this.getStatus()
                },
                timestamp: Date.now(),
                id: `welcome_${clientId}`
            });
            // Handle incoming messages
            ws.on('message', async (data) => {
                try {
                    const message = JSON.parse(data.toString());
                    await this.handleWebSocketMessage(ws, message, clientId);
                }
                catch (error) {
                    logger_1.logger.error(`❌ WebSocket message error from ${clientId}:`, error.message);
                    this.sendToClient(ws, {
                        type: 'error',
                        data: { error: 'Invalid message format' },
                        timestamp: Date.now(),
                        id: `error_${Date.now()}`
                    });
                }
            });
            // Handle client disconnect
            ws.on('close', (code, reason) => {
                logger_1.logger.info(`🔌 WebSocket client disconnected: ${clientId} (${code})`);
                this.connectedClients.delete(ws);
            });
            // Handle WebSocket errors
            ws.on('error', (error) => {
                logger_1.logger.error(`❌ WebSocket error from ${clientId}:`, error.message);
                this.connectedClients.delete(ws);
            });
            // Send periodic heartbeat
            const heartbeatInterval = setInterval(() => {
                if (ws.readyState === ws_1.WebSocket.OPEN) {
                    this.sendToClient(ws, {
                        type: 'heartbeat',
                        data: { timestamp: Date.now() },
                        timestamp: Date.now(),
                        id: `heartbeat_${Date.now()}`
                    });
                }
                else {
                    clearInterval(heartbeatInterval);
                }
            }, 30000); // 30 seconds
        });
        logger_1.logger.info('🔌 WebSocket server configured');
    }
    /**
     * Setup error handlers
     */
    setupErrorHandlers() {
        // Custom error handler
        this.app.use((error, req, res, next) => {
            logger_1.logger.error(`❌ API Error: ${error.message}`, error.stack);
            if (error instanceof ValidationError) {
                res.status(error.statusCode).json({
                    error: error.message,
                    code: error.code,
                    timestamp: Date.now()
                });
            }
            else if (error instanceof ExecutionError) {
                res.status(error.statusCode).json({
                    error: error.message,
                    code: error.code,
                    timestamp: Date.now()
                });
            }
            else if (error instanceof BridgeError) {
                res.status(error.statusCode).json({
                    error: error.message,
                    code: error.code,
                    timestamp: Date.now()
                });
            }
            else {
                // Generic error
                res.status(500).json({
                    error: 'Internal server error',
                    code: 'INTERNAL_ERROR',
                    timestamp: Date.now()
                });
            }
        });
        // 404 handler
        this.app.use((req, res) => {
            res.status(404).json({
                error: 'Endpoint not found',
                code: 'NOT_FOUND',
                path: req.path,
                timestamp: Date.now()
            });
        });
        // Unhandled promise rejection handler
        process.on('unhandledRejection', (reason, promise) => {
            logger_1.logger.error('❌ Unhandled promise rejection:', reason);
        });
        // Uncaught exception handler
        process.on('uncaughtException', (error) => {
            logger_1.logger.error('❌ Uncaught exception:', error);
            // Graceful shutdown
            this.stop().catch(() => {
                process.exit(1);
            });
        });
    }
    /**
     * Start signal processing loop
     */
    startSignalProcessing() {
        const processSignals = async () => {
            try {
                if (this.signalQueue.length > 0) {
                    const signal = this.signalQueue.shift();
                    if (signal) {
                        await this.processSignal(signal);
                    }
                }
            }
            catch (error) {
                logger_1.logger.error('❌ Signal processing error:', error.message);
            }
        };
        // Process signals every 100ms for low latency
        setInterval(processSignals, 100);
        logger_1.logger.info('⚡ Signal processing started (100ms intervals)');
    }
    /**
     * Process individual trading signal
     */
    async processSignal(signal) {
        const startTime = Date.now();
        try {
            logger_1.logger.debug(`🔄 Processing signal: ${signal.id}`);
            let executionResult;
            if (signal.action === 'close' && signal.symbol === 'ALL') {
                // Emergency close all positions
                executionResult = await this.executeEmergencyClose(signal);
            }
            else {
                // Regular signal processing
                executionResult = await this.executeSignal(signal);
            }
            // Store execution result
            this.executionResults.set(signal.id, executionResult);
            // Update metrics
            if (executionResult.success) {
                this.metrics.successfulExecutions++;
            }
            else {
                this.metrics.failedExecutions++;
            }
            const latency = Date.now() - startTime;
            this.metrics.totalLatency += latency;
            this.metrics.executionCount++;
            // Broadcast execution result
            this.broadcastToClients({
                type: 'execution',
                data: executionResult,
                timestamp: Date.now(),
                id: `execution_${signal.id}`
            });
            logger_1.logger.info(`✅ Signal processed: ${signal.id} (${latency}ms)`);
        }
        catch (error) {
            const latency = Date.now() - startTime;
            const executionResult = {
                signalId: signal.id,
                success: false,
                executedAt: Date.now(),
                error: error.message,
                latency
            };
            this.executionResults.set(signal.id, executionResult);
            this.metrics.failedExecutions++;
            this.metrics.totalLatency += latency;
            this.metrics.executionCount++;
            logger_1.logger.error(`❌ Signal execution failed: ${signal.id} - ${error.message}`);
            // Broadcast error
            this.broadcastToClients({
                type: 'error',
                data: { signalId: signal.id, error: error.message },
                timestamp: Date.now(),
                id: `error_${signal.id}`
            });
        }
    }
    /**
     * Execute regular trading signal
     */
    async executeSignal(signal) {
        const startTime = Date.now();
        // Risk assessment for the signal
        if (signal.source === 'ml_decision' || signal.source === 'manual') {
            // Create mock decision for risk assessment
            const mockDecision = {
                action: signal.action,
                confidence: signal.confidence,
                symbol: signal.symbol,
                timestamp: signal.timestamp,
                stopLoss: signal.metadata.stopLoss || 0,
                takeProfit: signal.metadata.takeProfit || 0,
                positionSize: signal.metadata.positionSize || 0.05,
                leverage: signal.metadata.leverage || 100,
                modelVotes: {},
                keyFeatures: {},
                riskScore: signal.metadata.riskScore || 0.5,
                maxDrawdown: 0,
                winProbability: 0,
                urgency: 'medium',
                timeToLive: 300000,
                reasoning: []
            };
            const riskAssessment = await this.riskManager.assessTradingRisk(mockDecision, 50000);
            if (!riskAssessment.isAcceptable) {
                throw new ExecutionError(`Risk assessment failed: ${riskAssessment.riskFactors.join(', ')}`);
            }
        }
        // Execute the signal based on action
        switch (signal.action) {
            case 'buy':
            case 'sell':
                return await this.executeEntry(signal);
            case 'close':
                return await this.executeExit(signal);
            case 'hold':
                return {
                    signalId: signal.id,
                    success: true,
                    executedAt: Date.now(),
                    latency: Date.now() - startTime
                };
            default:
                throw new ExecutionError(`Unknown signal action: ${signal.action}`);
        }
    }
    /**
     * Execute entry signal (buy/sell)
     */
    async executeEntry(signal) {
        const startTime = Date.now();
        try {
            // Create position from signal
            const mockDecision = {
                action: signal.action,
                confidence: signal.confidence,
                symbol: signal.symbol,
                timestamp: signal.timestamp,
                stopLoss: signal.metadata.stopLoss || 0,
                takeProfit: signal.metadata.takeProfit || 0,
                positionSize: signal.metadata.positionSize || 0.05,
                leverage: signal.metadata.leverage || 100,
                modelVotes: {},
                keyFeatures: {},
                riskScore: signal.metadata.riskScore || 0.5,
                maxDrawdown: 0,
                winProbability: 0,
                urgency: 'medium',
                timeToLive: 300000,
                reasoning: []
            };
            const currentPrice = 50000; // Mock price - would get from market data
            const position = await this.positionManager.createPosition(mockDecision, currentPrice);
            if (!position) {
                throw new ExecutionError('Failed to create position');
            }
            return {
                signalId: signal.id,
                success: true,
                executedAt: Date.now(),
                orderId: position.id,
                executedPrice: currentPrice,
                executedQuantity: position.quantity,
                latency: Date.now() - startTime
            };
        }
        catch (error) {
            throw new ExecutionError(`Entry execution failed: ${error.message}`);
        }
    }
    /**
     * Execute exit signal
     */
    async executeExit(signal) {
        const startTime = Date.now();
        try {
            const activePositions = this.positionManager.getActivePositions();
            const symbolPositions = activePositions.filter(pos => pos.symbol === signal.symbol);
            if (symbolPositions.length === 0) {
                throw new ExecutionError(`No active positions found for ${signal.symbol}`);
            }
            let closedPositions = 0;
            for (const position of symbolPositions) {
                const success = await this.positionManager.closePosition(position.id, position.currentPrice, 'Signal-triggered close');
                if (success)
                    closedPositions++;
            }
            return {
                signalId: signal.id,
                success: closedPositions > 0,
                executedAt: Date.now(),
                executedQuantity: closedPositions,
                latency: Date.now() - startTime
            };
        }
        catch (error) {
            throw new ExecutionError(`Exit execution failed: ${error.message}`);
        }
    }
    /**
     * Execute emergency close all positions
     */
    async executeEmergencyClose(signal) {
        const startTime = Date.now();
        try {
            const activePositions = this.positionManager.getActivePositions();
            let closedPositions = 0;
            for (const position of activePositions) {
                try {
                    const success = await this.positionManager.closePosition(position.id, position.currentPrice, 'Emergency close');
                    if (success)
                        closedPositions++;
                }
                catch (error) {
                    logger_1.logger.error(`❌ Failed to close position ${position.id}:`, error.message);
                }
            }
            return {
                signalId: signal.id,
                success: closedPositions > 0,
                executedAt: Date.now(),
                executedQuantity: closedPositions,
                latency: Date.now() - startTime
            };
        }
        catch (error) {
            throw new ExecutionError(`Emergency close failed: ${error.message}`);
        }
    }
    /**
     * Handle WebSocket message
     */
    async handleWebSocketMessage(ws, message, clientId) {
        try {
            const { type, data } = message;
            switch (type) {
                case 'ping':
                    this.sendToClient(ws, {
                        type: 'pong',
                        data: { timestamp: Date.now() },
                        timestamp: Date.now(),
                        id: `pong_${Date.now()}`
                    });
                    break;
                case 'get_status':
                    this.sendToClient(ws, {
                        type: 'status',
                        data: this.getStatus(),
                        timestamp: Date.now(),
                        id: `status_${Date.now()}`
                    });
                    break;
                case 'get_positions':
                    const positions = this.positionManager.getActivePositions();
                    const metrics = this.positionManager.getPerformanceMetrics();
                    this.sendToClient(ws, {
                        type: 'positions',
                        data: { positions, metrics },
                        timestamp: Date.now(),
                        id: `positions_${Date.now()}`
                    });
                    break;
                case 'get_risk':
                    const riskMetrics = this.riskManager.getRiskMetrics();
                    const failsafeMechanisms = this.riskManager.getFailsafeMechanisms();
                    this.sendToClient(ws, {
                        type: 'risk',
                        data: { metrics: riskMetrics, failsafeMechanisms },
                        timestamp: Date.now(),
                        id: `risk_${Date.now()}`
                    });
                    break;
                case 'send_signal':
                    if (!data.symbol || !data.action) {
                        throw new ValidationError('Symbol and action are required');
                    }
                    const signalId = await this.sendTradingSignal({
                        symbol: data.symbol,
                        action: data.action,
                        confidence: data.confidence || 0.5,
                        source: 'manual',
                        metadata: data.metadata || {}
                    });
                    this.sendToClient(ws, {
                        type: 'signal_sent',
                        data: { signalId },
                        timestamp: Date.now(),
                        id: `signal_sent_${signalId}`
                    });
                    break;
                default:
                    logger_1.logger.warn(`⚠️ Unknown WebSocket message type: ${type} from ${clientId}`);
                    this.sendToClient(ws, {
                        type: 'error',
                        data: { error: `Unknown message type: ${type}` },
                        timestamp: Date.now(),
                        id: `error_${Date.now()}`
                    });
            }
        }
        catch (error) {
            logger_1.logger.error(`❌ WebSocket message handling error from ${clientId}:`, error.message);
            this.sendToClient(ws, {
                type: 'error',
                data: { error: error.message },
                timestamp: Date.now(),
                id: `error_${Date.now()}`
            });
        }
    }
    /**
     * Send message to specific WebSocket client
     */
    sendToClient(ws, message) {
        try {
            if (ws.readyState === ws_1.WebSocket.OPEN) {
                ws.send(JSON.stringify(message));
            }
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to send WebSocket message:', error.message);
        }
    }
    /**
     * Broadcast message to all connected WebSocket clients
     */
    broadcastToClients(message) {
        const messageStr = JSON.stringify(message);
        this.connectedClients.forEach(client => {
            try {
                if (client.readyState === ws_1.WebSocket.OPEN) {
                    client.send(messageStr);
                }
            }
            catch (error) {
                logger_1.logger.error('❌ Failed to broadcast WebSocket message:', error.message);
                // Remove failed client
                this.connectedClients.delete(client);
            }
        });
        logger_1.logger.debug(`📡 Broadcasted ${message.type} to ${this.connectedClients.size} clients`);
    }
}
exports.AnalysisExecutionBridge = AnalysisExecutionBridge;
//# sourceMappingURL=AnalysisExecutionBridge.js.map