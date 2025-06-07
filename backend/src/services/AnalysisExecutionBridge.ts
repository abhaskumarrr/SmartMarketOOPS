/**
 * Analysis-Execution Bridge
 * Real-time coordination layer between analysis and execution engines
 * Built with FastAPI-style architecture, WebSocket support, and robust error handling
 */

import express, { Express, Request, Response, NextFunction } from 'express';
import { createServer, Server } from 'http';
import { WebSocketServer, WebSocket } from 'ws';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import { EnhancedTradingDecisionEngine, TradingDecision } from './EnhancedTradingDecisionEngine';
import { MLPositionManager, Position } from './MLPositionManager';
import { EnhancedRiskManagementSystem, RiskAssessment } from './EnhancedRiskManagementSystem';
import { DataCollectorIntegration } from './DataCollectorIntegration';
import { DeltaTradingBot } from './DeltaTradingBot';
import { logger } from '../utils/logger';

// Bridge types and interfaces
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

// WebSocket message types
export interface WebSocketMessage {
  type: 'signal' | 'execution' | 'status' | 'error' | 'heartbeat';
  data: any;
  timestamp: number;
  id: string;
}

// Custom error classes
export class BridgeError extends Error {
  constructor(
    message: string,
    public statusCode: number = 500,
    public code: string = 'BRIDGE_ERROR'
  ) {
    super(message);
    this.name = 'BridgeError';
  }
}

export class ValidationError extends BridgeError {
  constructor(message: string) {
    super(message, 400, 'VALIDATION_ERROR');
    this.name = 'ValidationError';
  }
}

export class ExecutionError extends BridgeError {
  constructor(message: string) {
    super(message, 500, 'EXECUTION_ERROR');
    this.name = 'ExecutionError';
  }
}

export class AnalysisExecutionBridge {
  private app: Express;
  private server: Server;
  private wss: WebSocketServer;
  private decisionEngine: EnhancedTradingDecisionEngine;
  private positionManager: MLPositionManager;
  private riskManager: EnhancedRiskManagementSystem;
  private dataIntegration: DataCollectorIntegration;
  private tradingBot: DeltaTradingBot;
  
  // Bridge state
  private isRunning: boolean = false;
  private connectedClients: Set<WebSocket> = new Set();
  private signalQueue: TradingSignal[] = [];
  private executionResults: Map<string, ExecutionResult> = new Map();
  private startTime: number = 0;
  
  // Performance metrics
  private metrics = {
    totalSignals: 0,
    successfulExecutions: 0,
    failedExecutions: 0,
    totalLatency: 0,
    executionCount: 0
  };

  // Configuration optimized for high-frequency trading
  private config: BridgeConfig = {
    port: 8000,
    host: '0.0.0.0',
    enableWebSocket: true,
    enableRateLimit: true,
    maxRequestsPerMinute: 1000,     // High rate limit for trading
    corsOrigins: ['http://localhost:3000', 'http://localhost:3001'],
    enableHelmet: true
  };

  constructor(customConfig?: Partial<BridgeConfig>) {
    if (customConfig) {
      this.config = { ...this.config, ...customConfig };
    }

    // Initialize Express app
    this.app = express();
    this.server = createServer(this.app);
    
    // Initialize WebSocket server
    this.wss = new WebSocketServer({ server: this.server });
    
    // Initialize trading components
    this.decisionEngine = new EnhancedTradingDecisionEngine();
    this.positionManager = new MLPositionManager();
    this.riskManager = new EnhancedRiskManagementSystem();
    this.dataIntegration = new DataCollectorIntegration();
    this.tradingBot = new DeltaTradingBot();
  }

  /**
   * Initialize the Analysis-Execution Bridge
   */
  public async initialize(): Promise<void> {
    try {
      logger.info('🌉 Initializing Analysis-Execution Bridge...');
      
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
      
      logger.info('✅ Analysis-Execution Bridge initialized successfully');
      logger.info(`🔧 Configuration: Port ${this.config.port}, WebSocket ${this.config.enableWebSocket ? 'ON' : 'OFF'}, Rate Limit ${this.config.maxRequestsPerMinute}/min`);
      
    } catch (error: any) {
      logger.error('❌ Failed to initialize Analysis-Execution Bridge:', error.message);
      throw error;
    }
  }

  /**
   * Start the bridge server
   */
  public async start(): Promise<void> {
    try {
      if (this.isRunning) {
        logger.warn('⚠️ Bridge is already running');
        return;
      }

      this.startTime = Date.now();
      
      // Start the server
      await new Promise<void>((resolve, reject) => {
        this.server.listen(this.config.port, this.config.host, () => {
          this.isRunning = true;
          logger.info(`🚀 Analysis-Execution Bridge started on ${this.config.host}:${this.config.port}`);
          resolve();
        });
        
        this.server.on('error', (error) => {
          logger.error('❌ Server error:', error);
          reject(error);
        });
      });

      // Start signal processing
      this.startSignalProcessing();
      
      logger.info('🌉 Analysis-Execution Bridge is now operational');
      
    } catch (error: any) {
      logger.error('❌ Failed to start Analysis-Execution Bridge:', error.message);
      throw error;
    }
  }

  /**
   * Stop the bridge server
   */
  public async stop(): Promise<void> {
    try {
      if (!this.isRunning) {
        logger.warn('⚠️ Bridge is not running');
        return;
      }

      logger.info('🛑 Stopping Analysis-Execution Bridge...');
      
      // Close WebSocket connections
      this.connectedClients.forEach(client => {
        if (client.readyState === WebSocket.OPEN) {
          client.close(1000, 'Server shutdown');
        }
      });
      this.connectedClients.clear();
      
      // Close WebSocket server
      this.wss.close();
      
      // Close HTTP server
      await new Promise<void>((resolve) => {
        this.server.close(() => {
          this.isRunning = false;
          logger.info('✅ Analysis-Execution Bridge stopped');
          resolve();
        });
      });
      
    } catch (error: any) {
      logger.error('❌ Error stopping Analysis-Execution Bridge:', error.message);
      throw error;
    }
  }

  /**
   * Get bridge status
   */
  public getStatus(): BridgeStatus {
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
  public async sendTradingSignal(signal: Omit<TradingSignal, 'id' | 'timestamp'>): Promise<string> {
    try {
      const tradingSignal: TradingSignal = {
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

      logger.info(`📡 Trading signal sent: ${tradingSignal.action.toUpperCase()} ${tradingSignal.symbol} (${(tradingSignal.confidence * 100).toFixed(1)}%)`);
      
      return tradingSignal.id;

    } catch (error: any) {
      logger.error('❌ Failed to send trading signal:', error.message);
      throw new BridgeError(`Failed to send trading signal: ${error.message}`);
    }
  }

  /**
   * Get execution result for a signal
   */
  public getExecutionResult(signalId: string): ExecutionResult | null {
    return this.executionResults.get(signalId) || null;
  }

  /**
   * Cleanup resources
   */
  public async cleanup(): Promise<void> {
    try {
      logger.info('🧹 Cleaning up Analysis-Execution Bridge...');
      
      await this.stop();
      
      // Cleanup trading components
      await this.positionManager.cleanup();
      await this.riskManager.cleanup();
      await this.dataIntegration.cleanup();
      
      logger.info('✅ Analysis-Execution Bridge cleanup completed');
    } catch (error: any) {
      logger.error('❌ Error during Analysis-Execution Bridge cleanup:', error.message);
    }
  }

  // Private methods for bridge implementation

  /**
   * Setup Express middleware
   */
  private setupMiddleware(): void {
    // Security middleware
    if (this.config.enableHelmet) {
      this.app.use(helmet({
        contentSecurityPolicy: false, // Disable for WebSocket support
        crossOriginEmbedderPolicy: false
      }));
    }

    // CORS middleware
    this.app.use(cors({
      origin: this.config.corsOrigins,
      credentials: true,
      methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
      allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
    }));

    // Rate limiting
    if (this.config.enableRateLimit) {
      const limiter = rateLimit({
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
    this.app.use(express.json({ limit: '10mb' }));
    this.app.use(express.urlencoded({ extended: true, limit: '10mb' }));

    // Request logging middleware
    this.app.use((req: Request, res: Response, next: NextFunction) => {
      const start = Date.now();
      res.on('finish', () => {
        const duration = Date.now() - start;
        logger.debug(`${req.method} ${req.path} - ${res.statusCode} (${duration}ms)`);
      });
      next();
    });
  }

  /**
   * Setup API routes
   */
  private setupRoutes(): void {
    // Health check endpoint
    this.app.get('/health', (req: Request, res: Response) => {
      const status = this.getStatus();
      res.json({
        status: 'healthy',
        timestamp: Date.now(),
        bridge: status
      });
    });

    // Get bridge status
    this.app.get('/api/status', (req: Request, res: Response) => {
      try {
        const status = this.getStatus();
        res.json(status);
      } catch (error: any) {
        res.status(500).json({ error: error.message });
      }
    });

    // Generate trading decision
    this.app.post('/api/decisions/:symbol', async (req: Request, res: Response, next: NextFunction) => {
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

      } catch (error) {
        next(error);
      }
    });

    // Get positions
    this.app.get('/api/positions', async (req: Request, res: Response, next: NextFunction) => {
      try {
        const positions = this.positionManager.getActivePositions();
        const metrics = this.positionManager.getPerformanceMetrics();

        res.json({
          positions,
          metrics,
          timestamp: Date.now()
        });

      } catch (error) {
        next(error);
      }
    });

    // Get risk metrics
    this.app.get('/api/risk', async (req: Request, res: Response, next: NextFunction) => {
      try {
        const riskMetrics = this.riskManager.getRiskMetrics();
        const failsafeMechanisms = this.riskManager.getFailsafeMechanisms();

        res.json({
          metrics: riskMetrics,
          failsafeMechanisms,
          timestamp: Date.now()
        });

      } catch (error) {
        next(error);
      }
    });

    // Manual trading signal
    this.app.post('/api/signals', async (req: Request, res: Response, next: NextFunction) => {
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

      } catch (error) {
        next(error);
      }
    });

    // Get execution result
    this.app.get('/api/executions/:signalId', (req: Request, res: Response, next: NextFunction) => {
      try {
        const { signalId } = req.params;
        const result = this.getExecutionResult(signalId);

        if (!result) {
          res.status(404).json({ error: 'Execution result not found' });
          return;
        }

        res.json(result);

      } catch (error) {
        next(error);
      }
    });

    // Emergency stop
    this.app.post('/api/emergency-stop', async (req: Request, res: Response, next: NextFunction) => {
      try {
        logger.warn('🚨 Emergency stop triggered via API');

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

      } catch (error) {
        next(error);
      }
    });
  }

  /**
   * Setup WebSocket handlers
   */
  private setupWebSocketHandlers(): void {
    if (!this.config.enableWebSocket) {
      return;
    }

    this.wss.on('connection', (ws: WebSocket, req) => {
      const clientId = `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      logger.info(`🔌 WebSocket client connected: ${clientId}`);
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
      ws.on('message', async (data: Buffer) => {
        try {
          const message = JSON.parse(data.toString());
          await this.handleWebSocketMessage(ws, message, clientId);
        } catch (error: any) {
          logger.error(`❌ WebSocket message error from ${clientId}:`, error.message);
          this.sendToClient(ws, {
            type: 'error',
            data: { error: 'Invalid message format' },
            timestamp: Date.now(),
            id: `error_${Date.now()}`
          });
        }
      });

      // Handle client disconnect
      ws.on('close', (code: number, reason: Buffer) => {
        logger.info(`🔌 WebSocket client disconnected: ${clientId} (${code})`);
        this.connectedClients.delete(ws);
      });

      // Handle WebSocket errors
      ws.on('error', (error: Error) => {
        logger.error(`❌ WebSocket error from ${clientId}:`, error.message);
        this.connectedClients.delete(ws);
      });

      // Send periodic heartbeat
      const heartbeatInterval = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          this.sendToClient(ws, {
            type: 'heartbeat',
            data: { timestamp: Date.now() },
            timestamp: Date.now(),
            id: `heartbeat_${Date.now()}`
          });
        } else {
          clearInterval(heartbeatInterval);
        }
      }, 30000); // 30 seconds
    });

    logger.info('🔌 WebSocket server configured');
  }

  /**
   * Setup error handlers
   */
  private setupErrorHandlers(): void {
    // Custom error handler
    this.app.use((error: Error, req: Request, res: Response, next: NextFunction) => {
      logger.error(`❌ API Error: ${error.message}`, error.stack);

      if (error instanceof ValidationError) {
        res.status(error.statusCode).json({
          error: error.message,
          code: error.code,
          timestamp: Date.now()
        });
      } else if (error instanceof ExecutionError) {
        res.status(error.statusCode).json({
          error: error.message,
          code: error.code,
          timestamp: Date.now()
        });
      } else if (error instanceof BridgeError) {
        res.status(error.statusCode).json({
          error: error.message,
          code: error.code,
          timestamp: Date.now()
        });
      } else {
        // Generic error
        res.status(500).json({
          error: 'Internal server error',
          code: 'INTERNAL_ERROR',
          timestamp: Date.now()
        });
      }
    });

    // 404 handler
    this.app.use((req: Request, res: Response) => {
      res.status(404).json({
        error: 'Endpoint not found',
        code: 'NOT_FOUND',
        path: req.path,
        timestamp: Date.now()
      });
    });

    // Unhandled promise rejection handler
    process.on('unhandledRejection', (reason: any, promise: Promise<any>) => {
      logger.error('❌ Unhandled promise rejection:', reason);
    });

    // Uncaught exception handler
    process.on('uncaughtException', (error: Error) => {
      logger.error('❌ Uncaught exception:', error);
      // Graceful shutdown
      this.stop().catch(() => {
        process.exit(1);
      });
    });
  }

  /**
   * Start signal processing loop
   */
  private startSignalProcessing(): void {
    const processSignals = async () => {
      try {
        if (this.signalQueue.length > 0) {
          const signal = this.signalQueue.shift();
          if (signal) {
            await this.processSignal(signal);
          }
        }
      } catch (error: any) {
        logger.error('❌ Signal processing error:', error.message);
      }
    };

    // Process signals every 100ms for low latency
    setInterval(processSignals, 100);

    logger.info('⚡ Signal processing started (100ms intervals)');
  }

  /**
   * Process individual trading signal
   */
  private async processSignal(signal: TradingSignal): Promise<void> {
    const startTime = Date.now();

    try {
      logger.debug(`🔄 Processing signal: ${signal.id}`);

      let executionResult: ExecutionResult;

      if (signal.action === 'close' && signal.symbol === 'ALL') {
        // Emergency close all positions
        executionResult = await this.executeEmergencyClose(signal);
      } else {
        // Regular signal processing
        executionResult = await this.executeSignal(signal);
      }

      // Store execution result
      this.executionResults.set(signal.id, executionResult);

      // Update metrics
      if (executionResult.success) {
        this.metrics.successfulExecutions++;
      } else {
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

      logger.info(`✅ Signal processed: ${signal.id} (${latency}ms)`);

    } catch (error: any) {
      const latency = Date.now() - startTime;

      const executionResult: ExecutionResult = {
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

      logger.error(`❌ Signal execution failed: ${signal.id} - ${error.message}`);

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
  private async executeSignal(signal: TradingSignal): Promise<ExecutionResult> {
    const startTime = Date.now();

    // Risk assessment for the signal
    if (signal.source === 'ml_decision' || signal.source === 'manual') {
      // Create mock decision for risk assessment
      const mockDecision: TradingDecision = {
        action: signal.action as any,
        confidence: signal.confidence,
        symbol: signal.symbol,
        timestamp: signal.timestamp,
        stopLoss: signal.metadata.stopLoss || 0,
        takeProfit: signal.metadata.takeProfit || 0,
        positionSize: signal.metadata.positionSize || 0.05,
        leverage: signal.metadata.leverage || 100,
        modelVotes: {} as any,
        keyFeatures: {} as any,
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
  private async executeEntry(signal: TradingSignal): Promise<ExecutionResult> {
    const startTime = Date.now();

    try {
      // Create position from signal
      const mockDecision: TradingDecision = {
        action: signal.action as any,
        confidence: signal.confidence,
        symbol: signal.symbol,
        timestamp: signal.timestamp,
        stopLoss: signal.metadata.stopLoss || 0,
        takeProfit: signal.metadata.takeProfit || 0,
        positionSize: signal.metadata.positionSize || 0.05,
        leverage: signal.metadata.leverage || 100,
        modelVotes: {} as any,
        keyFeatures: {} as any,
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

    } catch (error: any) {
      throw new ExecutionError(`Entry execution failed: ${error.message}`);
    }
  }

  /**
   * Execute exit signal
   */
  private async executeExit(signal: TradingSignal): Promise<ExecutionResult> {
    const startTime = Date.now();

    try {
      const activePositions = this.positionManager.getActivePositions();
      const symbolPositions = activePositions.filter(pos => pos.symbol === signal.symbol);

      if (symbolPositions.length === 0) {
        throw new ExecutionError(`No active positions found for ${signal.symbol}`);
      }

      let closedPositions = 0;
      for (const position of symbolPositions) {
        const success = await this.positionManager.closePosition(
          position.id,
          position.currentPrice,
          'Signal-triggered close'
        );
        if (success) closedPositions++;
      }

      return {
        signalId: signal.id,
        success: closedPositions > 0,
        executedAt: Date.now(),
        executedQuantity: closedPositions,
        latency: Date.now() - startTime
      };

    } catch (error: any) {
      throw new ExecutionError(`Exit execution failed: ${error.message}`);
    }
  }

  /**
   * Execute emergency close all positions
   */
  private async executeEmergencyClose(signal: TradingSignal): Promise<ExecutionResult> {
    const startTime = Date.now();

    try {
      const activePositions = this.positionManager.getActivePositions();
      let closedPositions = 0;

      for (const position of activePositions) {
        try {
          const success = await this.positionManager.closePosition(
            position.id,
            position.currentPrice,
            'Emergency close'
          );
          if (success) closedPositions++;
        } catch (error: any) {
          logger.error(`❌ Failed to close position ${position.id}:`, error.message);
        }
      }

      return {
        signalId: signal.id,
        success: closedPositions > 0,
        executedAt: Date.now(),
        executedQuantity: closedPositions,
        latency: Date.now() - startTime
      };

    } catch (error: any) {
      throw new ExecutionError(`Emergency close failed: ${error.message}`);
    }
  }

  /**
   * Handle WebSocket message
   */
  private async handleWebSocketMessage(ws: WebSocket, message: any, clientId: string): Promise<void> {
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
          logger.warn(`⚠️ Unknown WebSocket message type: ${type} from ${clientId}`);
          this.sendToClient(ws, {
            type: 'error',
            data: { error: `Unknown message type: ${type}` },
            timestamp: Date.now(),
            id: `error_${Date.now()}`
          });
      }

    } catch (error: any) {
      logger.error(`❌ WebSocket message handling error from ${clientId}:`, error.message);
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
  private sendToClient(ws: WebSocket, message: WebSocketMessage): void {
    try {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(message));
      }
    } catch (error: any) {
      logger.error('❌ Failed to send WebSocket message:', error.message);
    }
  }

  /**
   * Broadcast message to all connected WebSocket clients
   */
  private broadcastToClients(message: WebSocketMessage): void {
    const messageStr = JSON.stringify(message);

    this.connectedClients.forEach(client => {
      try {
        if (client.readyState === WebSocket.OPEN) {
          client.send(messageStr);
        }
      } catch (error: any) {
        logger.error('❌ Failed to broadcast WebSocket message:', error.message);
        // Remove failed client
        this.connectedClients.delete(client);
      }
    });

    logger.debug(`📡 Broadcasted ${message.type} to ${this.connectedClients.size} clients`);
  }
}
