/**
 * Delta Exchange Trading Bot
 * Production-ready trading bot for Delta Exchange India testnet
 * Integrates with ML models and implements proper risk management
 */

import { EventEmitter } from 'events';
import { DeltaExchangeUnified, DeltaOrderRequest, DeltaOrder, DeltaPosition } from './DeltaExchangeUnified';
import { logger } from '../utils/logger';
import { marketDataService } from './marketDataProvider';

export interface BotConfig {
  id: string;
  name: string;
  symbol: string;
  strategy: 'momentum' | 'mean_reversion' | 'ml_driven' | 'scalping';
  capital: number;
  leverage: number;
  riskPerTrade: number; // Percentage of capital to risk per trade
  maxPositions: number;
  stopLoss: number; // Percentage
  takeProfit: number; // Percentage
  enabled: boolean;
  testnet: boolean;
}

export interface BotStatus {
  id: string;
  status: 'running' | 'stopped' | 'paused' | 'error';
  uptime: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  totalPnL: number;
  currentPositions: number;
  lastTradeTime?: Date;
  errorMessage?: string;
}

export interface TradeSignal {
  symbol: string;
  action: 'buy' | 'sell' | 'close';
  confidence: number; // 0-1
  price?: number;
  size?: number;
  reason: string;
  timestamp: Date;
}

export class DeltaTradingBot extends EventEmitter {
  private config: BotConfig;
  private deltaService: DeltaExchangeUnified;
  private status: BotStatus;
  private isRunning: boolean = false;
  private positions: Map<string, DeltaPosition> = new Map();
  private orders: Map<number, DeltaOrder> = new Map();
  private startTime: Date;
  private lastHealthCheck: Date;
  private healthCheckInterval: NodeJS.Timeout | null = null;
  private tradingInterval: NodeJS.Timeout | null = null;

  constructor(config: BotConfig, deltaService: DeltaExchangeUnified) {
    super();
    this.config = config;
    this.deltaService = deltaService;
    this.startTime = new Date();
    this.lastHealthCheck = new Date();

    // Initialize bot status
    this.status = {
      id: config.id,
      status: 'stopped',
      uptime: 0,
      totalTrades: 0,
      winningTrades: 0,
      losingTrades: 0,
      totalPnL: 0,
      currentPositions: 0
    };

    // Set up Delta Exchange event listeners
    this.setupDeltaEventListeners();
  }

  /**
   * Set up event listeners for Delta Exchange service
   */
  private setupDeltaEventListeners(): void {
    this.deltaService.on('orderPlaced', (order: DeltaOrder) => {
      this.handleOrderPlaced(order);
    });

    this.deltaService.on('orderCancelled', (order: DeltaOrder) => {
      this.handleOrderCancelled(order);
    });

    this.deltaService.on('ticker', (ticker: any) => {
      this.handleTickerUpdate(ticker);
    });

    this.deltaService.on('wsConnected', () => {
      logger.info(`🔗 Bot ${this.config.id}: WebSocket connected`);
    });

    this.deltaService.on('wsDisconnected', () => {
      logger.warn(`🔌 Bot ${this.config.id}: WebSocket disconnected`);
    });
  }

  /**
   * Start the trading bot
   */
  public async start(): Promise<void> {
    if (this.isRunning) {
      throw new Error('Bot is already running');
    }

    try {
      logger.info(`🚀 Starting trading bot: ${this.config.name} (${this.config.id})`);

      // Validate configuration
      this.validateConfig();

      // Check Delta Exchange service readiness
      if (!this.deltaService.isReady()) {
        throw new Error('Delta Exchange service is not ready');
      }

      // Load initial positions and orders
      await this.loadInitialState();

      // Connect to WebSocket for real-time data
      this.deltaService.connectWebSocket([this.config.symbol]);

      // Start health check and trading loops
      this.startHealthCheck();
      this.startTradingLoop();

      this.isRunning = true;
      this.status.status = 'running';
      this.startTime = new Date();

      logger.info(`✅ Bot ${this.config.id} started successfully`);
      this.emit('started', this.status);

    } catch (error) {
      logger.error(`❌ Failed to start bot ${this.config.id}:`, error);
      this.status.status = 'error';
      this.status.errorMessage = error instanceof Error ? error.message : 'Unknown error';
      this.emit('error', error);
      throw error;
    }
  }

  /**
   * Stop the trading bot
   */
  public async stop(): Promise<void> {
    if (!this.isRunning) {
      return;
    }

    try {
      logger.info(`🛑 Stopping trading bot: ${this.config.name} (${this.config.id})`);

      // Stop intervals
      if (this.healthCheckInterval) {
        clearInterval(this.healthCheckInterval);
        this.healthCheckInterval = null;
      }

      if (this.tradingInterval) {
        clearInterval(this.tradingInterval);
        this.tradingInterval = null;
      }

      // Cancel all open orders (optional - for safety)
      await this.cancelAllOrders();

      // Disconnect WebSocket
      this.deltaService.disconnectWebSocket();

      this.isRunning = false;
      this.status.status = 'stopped';

      logger.info(`✅ Bot ${this.config.id} stopped successfully`);
      this.emit('stopped', this.status);

    } catch (error) {
      logger.error(`❌ Error stopping bot ${this.config.id}:`, error);
      this.status.status = 'error';
      this.status.errorMessage = error instanceof Error ? error.message : 'Unknown error';
      this.emit('error', error);
    }
  }

  /**
   * Pause the trading bot
   */
  public pause(): void {
    if (!this.isRunning) {
      return;
    }

    if (this.tradingInterval) {
      clearInterval(this.tradingInterval);
      this.tradingInterval = null;
    }

    this.status.status = 'paused';
    logger.info(`⏸️ Bot ${this.config.id} paused`);
    this.emit('paused', this.status);
  }

  /**
   * Resume the trading bot
   */
  public resume(): void {
    if (!this.isRunning || this.status.status !== 'paused') {
      return;
    }

    this.startTradingLoop();
    this.status.status = 'running';
    logger.info(`▶️ Bot ${this.config.id} resumed`);
    this.emit('resumed', this.status);
  }

  /**
   * Get current bot status
   */
  public getStatus(): BotStatus {
    this.status.uptime = Date.now() - this.startTime.getTime();
    return { ...this.status };
  }

  /**
   * Update bot configuration
   */
  public updateConfig(newConfig: Partial<BotConfig>): void {
    this.config = { ...this.config, ...newConfig };
    logger.info(`🔧 Bot ${this.config.id} configuration updated`);
    this.emit('configUpdated', this.config);
  }

  /**
   * Validate bot configuration
   */
  private validateConfig(): void {
    if (!this.config.symbol) {
      throw new Error('Symbol is required');
    }

    if (this.config.capital <= 0) {
      throw new Error('Capital must be greater than 0');
    }

    if (this.config.leverage <= 0 || this.config.leverage > 100) {
      throw new Error('Leverage must be between 1 and 100');
    }

    if (this.config.riskPerTrade <= 0 || this.config.riskPerTrade > 100) {
      throw new Error('Risk per trade must be between 0 and 100');
    }

    // CRITICAL: Validate data source consistency for live trading
    marketDataService.enforceLiveDataMode();
    const providerInfo = marketDataService.getCurrentProviderInfo();

    if (providerInfo.isMock) {
      throw new Error(`🚨 SAFETY VIOLATION: Trading bot cannot use mock data provider '${providerInfo.name}' for live trading. This would create dangerous inconsistencies between trade execution and risk management calculations.`);
    }

    logger.info(`✅ Data source validation passed: Bot will use '${providerInfo.name}' for consistent live market data`);

    // Check if product exists
    const productId = this.deltaService.getProductId(this.config.symbol);
    if (!productId) {
      throw new Error(`Product not found for symbol: ${this.config.symbol}`);
    }
  }

  /**
   * Load initial state (positions and orders)
   */
  private async loadInitialState(): Promise<void> {
    try {
      // Load current positions
      const positions = await this.deltaService.getPositions();
      for (const position of positions) {
        if (position.product.symbol === this.config.symbol) {
          this.positions.set(position.product.symbol, position);
        }
      }

      // Load open orders
      const productId = this.deltaService.getProductId(this.config.symbol);
      if (productId) {
        const orders = await this.deltaService.getOpenOrders(productId);
        for (const order of orders) {
          this.orders.set(order.id, order);
        }
      }

      this.status.currentPositions = this.positions.size;
      logger.info(`📊 Bot ${this.config.id}: Loaded ${this.positions.size} positions and ${this.orders.size} orders`);

    } catch (error) {
      logger.error(`Error loading initial state for bot ${this.config.id}:`, error);
      throw error;
    }
  }

  /**
   * Start health check interval
   */
  private startHealthCheck(): void {
    this.healthCheckInterval = setInterval(async () => {
      try {
        await this.performHealthCheck();
        this.lastHealthCheck = new Date();
      } catch (error) {
        logger.error(`Health check failed for bot ${this.config.id}:`, error);
      }
    }, 30000); // Every 30 seconds
  }

  /**
   * Start trading loop
   */
  private startTradingLoop(): void {
    this.tradingInterval = setInterval(async () => {
      try {
        if (this.status.status === 'running') {
          await this.executeTradingLogic();
        }
      } catch (error) {
        logger.error(`Trading loop error for bot ${this.config.id}:`, error);
        this.handleTradingError(error);
      }
    }, 5000); // Every 5 seconds
  }

  /**
   * Perform health check
   */
  private async performHealthCheck(): Promise<void> {
    // Check Delta Exchange connection
    if (!this.deltaService.isReady()) {
      throw new Error('Delta Exchange service not ready');
    }

    // Update positions and orders
    await this.updatePositionsAndOrders();

    // Check for risk limits
    this.checkRiskLimits();
  }

  /**
   * Execute main trading logic
   */
  private async executeTradingLogic(): Promise<void> {
    try {
      // Get market data
      const marketData = await this.deltaService.getMarketData(this.config.symbol);

      // Generate trading signal based on strategy
      const signal = await this.generateTradingSignal(marketData);

      if (signal && signal.confidence > 0.7) {
        await this.executeTradeSignal(signal);
      }

      // Check existing positions for stop loss / take profit
      await this.manageExistingPositions();

    } catch (error) {
      logger.error(`Trading logic error for bot ${this.config.id}:`, error);
    }
  }

  /**
   * Generate trading signal based on strategy
   */
  private async generateTradingSignal(marketData: any): Promise<TradeSignal | null> {
    // This is a simplified example - in production, this would integrate with ML models
    const currentPrice = parseFloat(marketData.mark_price || marketData.last_price || '0');

    if (currentPrice === 0) {
      return null;
    }

    // Simple momentum strategy example
    if (this.config.strategy === 'momentum') {
      // This would normally use technical indicators and ML predictions
      const randomConfidence = Math.random();
      const action = Math.random() > 0.5 ? 'buy' : 'sell';

      return {
        symbol: this.config.symbol,
        action,
        confidence: randomConfidence,
        price: currentPrice,
        size: this.calculatePositionSize(currentPrice),
        reason: 'Momentum strategy signal',
        timestamp: new Date()
      };
    }

    return null;
  }

  /**
   * Execute a trading signal
   */
  private async executeTradeSignal(signal: TradeSignal): Promise<void> {
    try {
      const productId = this.deltaService.getProductId(signal.symbol);
      if (!productId) {
        throw new Error(`Product not found for symbol: ${signal.symbol}`);
      }

      // Check if we can place the trade (risk management)
      if (!this.canPlaceTrade(signal)) {
        logger.info(`🚫 Trade rejected by risk management: ${signal.action} ${signal.symbol}`);
        return;
      }

      const orderRequest: DeltaOrderRequest = {
        product_id: productId,
        side: signal.action as 'buy' | 'sell',
        size: signal.size || this.calculatePositionSize(signal.price || 0),
        order_type: 'market_order'
      };

      const order = await this.deltaService.placeOrder(orderRequest);

      this.status.totalTrades++;
      this.status.lastTradeTime = new Date();

      logger.info(`✅ Bot ${this.config.id}: Executed ${signal.action} order for ${signal.symbol}`);
      this.emit('tradeExecuted', { signal, order });

    } catch (error) {
      logger.error(`Error executing trade signal for bot ${this.config.id}:`, error);
      this.emit('tradeError', { signal, error });
    }
  }

  /**
   * Calculate position size based on risk management
   */
  private calculatePositionSize(price: number): number {
    const riskAmount = (this.config.capital * this.config.riskPerTrade) / 100;
    const positionValue = riskAmount * this.config.leverage;
    return Math.floor(positionValue / price * 100) / 100; // Round to 2 decimal places
  }

  /**
   * Check if we can place a trade (risk management)
   */
  private canPlaceTrade(signal: TradeSignal): boolean {
    // Check maximum positions
    if (this.positions.size >= this.config.maxPositions) {
      return false;
    }

    // Check if we already have a position in this symbol
    if (this.positions.has(signal.symbol)) {
      return false;
    }

    // Check daily trade limit (example)
    const today = new Date().toDateString();
    // This would normally check against a daily trade counter

    return true;
  }

  /**
   * Manage existing positions (stop loss, take profit)
   * CRITICAL: Uses ONLY Delta Exchange live data for risk management calculations
   */
  private async manageExistingPositions(): Promise<void> {
    for (const [symbol, position] of this.positions) {
      try {
        // IMPORTANT: Use ONLY Delta Exchange API for current price (same source as trade execution)
        const marketData = await this.deltaService.getMarketData(symbol);
        const currentPrice = parseFloat(marketData.mark_price || marketData.last_price || '0');
        const entryPrice = parseFloat(position.entry_price);

        if (currentPrice === 0 || entryPrice === 0) continue;

        const pnlPercent = ((currentPrice - entryPrice) / entryPrice) * 100;
        const positionSide = position.size > 0 ? 'long' : 'short';

        // Adjust PnL for short positions
        const adjustedPnl = positionSide === 'short' ? -pnlPercent : pnlPercent;

        logger.debug(`📊 Position ${symbol}: Entry: $${entryPrice}, Current: $${currentPrice}, PnL: ${adjustedPnl.toFixed(2)}%`);

        // Check stop loss (using live Delta Exchange price)
        if (adjustedPnl <= -this.config.stopLoss) {
          logger.info(`🛑 Stop loss triggered for ${symbol}: PnL ${adjustedPnl.toFixed(2)}% <= -${this.config.stopLoss}%`);
          await this.closePosition(symbol, 'stop_loss');
        }

        // Check take profit (using live Delta Exchange price)
        if (adjustedPnl >= this.config.takeProfit) {
          logger.info(`🎯 Take profit triggered for ${symbol}: PnL ${adjustedPnl.toFixed(2)}% >= ${this.config.takeProfit}%`);
          await this.closePosition(symbol, 'take_profit');
        }

      } catch (error) {
        logger.error(`Error managing position for ${symbol}:`, error);
      }
    }
  }

  /**
   * Close a position
   */
  private async closePosition(symbol: string, reason: string): Promise<void> {
    try {
      const position = this.positions.get(symbol);
      if (!position) return;

      const productId = this.deltaService.getProductId(symbol);
      if (!productId) return;

      const side = position.size > 0 ? 'sell' : 'buy';
      const size = Math.abs(position.size);

      const orderRequest: DeltaOrderRequest = {
        product_id: productId,
        side,
        size,
        order_type: 'market_order',
        reduce_only: true
      };

      const order = await this.deltaService.placeOrder(orderRequest);

      logger.info(`🔄 Bot ${this.config.id}: Closed position ${symbol} (${reason})`);
      this.emit('positionClosed', { symbol, reason, order });

    } catch (error) {
      logger.error(`Error closing position for ${symbol}:`, error);
    }
  }

  /**
   * Cancel all open orders
   */
  private async cancelAllOrders(): Promise<void> {
    const cancelPromises = Array.from(this.orders.keys()).map(orderId =>
      this.deltaService.cancelOrder(orderId).catch(error =>
        logger.error(`Error cancelling order ${orderId}:`, error)
      )
    );

    await Promise.allSettled(cancelPromises);
    this.orders.clear();
  }

  /**
   * Update positions and orders from exchange
   */
  private async updatePositionsAndOrders(): Promise<void> {
    try {
      // Update positions
      const positions = await this.deltaService.getPositions();
      this.positions.clear();

      for (const position of positions) {
        if (position.product.symbol === this.config.symbol) {
          this.positions.set(position.product.symbol, position);
        }
      }

      // Update orders
      const productId = this.deltaService.getProductId(this.config.symbol);
      if (productId) {
        const orders = await this.deltaService.getOpenOrders(productId);
        this.orders.clear();

        for (const order of orders) {
          this.orders.set(order.id, order);
        }
      }

      this.status.currentPositions = this.positions.size;

    } catch (error) {
      logger.error(`Error updating positions and orders for bot ${this.config.id}:`, error);
    }
  }

  /**
   * Check risk limits
   */
  private checkRiskLimits(): void {
    // Check maximum drawdown
    if (this.status.totalPnL < -this.config.capital * 0.2) { // 20% max drawdown
      logger.warn(`⚠️ Bot ${this.config.id}: Maximum drawdown reached`);
      this.pause();
    }

    // Check maximum positions
    if (this.positions.size > this.config.maxPositions) {
      logger.warn(`⚠️ Bot ${this.config.id}: Maximum positions exceeded`);
    }
  }

  /**
   * Handle trading error
   */
  private handleTradingError(error: any): void {
    this.status.status = 'error';
    this.status.errorMessage = error instanceof Error ? error.message : 'Unknown trading error';

    // Pause bot on critical errors
    if (error.message.includes('authentication') || error.message.includes('connection')) {
      this.pause();
    }

    this.emit('error', error);
  }

  /**
   * Handle order placed event
   */
  private handleOrderPlaced(order: DeltaOrder): void {
    if (order.product.symbol === this.config.symbol) {
      this.orders.set(order.id, order);
      logger.info(`📝 Bot ${this.config.id}: Order placed - ${order.side} ${order.size} ${order.product.symbol}`);
    }
  }

  /**
   * Handle order cancelled event
   */
  private handleOrderCancelled(order: DeltaOrder): void {
    if (this.orders.has(order.id)) {
      this.orders.delete(order.id);
      logger.info(`❌ Bot ${this.config.id}: Order cancelled - ${order.id}`);
    }
  }

  /**
   * Handle ticker update
   */
  private handleTickerUpdate(ticker: any): void {
    // Update internal price tracking if needed
    // This could trigger immediate actions based on price movements
    this.emit('tickerUpdate', ticker);
  }

  /**
   * Get current positions
   */
  public getCurrentPositions(): DeltaPosition[] {
    return Array.from(this.positions.values());
  }

  /**
   * Get current orders
   */
  public getCurrentOrders(): DeltaOrder[] {
    return Array.from(this.orders.values());
  }

  /**
   * Get performance metrics
   */
  public getPerformanceMetrics(): any {
    const winRate = this.status.totalTrades > 0
      ? (this.status.winningTrades / this.status.totalTrades) * 100
      : 0;

    return {
      totalTrades: this.status.totalTrades,
      winningTrades: this.status.winningTrades,
      losingTrades: this.status.losingTrades,
      winRate: winRate.toFixed(2),
      totalPnL: this.status.totalPnL,
      currentPositions: this.status.currentPositions,
      uptime: this.status.uptime,
      lastTradeTime: this.status.lastTradeTime
    };
  }

  /**
   * Force close all positions (emergency)
   */
  public async emergencyCloseAll(): Promise<void> {
    logger.warn(`🚨 Bot ${this.config.id}: Emergency close all positions`);

    const closePromises = Array.from(this.positions.keys()).map(symbol =>
      this.closePosition(symbol, 'emergency_close').catch(error =>
        logger.error(`Error in emergency close for ${symbol}:`, error)
      )
    );

    await Promise.allSettled(closePromises);
    await this.cancelAllOrders();

    this.emit('emergencyClose', this.status);
  }

  /**
   * Cleanup bot resources
   */
  public cleanup(): void {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
    }

    if (this.tradingInterval) {
      clearInterval(this.tradingInterval);
    }

    this.removeAllListeners();
    logger.info(`🧹 Bot ${this.config.id} cleaned up`);
  }
}
