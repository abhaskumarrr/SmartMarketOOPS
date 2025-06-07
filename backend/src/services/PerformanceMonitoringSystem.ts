/**
 * Performance Monitoring System
 * Comprehensive metrics collection and monitoring for trading system performance
 * Built with Prometheus integration, real-time alerts, and advanced analytics
 */

import { register, collectDefaultMetrics, Counter, Histogram, Gauge, Summary } from 'prom-client';
import { EnhancedTradingDecisionEngine, TradingDecision } from './EnhancedTradingDecisionEngine';
import { MLPositionManager, Position } from './MLPositionManager';
import { EnhancedRiskManagementSystem } from './EnhancedRiskManagementSystem';
import { AnalysisExecutionBridge } from './AnalysisExecutionBridge';
import { DataCollectorIntegration } from './DataCollectorIntegration';
import { logger } from '../utils/logger';
import express, { Express, Request, Response } from 'express';
import { createServer, Server } from 'http';

// Performance monitoring types
export interface PerformanceMetrics {
  // Trading performance
  totalTrades: number;
  successfulTrades: number;
  failedTrades: number;
  winRate: number;
  totalPnL: number;
  averageTradeReturn: number;
  
  // System performance
  systemUptime: number;
  averageLatency: number;
  errorRate: number;
  throughput: number;
  
  // ML model performance
  modelAccuracy: number;
  modelPrecision: number;
  modelRecall: number;
  modelF1Score: number;
  
  // Risk metrics
  currentDrawdown: number;
  maxDrawdown: number;
  sharpeRatio: number;
  volatility: number;
}

export interface AlertRule {
  id: string;
  name: string;
  metric: string;
  condition: 'greater_than' | 'less_than' | 'equals' | 'not_equals';
  threshold: number;
  duration: number; // seconds
  severity: 'critical' | 'warning' | 'info';
  enabled: boolean;
  description: string;
}

export interface Alert {
  id: string;
  ruleId: string;
  name: string;
  severity: 'critical' | 'warning' | 'info';
  message: string;
  value: number;
  threshold: number;
  triggeredAt: number;
  resolvedAt?: number;
  status: 'firing' | 'resolved';
}

export interface MonitoringConfig {
  metricsPort: number;
  enableDefaultMetrics: boolean;
  scrapeInterval: number;
  alertCheckInterval: number;
  retentionPeriod: number; // days
  enableGrafanaIntegration: boolean;
}

export class PerformanceMonitoringSystem {
  private decisionEngine: EnhancedTradingDecisionEngine;
  private positionManager: MLPositionManager;
  private riskManager: EnhancedRiskManagementSystem;
  private bridge: AnalysisExecutionBridge;
  private dataIntegration: DataCollectorIntegration;
  
  // Express server for metrics endpoint
  private app: Express;
  private server: Server;
  
  // Prometheus metrics
  private tradingMetrics = {
    // Trading counters
    tradesTotal: new Counter({
      name: 'trading_trades_total',
      help: 'Total number of trades executed',
      labelNames: ['symbol', 'action', 'status']
    }),
    
    tradePnL: new Histogram({
      name: 'trading_trade_pnl',
      help: 'Profit/Loss per trade',
      labelNames: ['symbol', 'action'],
      buckets: [-1000, -500, -100, -50, -10, 0, 10, 50, 100, 500, 1000]
    }),
    
    tradeLatency: new Histogram({
      name: 'trading_trade_latency_seconds',
      help: 'Time from signal to execution',
      labelNames: ['symbol', 'action'],
      buckets: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]
    }),
    
    // ML model metrics
    modelAccuracy: new Gauge({
      name: 'ml_model_accuracy',
      help: 'Current ML model accuracy',
      labelNames: ['model_type', 'symbol']
    }),
    
    modelPredictions: new Counter({
      name: 'ml_model_predictions_total',
      help: 'Total ML model predictions',
      labelNames: ['model_type', 'symbol', 'prediction']
    }),
    
    modelLatency: new Histogram({
      name: 'ml_model_latency_seconds',
      help: 'ML model inference latency',
      labelNames: ['model_type', 'symbol'],
      buckets: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
    }),
    
    // Risk metrics
    portfolioValue: new Gauge({
      name: 'risk_portfolio_value',
      help: 'Current portfolio value'
    }),
    
    drawdown: new Gauge({
      name: 'risk_drawdown_percentage',
      help: 'Current portfolio drawdown percentage'
    }),
    
    riskScore: new Gauge({
      name: 'risk_overall_score',
      help: 'Overall risk score (0-1)',
      labelNames: ['risk_type']
    }),
    
    // System metrics
    systemErrors: new Counter({
      name: 'system_errors_total',
      help: 'Total system errors',
      labelNames: ['component', 'error_type']
    }),
    
    apiRequests: new Counter({
      name: 'api_requests_total',
      help: 'Total API requests',
      labelNames: ['method', 'endpoint', 'status']
    }),
    
    apiLatency: new Histogram({
      name: 'api_request_latency_seconds',
      help: 'API request latency',
      labelNames: ['method', 'endpoint'],
      buckets: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]
    }),
    
    // Data quality metrics
    dataQuality: new Gauge({
      name: 'data_quality_score',
      help: 'Data quality score (0-1)',
      labelNames: ['symbol', 'timeframe']
    }),
    
    dataLatency: new Histogram({
      name: 'data_collection_latency_seconds',
      help: 'Data collection latency',
      labelNames: ['symbol', 'timeframe'],
      buckets: [0.1, 0.5, 1, 2, 5, 10, 30]
    })
  };
  
  // Alert management
  private alertRules: Map<string, AlertRule> = new Map();
  private activeAlerts: Map<string, Alert> = new Map();
  private alertHistory: Alert[] = [];
  
  // Performance tracking
  private performanceHistory: PerformanceMetrics[] = [];
  private startTime: number = Date.now();
  
  // Configuration
  private config: MonitoringConfig = {
    metricsPort: 9090,
    enableDefaultMetrics: true,
    scrapeInterval: 15000, // 15 seconds
    alertCheckInterval: 30000, // 30 seconds
    retentionPeriod: 30, // 30 days
    enableGrafanaIntegration: true
  };

  constructor(customConfig?: Partial<MonitoringConfig>) {
    if (customConfig) {
      this.config = { ...this.config, ...customConfig };
    }

    // Initialize components
    this.decisionEngine = new EnhancedTradingDecisionEngine();
    this.positionManager = new MLPositionManager();
    this.riskManager = new EnhancedRiskManagementSystem();
    this.bridge = new AnalysisExecutionBridge();
    this.dataIntegration = new DataCollectorIntegration();
    
    // Initialize Express app for metrics
    this.app = express();
    this.server = createServer(this.app);
  }

  /**
   * Initialize the Performance Monitoring System
   */
  public async initialize(): Promise<void> {
    try {
      logger.info('📊 Initializing Performance Monitoring System...');
      
      // Initialize trading components
      await this.decisionEngine.initialize();
      await this.positionManager.initialize();
      await this.riskManager.initialize();
      await this.dataIntegration.initialize();
      
      // Setup Prometheus metrics
      this.setupPrometheusMetrics();
      
      // Setup metrics server
      this.setupMetricsServer();
      
      // Initialize default alert rules
      this.initializeDefaultAlertRules();
      
      // Start monitoring loops
      this.startMonitoringLoops();
      
      logger.info('✅ Performance Monitoring System initialized successfully');
      logger.info(`📈 Metrics endpoint: http://localhost:${this.config.metricsPort}/metrics`);
      logger.info(`🚨 Alert rules: ${this.alertRules.size} configured`);
      
    } catch (error: any) {
      logger.error('❌ Failed to initialize Performance Monitoring System:', error.message);
      throw error;
    }
  }

  /**
   * Start the monitoring system
   */
  public async start(): Promise<void> {
    try {
      logger.info('🚀 Starting Performance Monitoring System...');
      
      // Start metrics server
      await new Promise<void>((resolve, reject) => {
        this.server.listen(this.config.metricsPort, () => {
          logger.info(`📊 Metrics server started on port ${this.config.metricsPort}`);
          resolve();
        });
        
        this.server.on('error', (error) => {
          logger.error('❌ Metrics server error:', error);
          reject(error);
        });
      });
      
      logger.info('✅ Performance Monitoring System started successfully');
      
    } catch (error: any) {
      logger.error('❌ Failed to start Performance Monitoring System:', error.message);
      throw error;
    }
  }

  /**
   * Record trading decision metrics
   */
  public recordTradingDecision(decision: TradingDecision, latency: number): void {
    try {
      // Record model prediction
      this.tradingMetrics.modelPredictions.inc({
        model_type: 'ensemble',
        symbol: decision.symbol,
        prediction: decision.action
      });
      
      // Record model latency
      this.tradingMetrics.modelLatency.observe({
        model_type: 'ensemble',
        symbol: decision.symbol
      }, latency / 1000); // Convert to seconds
      
      // Record model accuracy (based on confidence)
      this.tradingMetrics.modelAccuracy.set({
        model_type: 'ensemble',
        symbol: decision.symbol
      }, decision.confidence);
      
      logger.debug(`📊 Recorded trading decision metrics: ${decision.symbol} ${decision.action} (${latency}ms)`);
      
    } catch (error: any) {
      logger.error('❌ Failed to record trading decision metrics:', error.message);
      this.recordError('performance_monitoring', 'metrics_recording_error');
    }
  }

  /**
   * Record trade execution metrics
   */
  public recordTradeExecution(
    symbol: string, 
    action: string, 
    status: 'success' | 'failed', 
    pnl?: number, 
    latency?: number
  ): void {
    try {
      // Record trade count
      this.tradingMetrics.tradesTotal.inc({
        symbol,
        action,
        status
      });
      
      // Record P&L if provided
      if (pnl !== undefined) {
        this.tradingMetrics.tradePnL.observe({
          symbol,
          action
        }, pnl);
      }
      
      // Record execution latency if provided
      if (latency !== undefined) {
        this.tradingMetrics.tradeLatency.observe({
          symbol,
          action
        }, latency / 1000); // Convert to seconds
      }
      
      logger.debug(`📊 Recorded trade execution: ${symbol} ${action} ${status} (PnL: ${pnl}, Latency: ${latency}ms)`);
      
    } catch (error: any) {
      logger.error('❌ Failed to record trade execution metrics:', error.message);
      this.recordError('performance_monitoring', 'metrics_recording_error');
    }
  }

  /**
   * Record API request metrics
   */
  public recordApiRequest(method: string, endpoint: string, status: number, latency: number): void {
    try {
      this.tradingMetrics.apiRequests.inc({
        method,
        endpoint,
        status: status.toString()
      });
      
      this.tradingMetrics.apiLatency.observe({
        method,
        endpoint
      }, latency / 1000); // Convert to seconds
      
    } catch (error: any) {
      logger.error('❌ Failed to record API request metrics:', error.message);
    }
  }

  /**
   * Record system error
   */
  public recordError(component: string, errorType: string): void {
    try {
      this.tradingMetrics.systemErrors.inc({
        component,
        error_type: errorType
      });
      
    } catch (error: any) {
      logger.error('❌ Failed to record system error metric:', error.message);
    }
  }

  /**
   * Update risk metrics
   */
  public updateRiskMetrics(): void {
    try {
      const riskMetrics = this.riskManager.getRiskMetrics();
      const performanceMetrics = this.positionManager.getPerformanceMetrics();
      
      // Update portfolio value
      this.tradingMetrics.portfolioValue.set(parseFloat(performanceMetrics.totalPnL) + 10000); // Base + PnL
      
      // Update drawdown
      this.tradingMetrics.drawdown.set(Math.abs(riskMetrics.currentDrawdown) * 100);
      
      // Update risk scores
      this.tradingMetrics.riskScore.set({ risk_type: 'overall' }, riskMetrics.overallRiskScore);
      this.tradingMetrics.riskScore.set({ risk_type: 'market_regime' }, riskMetrics.marketRegimeRisk);
      this.tradingMetrics.riskScore.set({ risk_type: 'concentration' }, riskMetrics.concentrationRisk);
      
    } catch (error: any) {
      logger.error('❌ Failed to update risk metrics:', error.message);
      this.recordError('performance_monitoring', 'risk_metrics_update_error');
    }
  }

  /**
   * Update data quality metrics
   */
  public updateDataQualityMetrics(symbol: string, timeframe: string, quality: number, latency: number): void {
    try {
      this.tradingMetrics.dataQuality.set({
        symbol,
        timeframe
      }, quality);
      
      this.tradingMetrics.dataLatency.observe({
        symbol,
        timeframe
      }, latency / 1000); // Convert to seconds
      
    } catch (error: any) {
      logger.error('❌ Failed to update data quality metrics:', error.message);
      this.recordError('performance_monitoring', 'data_quality_update_error');
    }
  }

  /**
   * Get current performance metrics
   */
  public getPerformanceMetrics(): PerformanceMetrics {
    try {
      const riskMetrics = this.riskManager.getRiskMetrics();
      const performanceMetrics = this.positionManager.getPerformanceMetrics();
      const bridgeStatus = this.bridge.getStatus();
      
      return {
        // Trading performance
        totalTrades: parseInt(performanceMetrics.totalTrades),
        successfulTrades: parseInt(performanceMetrics.winningTrades),
        failedTrades: parseInt(performanceMetrics.losingTrades),
        winRate: parseFloat(performanceMetrics.winRate) / 100,
        totalPnL: parseFloat(performanceMetrics.totalPnL),
        averageTradeReturn: parseFloat(performanceMetrics.averageReturn),
        
        // System performance
        systemUptime: Date.now() - this.startTime,
        averageLatency: bridgeStatus.averageLatency,
        errorRate: this.calculateErrorRate(),
        throughput: this.calculateThroughput(),
        
        // ML model performance (simplified)
        modelAccuracy: 0.85, // Would be calculated from actual model performance
        modelPrecision: 0.82,
        modelRecall: 0.88,
        modelF1Score: 0.85,
        
        // Risk metrics
        currentDrawdown: Math.abs(riskMetrics.currentDrawdown),
        maxDrawdown: Math.abs(riskMetrics.maxDrawdown),
        sharpeRatio: riskMetrics.sharpeRatio,
        volatility: riskMetrics.volatilityIndex
      };
      
    } catch (error: any) {
      logger.error('❌ Failed to get performance metrics:', error.message);
      this.recordError('performance_monitoring', 'metrics_calculation_error');
      
      // Return default metrics
      return {
        totalTrades: 0, successfulTrades: 0, failedTrades: 0, winRate: 0,
        totalPnL: 0, averageTradeReturn: 0, systemUptime: 0, averageLatency: 0,
        errorRate: 0, throughput: 0, modelAccuracy: 0, modelPrecision: 0,
        modelRecall: 0, modelF1Score: 0, currentDrawdown: 0, maxDrawdown: 0,
        sharpeRatio: 0, volatility: 0
      };
    }
  }

  // Private methods for monitoring system

  /**
   * Setup Prometheus metrics collection
   */
  private setupPrometheusMetrics(): void {
    try {
      // Clear existing metrics
      register.clear();

      // Enable default metrics if configured
      if (this.config.enableDefaultMetrics) {
        collectDefaultMetrics({
          register,
          prefix: 'trading_system_',
          gcDurationBuckets: [0.001, 0.01, 0.1, 1, 2, 5],
          eventLoopMonitoringPrecision: 5
        });
      }

      // Register custom metrics
      register.registerMetric(this.tradingMetrics.tradesTotal);
      register.registerMetric(this.tradingMetrics.tradePnL);
      register.registerMetric(this.tradingMetrics.tradeLatency);
      register.registerMetric(this.tradingMetrics.modelAccuracy);
      register.registerMetric(this.tradingMetrics.modelPredictions);
      register.registerMetric(this.tradingMetrics.modelLatency);
      register.registerMetric(this.tradingMetrics.portfolioValue);
      register.registerMetric(this.tradingMetrics.drawdown);
      register.registerMetric(this.tradingMetrics.riskScore);
      register.registerMetric(this.tradingMetrics.systemErrors);
      register.registerMetric(this.tradingMetrics.apiRequests);
      register.registerMetric(this.tradingMetrics.apiLatency);
      register.registerMetric(this.tradingMetrics.dataQuality);
      register.registerMetric(this.tradingMetrics.dataLatency);

      logger.info('📊 Prometheus metrics configured successfully');

    } catch (error: any) {
      logger.error('❌ Failed to setup Prometheus metrics:', error.message);
      throw error;
    }
  }

  /**
   * Setup metrics server endpoints
   */
  private setupMetricsServer(): void {
    try {
      // Prometheus metrics endpoint
      this.app.get('/metrics', async (req: Request, res: Response) => {
        try {
          res.set('Content-Type', register.contentType);
          const metrics = await register.metrics();
          res.end(metrics);
        } catch (error: any) {
          logger.error('❌ Failed to generate metrics:', error.message);
          res.status(500).end('Error generating metrics');
        }
      });

      // Health check endpoint
      this.app.get('/health', (req: Request, res: Response) => {
        const uptime = Date.now() - this.startTime;
        res.json({
          status: 'healthy',
          uptime: uptime,
          timestamp: Date.now(),
          version: '1.0.0'
        });
      });

      // Performance metrics endpoint
      this.app.get('/performance', (req: Request, res: Response) => {
        try {
          const metrics = this.getPerformanceMetrics();
          res.json(metrics);
        } catch (error: any) {
          logger.error('❌ Failed to get performance metrics:', error.message);
          res.status(500).json({ error: 'Failed to get performance metrics' });
        }
      });

      // Alerts endpoint
      this.app.get('/alerts', (req: Request, res: Response) => {
        try {
          const alerts = Array.from(this.activeAlerts.values());
          res.json({
            active: alerts.filter(alert => alert.status === 'firing'),
            resolved: alerts.filter(alert => alert.status === 'resolved'),
            total: alerts.length
          });
        } catch (error: any) {
          logger.error('❌ Failed to get alerts:', error.message);
          res.status(500).json({ error: 'Failed to get alerts' });
        }
      });

      // Alert rules endpoint
      this.app.get('/alert-rules', (req: Request, res: Response) => {
        try {
          const rules = Array.from(this.alertRules.values());
          res.json(rules);
        } catch (error: any) {
          logger.error('❌ Failed to get alert rules:', error.message);
          res.status(500).json({ error: 'Failed to get alert rules' });
        }
      });

      logger.info('📊 Metrics server endpoints configured');

    } catch (error: any) {
      logger.error('❌ Failed to setup metrics server:', error.message);
      throw error;
    }
  }

  /**
   * Initialize default alert rules
   */
  private initializeDefaultAlertRules(): void {
    try {
      const defaultRules: Omit<AlertRule, 'id'>[] = [
        {
          name: 'High Error Rate',
          metric: 'system_errors_total',
          condition: 'greater_than',
          threshold: 10,
          duration: 300, // 5 minutes
          severity: 'critical',
          enabled: true,
          description: 'System error rate is too high'
        },
        {
          name: 'Low Win Rate',
          metric: 'trading_win_rate',
          condition: 'less_than',
          threshold: 0.4, // 40%
          duration: 1800, // 30 minutes
          severity: 'warning',
          enabled: true,
          description: 'Trading win rate is below acceptable threshold'
        },
        {
          name: 'High Drawdown',
          metric: 'risk_drawdown_percentage',
          condition: 'greater_than',
          threshold: 15, // 15%
          duration: 60, // 1 minute
          severity: 'critical',
          enabled: true,
          description: 'Portfolio drawdown exceeds risk limits'
        },
        {
          name: 'High API Latency',
          metric: 'api_request_latency_seconds',
          condition: 'greater_than',
          threshold: 1.0, // 1 second
          duration: 300, // 5 minutes
          severity: 'warning',
          enabled: true,
          description: 'API response times are too high'
        },
        {
          name: 'Low Data Quality',
          metric: 'data_quality_score',
          condition: 'less_than',
          threshold: 0.8, // 80%
          duration: 600, // 10 minutes
          severity: 'warning',
          enabled: true,
          description: 'Data quality is below acceptable threshold'
        },
        {
          name: 'Model Accuracy Drop',
          metric: 'ml_model_accuracy',
          condition: 'less_than',
          threshold: 0.7, // 70%
          duration: 900, // 15 minutes
          severity: 'warning',
          enabled: true,
          description: 'ML model accuracy has dropped significantly'
        }
      ];

      for (const rule of defaultRules) {
        const alertRule: AlertRule = {
          ...rule,
          id: `alert_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
        };
        this.alertRules.set(alertRule.id, alertRule);
      }

      logger.info(`🚨 Initialized ${this.alertRules.size} default alert rules`);

    } catch (error: any) {
      logger.error('❌ Failed to initialize default alert rules:', error.message);
    }
  }

  /**
   * Start monitoring loops
   */
  private startMonitoringLoops(): void {
    try {
      // Performance metrics update loop
      setInterval(() => {
        this.updateRiskMetrics();
        this.updatePerformanceHistory();
      }, this.config.scrapeInterval);

      // Alert checking loop
      setInterval(() => {
        this.checkAlerts();
      }, this.config.alertCheckInterval);

      // Cleanup old data loop (daily)
      setInterval(() => {
        this.cleanupOldData();
      }, 24 * 60 * 60 * 1000);

      logger.info('⏰ Monitoring loops started');
      logger.info(`   Performance updates: every ${this.config.scrapeInterval / 1000}s`);
      logger.info(`   Alert checks: every ${this.config.alertCheckInterval / 1000}s`);

    } catch (error: any) {
      logger.error('❌ Failed to start monitoring loops:', error.message);
    }
  }

  /**
   * Update performance history
   */
  private updatePerformanceHistory(): void {
    try {
      const metrics = this.getPerformanceMetrics();
      this.performanceHistory.push(metrics);

      // Keep only last 1000 entries
      if (this.performanceHistory.length > 1000) {
        this.performanceHistory = this.performanceHistory.slice(-1000);
      }

    } catch (error: any) {
      logger.error('❌ Failed to update performance history:', error.message);
    }
  }

  /**
   * Check alert rules and trigger alerts
   */
  private checkAlerts(): void {
    try {
      for (const rule of this.alertRules.values()) {
        if (!rule.enabled) continue;

        this.evaluateAlertRule(rule);
      }

    } catch (error: any) {
      logger.error('❌ Failed to check alerts:', error.message);
      this.recordError('performance_monitoring', 'alert_check_error');
    }
  }

  /**
   * Evaluate individual alert rule
   */
  private async evaluateAlertRule(rule: AlertRule): Promise<void> {
    try {
      const currentValue = await this.getMetricValue(rule.metric);
      const shouldFire = this.evaluateCondition(currentValue, rule.condition, rule.threshold);

      const existingAlert = Array.from(this.activeAlerts.values())
        .find(alert => alert.ruleId === rule.id && alert.status === 'firing');

      if (shouldFire && !existingAlert) {
        // Create new alert
        const alert: Alert = {
          id: `alert_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          ruleId: rule.id,
          name: rule.name,
          severity: rule.severity,
          message: `${rule.description}. Current value: ${currentValue.toFixed(4)}, Threshold: ${rule.threshold}`,
          value: currentValue,
          threshold: rule.threshold,
          triggeredAt: Date.now(),
          status: 'firing'
        };

        this.activeAlerts.set(alert.id, alert);
        this.alertHistory.push(alert);

        logger.warn(`🚨 Alert triggered: ${alert.name} (${alert.severity})`);
        logger.warn(`   Message: ${alert.message}`);

      } else if (!shouldFire && existingAlert) {
        // Resolve existing alert
        existingAlert.status = 'resolved';
        existingAlert.resolvedAt = Date.now();

        logger.info(`✅ Alert resolved: ${existingAlert.name}`);
      }

    } catch (error: any) {
      logger.error(`❌ Failed to evaluate alert rule ${rule.name}:`, error.message);
    }
  }

  /**
   * Get current value for a metric
   */
  private async getMetricValue(metricName: string): Promise<number> {
    try {
      const metrics = await register.getSingleMetricAsString(metricName);

      // Parse the metric value (simplified)
      const lines = metrics.split('\n');
      for (const line of lines) {
        if (line.startsWith(metricName) && !line.startsWith('#')) {
          const parts = line.split(' ');
          if (parts.length >= 2) {
            return parseFloat(parts[1]);
          }
        }
      }

      // Fallback to performance metrics
      const performanceMetrics = this.getPerformanceMetrics();
      switch (metricName) {
        case 'trading_win_rate':
          return performanceMetrics.winRate;
        case 'risk_drawdown_percentage':
          return performanceMetrics.currentDrawdown * 100;
        case 'ml_model_accuracy':
          return performanceMetrics.modelAccuracy;
        default:
          return 0;
      }

    } catch (error: any) {
      logger.error(`❌ Failed to get metric value for ${metricName}:`, error.message);
      return 0;
    }
  }

  /**
   * Evaluate alert condition
   */
  private evaluateCondition(value: number, condition: string, threshold: number): boolean {
    switch (condition) {
      case 'greater_than':
        return value > threshold;
      case 'less_than':
        return value < threshold;
      case 'equals':
        return Math.abs(value - threshold) < 0.001;
      case 'not_equals':
        return Math.abs(value - threshold) >= 0.001;
      default:
        return false;
    }
  }

  /**
   * Calculate system error rate
   */
  private calculateErrorRate(): number {
    try {
      // Simplified error rate calculation
      const totalRequests = this.bridge.getStatus().totalSignals;
      const failedRequests = this.bridge.getStatus().failedExecutions;

      if (totalRequests === 0) return 0;
      return failedRequests / totalRequests;

    } catch (error: any) {
      return 0;
    }
  }

  /**
   * Calculate system throughput
   */
  private calculateThroughput(): number {
    try {
      const uptime = Date.now() - this.startTime;
      const totalSignals = this.bridge.getStatus().totalSignals;

      if (uptime === 0) return 0;
      return (totalSignals / uptime) * 1000; // signals per second

    } catch (error: any) {
      return 0;
    }
  }

  /**
   * Cleanup old data
   */
  private cleanupOldData(): void {
    try {
      const cutoffTime = Date.now() - (this.config.retentionPeriod * 24 * 60 * 60 * 1000);

      // Cleanup old alerts
      this.alertHistory = this.alertHistory.filter(alert => alert.triggeredAt > cutoffTime);

      // Cleanup resolved alerts from active alerts
      for (const [id, alert] of this.activeAlerts) {
        if (alert.status === 'resolved' && alert.resolvedAt && alert.resolvedAt < cutoffTime) {
          this.activeAlerts.delete(id);
        }
      }

      logger.info('🧹 Cleaned up old monitoring data');

    } catch (error: any) {
      logger.error('❌ Failed to cleanup old data:', error.message);
    }
  }

  /**
   * Add custom alert rule
   */
  public addAlertRule(rule: Omit<AlertRule, 'id'>): string {
    try {
      const alertRule: AlertRule = {
        ...rule,
        id: `alert_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
      };

      this.alertRules.set(alertRule.id, alertRule);

      logger.info(`🚨 Added new alert rule: ${alertRule.name}`);
      return alertRule.id;

    } catch (error: any) {
      logger.error('❌ Failed to add alert rule:', error.message);
      throw error;
    }
  }

  /**
   * Remove alert rule
   */
  public removeAlertRule(ruleId: string): boolean {
    try {
      const removed = this.alertRules.delete(ruleId);

      if (removed) {
        logger.info(`🗑️ Removed alert rule: ${ruleId}`);
      }

      return removed;

    } catch (error: any) {
      logger.error('❌ Failed to remove alert rule:', error.message);
      return false;
    }
  }

  /**
   * Get active alerts
   */
  public getActiveAlerts(): Alert[] {
    return Array.from(this.activeAlerts.values()).filter(alert => alert.status === 'firing');
  }

  /**
   * Get alert history
   */
  public getAlertHistory(limit: number = 100): Alert[] {
    return this.alertHistory.slice(-limit);
  }

  /**
   * Get performance history
   */
  public getPerformanceHistory(limit: number = 100): PerformanceMetrics[] {
    return this.performanceHistory.slice(-limit);
  }

  /**
   * Stop the monitoring system
   */
  public async stop(): Promise<void> {
    try {
      logger.info('🛑 Stopping Performance Monitoring System...');

      // Close metrics server
      await new Promise<void>((resolve) => {
        this.server.close(() => {
          logger.info('📊 Metrics server stopped');
          resolve();
        });
      });

      // Clear metrics registry
      register.clear();

      logger.info('✅ Performance Monitoring System stopped');

    } catch (error: any) {
      logger.error('❌ Error stopping Performance Monitoring System:', error.message);
    }
  }

  /**
   * Cleanup resources
   */
  public async cleanup(): Promise<void> {
    try {
      logger.info('🧹 Cleaning up Performance Monitoring System...');

      await this.stop();

      // Cleanup trading components
      await this.positionManager.cleanup();
      await this.riskManager.cleanup();
      await this.dataIntegration.cleanup();

      logger.info('✅ Performance Monitoring System cleanup completed');
    } catch (error: any) {
      logger.error('❌ Error during Performance Monitoring System cleanup:', error.message);
    }
  }
}
