"use strict";
/**
 * Performance Monitoring System
 * Comprehensive metrics collection and monitoring for trading system performance
 * Built with Prometheus integration, real-time alerts, and advanced analytics
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.PerformanceMonitoringSystem = void 0;
const prom_client_1 = require("prom-client");
const EnhancedTradingDecisionEngine_1 = require("./EnhancedTradingDecisionEngine");
const MLPositionManager_1 = require("./MLPositionManager");
const EnhancedRiskManagementSystem_1 = require("./EnhancedRiskManagementSystem");
const AnalysisExecutionBridge_1 = require("./AnalysisExecutionBridge");
const DataCollectorIntegration_1 = require("./DataCollectorIntegration");
const logger_1 = require("../utils/logger");
const express_1 = __importDefault(require("express"));
const http_1 = require("http");
class PerformanceMonitoringSystem {
    constructor(customConfig) {
        // Prometheus metrics
        this.tradingMetrics = {
            // Trading counters
            tradesTotal: new prom_client_1.Counter({
                name: 'trading_trades_total',
                help: 'Total number of trades executed',
                labelNames: ['symbol', 'action', 'status']
            }),
            tradePnL: new prom_client_1.Histogram({
                name: 'trading_trade_pnl',
                help: 'Profit/Loss per trade',
                labelNames: ['symbol', 'action'],
                buckets: [-1000, -500, -100, -50, -10, 0, 10, 50, 100, 500, 1000]
            }),
            tradeLatency: new prom_client_1.Histogram({
                name: 'trading_trade_latency_seconds',
                help: 'Time from signal to execution',
                labelNames: ['symbol', 'action'],
                buckets: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]
            }),
            // ML model metrics
            modelAccuracy: new prom_client_1.Gauge({
                name: 'ml_model_accuracy',
                help: 'Current ML model accuracy',
                labelNames: ['model_type', 'symbol']
            }),
            modelPredictions: new prom_client_1.Counter({
                name: 'ml_model_predictions_total',
                help: 'Total ML model predictions',
                labelNames: ['model_type', 'symbol', 'prediction']
            }),
            modelLatency: new prom_client_1.Histogram({
                name: 'ml_model_latency_seconds',
                help: 'ML model inference latency',
                labelNames: ['model_type', 'symbol'],
                buckets: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
            }),
            // Risk metrics
            portfolioValue: new prom_client_1.Gauge({
                name: 'risk_portfolio_value',
                help: 'Current portfolio value'
            }),
            drawdown: new prom_client_1.Gauge({
                name: 'risk_drawdown_percentage',
                help: 'Current portfolio drawdown percentage'
            }),
            riskScore: new prom_client_1.Gauge({
                name: 'risk_overall_score',
                help: 'Overall risk score (0-1)',
                labelNames: ['risk_type']
            }),
            // System metrics
            systemErrors: new prom_client_1.Counter({
                name: 'system_errors_total',
                help: 'Total system errors',
                labelNames: ['component', 'error_type']
            }),
            apiRequests: new prom_client_1.Counter({
                name: 'api_requests_total',
                help: 'Total API requests',
                labelNames: ['method', 'endpoint', 'status']
            }),
            apiLatency: new prom_client_1.Histogram({
                name: 'api_request_latency_seconds',
                help: 'API request latency',
                labelNames: ['method', 'endpoint'],
                buckets: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]
            }),
            // Data quality metrics
            dataQuality: new prom_client_1.Gauge({
                name: 'data_quality_score',
                help: 'Data quality score (0-1)',
                labelNames: ['symbol', 'timeframe']
            }),
            dataLatency: new prom_client_1.Histogram({
                name: 'data_collection_latency_seconds',
                help: 'Data collection latency',
                labelNames: ['symbol', 'timeframe'],
                buckets: [0.1, 0.5, 1, 2, 5, 10, 30]
            })
        };
        // Alert management
        this.alertRules = new Map();
        this.activeAlerts = new Map();
        this.alertHistory = [];
        // Performance tracking
        this.performanceHistory = [];
        this.startTime = Date.now();
        // Configuration
        this.config = {
            metricsPort: 9090,
            enableDefaultMetrics: true,
            scrapeInterval: 15000, // 15 seconds
            alertCheckInterval: 30000, // 30 seconds
            retentionPeriod: 30, // 30 days
            enableGrafanaIntegration: true
        };
        if (customConfig) {
            this.config = { ...this.config, ...customConfig };
        }
        // Initialize components
        this.decisionEngine = new EnhancedTradingDecisionEngine_1.EnhancedTradingDecisionEngine();
        this.positionManager = new MLPositionManager_1.MLPositionManager();
        this.riskManager = new EnhancedRiskManagementSystem_1.EnhancedRiskManagementSystem();
        this.bridge = new AnalysisExecutionBridge_1.AnalysisExecutionBridge();
        this.dataIntegration = new DataCollectorIntegration_1.DataCollectorIntegration();
        // Initialize Express app for metrics
        this.app = (0, express_1.default)();
        this.server = (0, http_1.createServer)(this.app);
    }
    /**
     * Initialize the Performance Monitoring System
     */
    async initialize() {
        try {
            logger_1.logger.info('📊 Initializing Performance Monitoring System...');
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
            logger_1.logger.info('✅ Performance Monitoring System initialized successfully');
            logger_1.logger.info(`📈 Metrics endpoint: http://localhost:${this.config.metricsPort}/metrics`);
            logger_1.logger.info(`🚨 Alert rules: ${this.alertRules.size} configured`);
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to initialize Performance Monitoring System:', error.message);
            throw error;
        }
    }
    /**
     * Start the monitoring system
     */
    async start() {
        try {
            logger_1.logger.info('🚀 Starting Performance Monitoring System...');
            // Start metrics server
            await new Promise((resolve, reject) => {
                this.server.listen(this.config.metricsPort, () => {
                    logger_1.logger.info(`📊 Metrics server started on port ${this.config.metricsPort}`);
                    resolve();
                });
                this.server.on('error', (error) => {
                    logger_1.logger.error('❌ Metrics server error:', error);
                    reject(error);
                });
            });
            logger_1.logger.info('✅ Performance Monitoring System started successfully');
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to start Performance Monitoring System:', error.message);
            throw error;
        }
    }
    /**
     * Record trading decision metrics
     */
    recordTradingDecision(decision, latency) {
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
            logger_1.logger.debug(`📊 Recorded trading decision metrics: ${decision.symbol} ${decision.action} (${latency}ms)`);
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to record trading decision metrics:', error.message);
            this.recordError('performance_monitoring', 'metrics_recording_error');
        }
    }
    /**
     * Record trade execution metrics
     */
    recordTradeExecution(symbol, action, status, pnl, latency) {
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
            logger_1.logger.debug(`📊 Recorded trade execution: ${symbol} ${action} ${status} (PnL: ${pnl}, Latency: ${latency}ms)`);
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to record trade execution metrics:', error.message);
            this.recordError('performance_monitoring', 'metrics_recording_error');
        }
    }
    /**
     * Record API request metrics
     */
    recordApiRequest(method, endpoint, status, latency) {
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
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to record API request metrics:', error.message);
        }
    }
    /**
     * Record system error
     */
    recordError(component, errorType) {
        try {
            this.tradingMetrics.systemErrors.inc({
                component,
                error_type: errorType
            });
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to record system error metric:', error.message);
        }
    }
    /**
     * Update risk metrics
     */
    updateRiskMetrics() {
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
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to update risk metrics:', error.message);
            this.recordError('performance_monitoring', 'risk_metrics_update_error');
        }
    }
    /**
     * Update data quality metrics
     */
    updateDataQualityMetrics(symbol, timeframe, quality, latency) {
        try {
            this.tradingMetrics.dataQuality.set({
                symbol,
                timeframe
            }, quality);
            this.tradingMetrics.dataLatency.observe({
                symbol,
                timeframe
            }, latency / 1000); // Convert to seconds
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to update data quality metrics:', error.message);
            this.recordError('performance_monitoring', 'data_quality_update_error');
        }
    }
    /**
     * Get current performance metrics
     */
    getPerformanceMetrics() {
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
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to get performance metrics:', error.message);
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
    setupPrometheusMetrics() {
        try {
            // Clear existing metrics
            prom_client_1.register.clear();
            // Enable default metrics if configured
            if (this.config.enableDefaultMetrics) {
                (0, prom_client_1.collectDefaultMetrics)({
                    register: prom_client_1.register,
                    prefix: 'trading_system_',
                    gcDurationBuckets: [0.001, 0.01, 0.1, 1, 2, 5],
                    eventLoopMonitoringPrecision: 5
                });
            }
            // Register custom metrics
            prom_client_1.register.registerMetric(this.tradingMetrics.tradesTotal);
            prom_client_1.register.registerMetric(this.tradingMetrics.tradePnL);
            prom_client_1.register.registerMetric(this.tradingMetrics.tradeLatency);
            prom_client_1.register.registerMetric(this.tradingMetrics.modelAccuracy);
            prom_client_1.register.registerMetric(this.tradingMetrics.modelPredictions);
            prom_client_1.register.registerMetric(this.tradingMetrics.modelLatency);
            prom_client_1.register.registerMetric(this.tradingMetrics.portfolioValue);
            prom_client_1.register.registerMetric(this.tradingMetrics.drawdown);
            prom_client_1.register.registerMetric(this.tradingMetrics.riskScore);
            prom_client_1.register.registerMetric(this.tradingMetrics.systemErrors);
            prom_client_1.register.registerMetric(this.tradingMetrics.apiRequests);
            prom_client_1.register.registerMetric(this.tradingMetrics.apiLatency);
            prom_client_1.register.registerMetric(this.tradingMetrics.dataQuality);
            prom_client_1.register.registerMetric(this.tradingMetrics.dataLatency);
            logger_1.logger.info('📊 Prometheus metrics configured successfully');
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to setup Prometheus metrics:', error.message);
            throw error;
        }
    }
    /**
     * Setup metrics server endpoints
     */
    setupMetricsServer() {
        try {
            // Prometheus metrics endpoint
            this.app.get('/metrics', async (req, res) => {
                try {
                    res.set('Content-Type', prom_client_1.register.contentType);
                    const metrics = await prom_client_1.register.metrics();
                    res.end(metrics);
                }
                catch (error) {
                    logger_1.logger.error('❌ Failed to generate metrics:', error.message);
                    res.status(500).end('Error generating metrics');
                }
            });
            // Health check endpoint
            this.app.get('/health', (req, res) => {
                const uptime = Date.now() - this.startTime;
                res.json({
                    status: 'healthy',
                    uptime: uptime,
                    timestamp: Date.now(),
                    version: '1.0.0'
                });
            });
            // Performance metrics endpoint
            this.app.get('/performance', (req, res) => {
                try {
                    const metrics = this.getPerformanceMetrics();
                    res.json(metrics);
                }
                catch (error) {
                    logger_1.logger.error('❌ Failed to get performance metrics:', error.message);
                    res.status(500).json({ error: 'Failed to get performance metrics' });
                }
            });
            // Alerts endpoint
            this.app.get('/alerts', (req, res) => {
                try {
                    const alerts = Array.from(this.activeAlerts.values());
                    res.json({
                        active: alerts.filter(alert => alert.status === 'firing'),
                        resolved: alerts.filter(alert => alert.status === 'resolved'),
                        total: alerts.length
                    });
                }
                catch (error) {
                    logger_1.logger.error('❌ Failed to get alerts:', error.message);
                    res.status(500).json({ error: 'Failed to get alerts' });
                }
            });
            // Alert rules endpoint
            this.app.get('/alert-rules', (req, res) => {
                try {
                    const rules = Array.from(this.alertRules.values());
                    res.json(rules);
                }
                catch (error) {
                    logger_1.logger.error('❌ Failed to get alert rules:', error.message);
                    res.status(500).json({ error: 'Failed to get alert rules' });
                }
            });
            logger_1.logger.info('📊 Metrics server endpoints configured');
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to setup metrics server:', error.message);
            throw error;
        }
    }
    /**
     * Initialize default alert rules
     */
    initializeDefaultAlertRules() {
        try {
            const defaultRules = [
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
                const alertRule = {
                    ...rule,
                    id: `alert_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
                };
                this.alertRules.set(alertRule.id, alertRule);
            }
            logger_1.logger.info(`🚨 Initialized ${this.alertRules.size} default alert rules`);
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to initialize default alert rules:', error.message);
        }
    }
    /**
     * Start monitoring loops
     */
    startMonitoringLoops() {
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
            logger_1.logger.info('⏰ Monitoring loops started');
            logger_1.logger.info(`   Performance updates: every ${this.config.scrapeInterval / 1000}s`);
            logger_1.logger.info(`   Alert checks: every ${this.config.alertCheckInterval / 1000}s`);
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to start monitoring loops:', error.message);
        }
    }
    /**
     * Update performance history
     */
    updatePerformanceHistory() {
        try {
            const metrics = this.getPerformanceMetrics();
            this.performanceHistory.push(metrics);
            // Keep only last 1000 entries
            if (this.performanceHistory.length > 1000) {
                this.performanceHistory = this.performanceHistory.slice(-1000);
            }
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to update performance history:', error.message);
        }
    }
    /**
     * Check alert rules and trigger alerts
     */
    checkAlerts() {
        try {
            for (const rule of this.alertRules.values()) {
                if (!rule.enabled)
                    continue;
                this.evaluateAlertRule(rule);
            }
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to check alerts:', error.message);
            this.recordError('performance_monitoring', 'alert_check_error');
        }
    }
    /**
     * Evaluate individual alert rule
     */
    async evaluateAlertRule(rule) {
        try {
            const currentValue = await this.getMetricValue(rule.metric);
            const shouldFire = this.evaluateCondition(currentValue, rule.condition, rule.threshold);
            const existingAlert = Array.from(this.activeAlerts.values())
                .find(alert => alert.ruleId === rule.id && alert.status === 'firing');
            if (shouldFire && !existingAlert) {
                // Create new alert
                const alert = {
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
                logger_1.logger.warn(`🚨 Alert triggered: ${alert.name} (${alert.severity})`);
                logger_1.logger.warn(`   Message: ${alert.message}`);
            }
            else if (!shouldFire && existingAlert) {
                // Resolve existing alert
                existingAlert.status = 'resolved';
                existingAlert.resolvedAt = Date.now();
                logger_1.logger.info(`✅ Alert resolved: ${existingAlert.name}`);
            }
        }
        catch (error) {
            logger_1.logger.error(`❌ Failed to evaluate alert rule ${rule.name}:`, error.message);
        }
    }
    /**
     * Get current value for a metric
     */
    async getMetricValue(metricName) {
        try {
            const metrics = await prom_client_1.register.getSingleMetricAsString(metricName);
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
        }
        catch (error) {
            logger_1.logger.error(`❌ Failed to get metric value for ${metricName}:`, error.message);
            return 0;
        }
    }
    /**
     * Evaluate alert condition
     */
    evaluateCondition(value, condition, threshold) {
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
    calculateErrorRate() {
        try {
            // Simplified error rate calculation
            const totalRequests = this.bridge.getStatus().totalSignals;
            const failedRequests = this.bridge.getStatus().failedExecutions;
            if (totalRequests === 0)
                return 0;
            return failedRequests / totalRequests;
        }
        catch (error) {
            return 0;
        }
    }
    /**
     * Calculate system throughput
     */
    calculateThroughput() {
        try {
            const uptime = Date.now() - this.startTime;
            const totalSignals = this.bridge.getStatus().totalSignals;
            if (uptime === 0)
                return 0;
            return (totalSignals / uptime) * 1000; // signals per second
        }
        catch (error) {
            return 0;
        }
    }
    /**
     * Cleanup old data
     */
    cleanupOldData() {
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
            logger_1.logger.info('🧹 Cleaned up old monitoring data');
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to cleanup old data:', error.message);
        }
    }
    /**
     * Add custom alert rule
     */
    addAlertRule(rule) {
        try {
            const alertRule = {
                ...rule,
                id: `alert_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
            };
            this.alertRules.set(alertRule.id, alertRule);
            logger_1.logger.info(`🚨 Added new alert rule: ${alertRule.name}`);
            return alertRule.id;
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to add alert rule:', error.message);
            throw error;
        }
    }
    /**
     * Remove alert rule
     */
    removeAlertRule(ruleId) {
        try {
            const removed = this.alertRules.delete(ruleId);
            if (removed) {
                logger_1.logger.info(`🗑️ Removed alert rule: ${ruleId}`);
            }
            return removed;
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to remove alert rule:', error.message);
            return false;
        }
    }
    /**
     * Get active alerts
     */
    getActiveAlerts() {
        return Array.from(this.activeAlerts.values()).filter(alert => alert.status === 'firing');
    }
    /**
     * Get alert history
     */
    getAlertHistory(limit = 100) {
        return this.alertHistory.slice(-limit);
    }
    /**
     * Get performance history
     */
    getPerformanceHistory(limit = 100) {
        return this.performanceHistory.slice(-limit);
    }
    /**
     * Stop the monitoring system
     */
    async stop() {
        try {
            logger_1.logger.info('🛑 Stopping Performance Monitoring System...');
            // Close metrics server
            await new Promise((resolve) => {
                this.server.close(() => {
                    logger_1.logger.info('📊 Metrics server stopped');
                    resolve();
                });
            });
            // Clear metrics registry
            prom_client_1.register.clear();
            logger_1.logger.info('✅ Performance Monitoring System stopped');
        }
        catch (error) {
            logger_1.logger.error('❌ Error stopping Performance Monitoring System:', error.message);
        }
    }
    /**
     * Cleanup resources
     */
    async cleanup() {
        try {
            logger_1.logger.info('🧹 Cleaning up Performance Monitoring System...');
            await this.stop();
            // Cleanup trading components
            await this.positionManager.cleanup();
            await this.riskManager.cleanup();
            await this.dataIntegration.cleanup();
            logger_1.logger.info('✅ Performance Monitoring System cleanup completed');
        }
        catch (error) {
            logger_1.logger.error('❌ Error during Performance Monitoring System cleanup:', error.message);
        }
    }
}
exports.PerformanceMonitoringSystem = PerformanceMonitoringSystem;
//# sourceMappingURL=PerformanceMonitoringSystem.js.map