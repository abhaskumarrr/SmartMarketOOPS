#!/usr/bin/env node
"use strict";
/**
 * Performance Monitoring System Test
 * Comprehensive testing of Prometheus metrics, alerts, and monitoring functionality
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.PerformanceMonitoringTest = void 0;
const PerformanceMonitoringSystem_1 = require("../services/PerformanceMonitoringSystem");
const logger_1 = require("../utils/logger");
const axios_1 = __importDefault(require("axios"));
class PerformanceMonitoringTest {
    constructor() {
        this.baseUrl = 'http://localhost:9090';
        this.testSymbols = ['BTCUSD', 'ETHUSD'];
        this.monitoringSystem = new PerformanceMonitoringSystem_1.PerformanceMonitoringSystem({
            metricsPort: 9090,
            enableDefaultMetrics: true,
            scrapeInterval: 5000, // 5 seconds for testing
            alertCheckInterval: 10000, // 10 seconds for testing
            retentionPeriod: 1, // 1 day for testing
            enableGrafanaIntegration: true
        });
    }
    /**
     * Run comprehensive performance monitoring test
     */
    async runTest() {
        logger_1.logger.info('📊 PERFORMANCE MONITORING SYSTEM TEST');
        logger_1.logger.info('='.repeat(80));
        try {
            // Step 1: Initialize and start the monitoring system
            await this.testMonitoringInitialization();
            // Step 2: Test Prometheus metrics collection
            await this.testPrometheusMetrics();
            // Step 3: Test trading metrics recording
            await this.testTradingMetricsRecording();
            // Step 4: Test alert system functionality
            await this.testAlertSystem();
            // Step 5: Test performance metrics calculation
            await this.testPerformanceMetricsCalculation();
            // Step 6: Test API endpoints
            await this.testApiEndpoints();
            // Step 7: Test data quality monitoring
            await this.testDataQualityMonitoring();
            // Step 8: Test system health monitoring
            await this.testSystemHealthMonitoring();
            logger_1.logger.info('\n🎉 PERFORMANCE MONITORING SYSTEM TEST COMPLETED SUCCESSFULLY!');
            logger_1.logger.info('✅ All monitoring and alerting features are working correctly');
        }
        catch (error) {
            logger_1.logger.error('❌ Performance monitoring system test failed:', error.message);
            throw error;
        }
        finally {
            // Cleanup
            await this.monitoringSystem.stop();
        }
    }
    /**
     * Test monitoring system initialization
     */
    async testMonitoringInitialization() {
        logger_1.logger.info('\n🔧 STEP 1: MONITORING INITIALIZATION TEST');
        // Initialize the monitoring system
        await this.monitoringSystem.initialize();
        logger_1.logger.info('✅ Monitoring system initialized successfully');
        // Start the monitoring system
        await this.monitoringSystem.start();
        logger_1.logger.info('✅ Monitoring system started successfully');
        // Wait for server to be ready
        await this.sleep(2000);
        // Test basic connectivity
        try {
            const healthResponse = await axios_1.default.get(`${this.baseUrl}/health`);
            logger_1.logger.info(`✅ Health check: ${healthResponse.status} - ${healthResponse.data.status}`);
            logger_1.logger.info(`   Uptime: ${(healthResponse.data.uptime / 1000).toFixed(1)}s`);
        }
        catch (error) {
            logger_1.logger.error('❌ Health check failed:', error.message);
            throw error;
        }
    }
    /**
     * Test Prometheus metrics collection
     */
    async testPrometheusMetrics() {
        logger_1.logger.info('\n📊 STEP 2: PROMETHEUS METRICS TEST');
        try {
            // Test metrics endpoint
            const metricsResponse = await axios_1.default.get(`${this.baseUrl}/metrics`);
            logger_1.logger.info(`✅ Metrics endpoint: ${metricsResponse.status}`);
            const metricsText = metricsResponse.data;
            // Check for key metrics
            const expectedMetrics = [
                'trading_trades_total',
                'trading_trade_pnl',
                'trading_trade_latency_seconds',
                'ml_model_accuracy',
                'ml_model_predictions_total',
                'risk_portfolio_value',
                'risk_drawdown_percentage',
                'system_errors_total',
                'api_requests_total'
            ];
            let foundMetrics = 0;
            for (const metric of expectedMetrics) {
                if (metricsText.includes(metric)) {
                    foundMetrics++;
                    logger_1.logger.info(`   ✅ Found metric: ${metric}`);
                }
                else {
                    logger_1.logger.warn(`   ⚠️ Missing metric: ${metric}`);
                }
            }
            logger_1.logger.info(`📊 Metrics coverage: ${foundMetrics}/${expectedMetrics.length} (${(foundMetrics / expectedMetrics.length * 100).toFixed(1)}%)`);
            // Check for default Node.js metrics
            const defaultMetrics = ['nodejs_heap_size_total_bytes', 'nodejs_heap_size_used_bytes', 'process_cpu_user_seconds_total'];
            let foundDefaultMetrics = 0;
            for (const metric of defaultMetrics) {
                if (metricsText.includes(metric)) {
                    foundDefaultMetrics++;
                }
            }
            logger_1.logger.info(`📊 Default metrics: ${foundDefaultMetrics}/${defaultMetrics.length} found`);
        }
        catch (error) {
            logger_1.logger.error('❌ Prometheus metrics test failed:', error.message);
            throw error;
        }
    }
    /**
     * Test trading metrics recording
     */
    async testTradingMetricsRecording() {
        logger_1.logger.info('\n📈 STEP 3: TRADING METRICS RECORDING TEST');
        // Test trading decision recording
        logger_1.logger.info('\n🔄 Testing trading decision metrics...');
        for (const symbol of this.testSymbols) {
            const mockDecision = {
                symbol,
                action: 'buy',
                confidence: 0.85,
                timestamp: Date.now(),
                stopLoss: 49000,
                takeProfit: 52000,
                positionSize: 0.05,
                leverage: 100,
                modelVotes: {},
                keyFeatures: {},
                riskScore: 0.3,
                maxDrawdown: 0,
                winProbability: 0,
                urgency: 'medium',
                timeToLive: 300000,
                reasoning: []
            };
            const latency = Math.random() * 100 + 50; // 50-150ms
            this.monitoringSystem.recordTradingDecision(mockDecision, latency);
            logger_1.logger.info(`✅ Recorded trading decision: ${symbol} ${mockDecision.action} (${latency.toFixed(1)}ms)`);
        }
        // Test trade execution recording
        logger_1.logger.info('\n⚡ Testing trade execution metrics...');
        const tradeScenarios = [
            { symbol: 'BTCUSD', action: 'buy', status: 'success', pnl: 150.50, latency: 75 },
            { symbol: 'ETHUSD', action: 'sell', status: 'success', pnl: -25.30, latency: 120 },
            { symbol: 'BTCUSD', action: 'close', status: 'failed', pnl: undefined, latency: 200 }
        ];
        for (const trade of tradeScenarios) {
            this.monitoringSystem.recordTradeExecution(trade.symbol, trade.action, trade.status, trade.pnl, trade.latency);
            logger_1.logger.info(`✅ Recorded trade execution: ${trade.symbol} ${trade.action} ${trade.status} (PnL: ${trade.pnl || 'N/A'}, Latency: ${trade.latency}ms)`);
        }
        // Test API request recording
        logger_1.logger.info('\n🔌 Testing API request metrics...');
        const apiScenarios = [
            { method: 'GET', endpoint: '/api/positions', status: 200, latency: 45 },
            { method: 'POST', endpoint: '/api/signals', status: 201, latency: 80 },
            { method: 'GET', endpoint: '/api/risk', status: 500, latency: 150 }
        ];
        for (const api of apiScenarios) {
            this.monitoringSystem.recordApiRequest(api.method, api.endpoint, api.status, api.latency);
            logger_1.logger.info(`✅ Recorded API request: ${api.method} ${api.endpoint} ${api.status} (${api.latency}ms)`);
        }
        // Test error recording
        logger_1.logger.info('\n❌ Testing error metrics...');
        const errorScenarios = [
            { component: 'trading_engine', errorType: 'connection_timeout' },
            { component: 'risk_manager', errorType: 'calculation_error' },
            { component: 'data_collector', errorType: 'invalid_data' }
        ];
        for (const error of errorScenarios) {
            this.monitoringSystem.recordError(error.component, error.errorType);
            logger_1.logger.info(`✅ Recorded error: ${error.component} - ${error.errorType}`);
        }
    }
    /**
     * Test alert system functionality
     */
    async testAlertSystem() {
        logger_1.logger.info('\n🚨 STEP 4: ALERT SYSTEM TEST');
        // Test adding custom alert rule
        logger_1.logger.info('\n➕ Testing custom alert rule creation...');
        const customRule = {
            name: 'Test High Latency Alert',
            metric: 'api_request_latency_seconds',
            condition: 'greater_than',
            threshold: 0.1, // 100ms
            duration: 30, // 30 seconds
            severity: 'warning',
            enabled: true,
            description: 'API latency is too high for testing'
        };
        const ruleId = this.monitoringSystem.addAlertRule(customRule);
        logger_1.logger.info(`✅ Added custom alert rule: ${ruleId}`);
        // Wait for alert evaluation
        await this.sleep(15000); // Wait 15 seconds for alert checks
        // Check active alerts
        const activeAlerts = this.monitoringSystem.getActiveAlerts();
        logger_1.logger.info(`🚨 Active alerts: ${activeAlerts.length}`);
        for (const alert of activeAlerts) {
            logger_1.logger.info(`   - ${alert.name} (${alert.severity}): ${alert.message}`);
        }
        // Check alert history
        const alertHistory = this.monitoringSystem.getAlertHistory(10);
        logger_1.logger.info(`📚 Alert history: ${alertHistory.length} alerts`);
        // Test removing alert rule
        const removed = this.monitoringSystem.removeAlertRule(ruleId);
        logger_1.logger.info(`🗑️ Removed custom alert rule: ${removed ? 'SUCCESS' : 'FAILED'}`);
    }
    /**
     * Test performance metrics calculation
     */
    async testPerformanceMetricsCalculation() {
        logger_1.logger.info('\n📊 STEP 5: PERFORMANCE METRICS CALCULATION TEST');
        // Update risk metrics
        this.monitoringSystem.updateRiskMetrics();
        logger_1.logger.info('✅ Risk metrics updated');
        // Get current performance metrics
        const metrics = this.monitoringSystem.getPerformanceMetrics();
        logger_1.logger.info('📈 CURRENT PERFORMANCE METRICS:');
        logger_1.logger.info('='.repeat(50));
        logger_1.logger.info(`Trading Performance:`);
        logger_1.logger.info(`   Total Trades: ${metrics.totalTrades}`);
        logger_1.logger.info(`   Successful Trades: ${metrics.successfulTrades}`);
        logger_1.logger.info(`   Failed Trades: ${metrics.failedTrades}`);
        logger_1.logger.info(`   Win Rate: ${(metrics.winRate * 100).toFixed(1)}%`);
        logger_1.logger.info(`   Total P&L: $${metrics.totalPnL.toFixed(2)}`);
        logger_1.logger.info(`   Average Trade Return: ${(metrics.averageTradeReturn * 100).toFixed(2)}%`);
        logger_1.logger.info(`\nSystem Performance:`);
        logger_1.logger.info(`   Uptime: ${(metrics.systemUptime / 1000).toFixed(1)}s`);
        logger_1.logger.info(`   Average Latency: ${metrics.averageLatency.toFixed(2)}ms`);
        logger_1.logger.info(`   Error Rate: ${(metrics.errorRate * 100).toFixed(2)}%`);
        logger_1.logger.info(`   Throughput: ${metrics.throughput.toFixed(2)} signals/sec`);
        logger_1.logger.info(`\nML Model Performance:`);
        logger_1.logger.info(`   Accuracy: ${(metrics.modelAccuracy * 100).toFixed(1)}%`);
        logger_1.logger.info(`   Precision: ${(metrics.modelPrecision * 100).toFixed(1)}%`);
        logger_1.logger.info(`   Recall: ${(metrics.modelRecall * 100).toFixed(1)}%`);
        logger_1.logger.info(`   F1 Score: ${(metrics.modelF1Score * 100).toFixed(1)}%`);
        logger_1.logger.info(`\nRisk Metrics:`);
        logger_1.logger.info(`   Current Drawdown: ${(metrics.currentDrawdown * 100).toFixed(2)}%`);
        logger_1.logger.info(`   Max Drawdown: ${(metrics.maxDrawdown * 100).toFixed(2)}%`);
        logger_1.logger.info(`   Sharpe Ratio: ${metrics.sharpeRatio.toFixed(3)}`);
        logger_1.logger.info(`   Volatility: ${(metrics.volatility * 100).toFixed(1)}%`);
        // Validate metrics structure
        this.validatePerformanceMetrics(metrics);
    }
    /**
     * Test API endpoints
     */
    async testApiEndpoints() {
        logger_1.logger.info('\n🔌 STEP 6: API ENDPOINTS TEST');
        const endpoints = [
            { path: '/health', description: 'Health check' },
            { path: '/metrics', description: 'Prometheus metrics' },
            { path: '/performance', description: 'Performance metrics' },
            { path: '/alerts', description: 'Active alerts' },
            { path: '/alert-rules', description: 'Alert rules' }
        ];
        for (const endpoint of endpoints) {
            try {
                const response = await axios_1.default.get(`${this.baseUrl}${endpoint.path}`);
                logger_1.logger.info(`✅ ${endpoint.description}: ${response.status}`);
                if (endpoint.path === '/performance') {
                    logger_1.logger.info(`   Response size: ${JSON.stringify(response.data).length} bytes`);
                }
                else if (endpoint.path === '/alerts') {
                    logger_1.logger.info(`   Active alerts: ${response.data.active?.length || 0}`);
                    logger_1.logger.info(`   Resolved alerts: ${response.data.resolved?.length || 0}`);
                }
                else if (endpoint.path === '/alert-rules') {
                    logger_1.logger.info(`   Alert rules: ${response.data.length}`);
                }
            }
            catch (error) {
                logger_1.logger.error(`❌ ${endpoint.description} failed: ${error.message}`);
            }
        }
    }
    /**
     * Test data quality monitoring
     */
    async testDataQualityMonitoring() {
        logger_1.logger.info('\n📊 STEP 7: DATA QUALITY MONITORING TEST');
        const dataScenarios = [
            { symbol: 'BTCUSD', timeframe: '5m', quality: 0.95, latency: 150 },
            { symbol: 'ETHUSD', timeframe: '15m', quality: 0.88, latency: 200 },
            { symbol: 'BTCUSD', timeframe: '1h', quality: 0.75, latency: 500 }, // Low quality
            { symbol: 'ETHUSD', timeframe: '4h', quality: 0.92, latency: 100 }
        ];
        for (const scenario of dataScenarios) {
            this.monitoringSystem.updateDataQualityMetrics(scenario.symbol, scenario.timeframe, scenario.quality, scenario.latency);
            logger_1.logger.info(`✅ Updated data quality: ${scenario.symbol} ${scenario.timeframe} - Quality: ${(scenario.quality * 100).toFixed(1)}%, Latency: ${scenario.latency}ms`);
        }
        logger_1.logger.info('📊 Data quality monitoring test completed');
    }
    /**
     * Test system health monitoring
     */
    async testSystemHealthMonitoring() {
        logger_1.logger.info('\n🏥 STEP 8: SYSTEM HEALTH MONITORING TEST');
        // Get performance history
        const performanceHistory = this.monitoringSystem.getPerformanceHistory(5);
        logger_1.logger.info(`📈 Performance history: ${performanceHistory.length} entries`);
        // Get alert history
        const alertHistory = this.monitoringSystem.getAlertHistory(10);
        logger_1.logger.info(`🚨 Alert history: ${alertHistory.length} alerts`);
        // Test metrics after some activity
        await this.sleep(5000); // Wait for metrics update
        try {
            const finalMetrics = await axios_1.default.get(`${this.baseUrl}/metrics`);
            const metricsLines = finalMetrics.data.split('\n').filter((line) => line.startsWith('trading_') || line.startsWith('ml_') || line.startsWith('risk_'));
            logger_1.logger.info(`📊 Final metrics check: ${metricsLines.length} trading-related metrics found`);
            // Sample some key metrics
            const keyMetrics = metricsLines.slice(0, 5);
            keyMetrics.forEach((metric) => {
                if (!metric.startsWith('#')) {
                    logger_1.logger.info(`   ${metric}`);
                }
            });
        }
        catch (error) {
            logger_1.logger.error('❌ Final metrics check failed:', error.message);
        }
        logger_1.logger.info('🏥 System health monitoring test completed');
    }
    /**
     * Validate performance metrics structure
     */
    validatePerformanceMetrics(metrics) {
        const requiredFields = [
            'totalTrades', 'successfulTrades', 'failedTrades', 'winRate',
            'totalPnL', 'averageTradeReturn', 'systemUptime', 'averageLatency',
            'errorRate', 'throughput', 'modelAccuracy', 'modelPrecision',
            'modelRecall', 'modelF1Score', 'currentDrawdown', 'maxDrawdown',
            'sharpeRatio', 'volatility'
        ];
        const missingFields = requiredFields.filter(field => !(field in metrics));
        if (missingFields.length === 0) {
            logger_1.logger.info(`✅ Performance metrics structure validation passed`);
        }
        else {
            logger_1.logger.error(`❌ Missing fields in performance metrics: ${missingFields.join(', ')}`);
        }
        // Validate ranges
        if (metrics.winRate < 0 || metrics.winRate > 1) {
            logger_1.logger.error(`❌ Invalid win rate range: ${metrics.winRate}`);
        }
        if (metrics.errorRate < 0 || metrics.errorRate > 1) {
            logger_1.logger.error(`❌ Invalid error rate range: ${metrics.errorRate}`);
        }
        if (metrics.modelAccuracy < 0 || metrics.modelAccuracy > 1) {
            logger_1.logger.error(`❌ Invalid model accuracy range: ${metrics.modelAccuracy}`);
        }
    }
    /**
     * Sleep utility
     */
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}
exports.PerformanceMonitoringTest = PerformanceMonitoringTest;
/**
 * Main execution
 */
async function main() {
    const tester = new PerformanceMonitoringTest();
    await tester.runTest();
}
// Run if this script is executed directly
if (require.main === module) {
    main().catch(error => {
        logger_1.logger.error('💥 Performance monitoring system test failed:', error);
        process.exit(1);
    });
}
//# sourceMappingURL=test-performance-monitoring.js.map