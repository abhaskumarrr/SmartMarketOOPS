# Performance Monitoring System

## 📊 Overview

The Performance Monitoring System provides comprehensive metrics collection, real-time monitoring, and alerting for the trading system. Built with Prometheus integration, Grafana dashboards, and advanced analytics, it delivers enterprise-grade observability for high-frequency trading operations.

## 🏗️ Architecture

### Core Components

1. **Prometheus Metrics Collection** - High-performance time-series metrics storage
2. **Real-Time Alerting System** - Intelligent alert rules with severity levels
3. **Performance Analytics Engine** - Advanced trading performance calculations
4. **Grafana Dashboard Integration** - Professional visualization and monitoring
5. **API Endpoints** - RESTful access to metrics and alerts
6. **Health Monitoring** - System-wide health checks and diagnostics

### Monitoring Flow
```
Trading System → Metrics Collection → Prometheus Storage → Grafana Visualization
                                                        ↓
                                              Alert Evaluation → Notifications
```

## 🎯 Key Features

### 1. **Comprehensive Metrics Collection**
```typescript
// Trading metrics
tradesTotal: Counter              // Total trades by symbol/action/status
tradePnL: Histogram              // P&L distribution with buckets
tradeLatency: Histogram          // Execution latency tracking

// ML model metrics
modelAccuracy: Gauge             // Real-time model accuracy
modelPredictions: Counter        // Prediction counts by type
modelLatency: Histogram          // Inference time tracking

// Risk metrics
portfolioValue: Gauge            // Current portfolio value
drawdown: Gauge                  // Portfolio drawdown percentage
riskScore: Gauge                 // Multi-dimensional risk scores

// System metrics
systemErrors: Counter            // Error tracking by component
apiRequests: Counter             // API usage statistics
apiLatency: Histogram            // API response times
```

### 2. **Advanced Alert System**
```typescript
// Alert rule structure
interface AlertRule {
  id: string;
  name: string;
  metric: string;
  condition: 'greater_than' | 'less_than' | 'equals' | 'not_equals';
  threshold: number;
  duration: number;              // Evaluation period
  severity: 'critical' | 'warning' | 'info';
  enabled: boolean;
  description: string;
}

// Default alert rules
- High Error Rate (>10 errors/5min) → CRITICAL
- Low Win Rate (<40%) → WARNING  
- High Drawdown (>15%) → CRITICAL
- High API Latency (>1s) → WARNING
- Low Data Quality (<80%) → WARNING
- Model Accuracy Drop (<70%) → WARNING
```

### 3. **Performance Analytics**
```typescript
// Comprehensive performance metrics
interface PerformanceMetrics {
  // Trading performance
  totalTrades: number;
  successfulTrades: number;
  winRate: number;               // Success rate percentage
  totalPnL: number;              // Total profit/loss
  averageTradeReturn: number;    // Average return per trade
  
  // System performance  
  systemUptime: number;          // System uptime in ms
  averageLatency: number;        // Average response latency
  errorRate: number;             // System error rate
  throughput: number;            // Signals processed per second
  
  // ML model performance
  modelAccuracy: number;         // Current model accuracy
  modelPrecision: number;        // Model precision score
  modelRecall: number;           // Model recall score
  modelF1Score: number;          // F1 score
  
  // Risk metrics
  currentDrawdown: number;       // Current portfolio drawdown
  maxDrawdown: number;           // Maximum historical drawdown
  sharpeRatio: number;           // Risk-adjusted returns
  volatility: number;            // Portfolio volatility
}
```

### 4. **Prometheus Integration**
```typescript
// Prometheus configuration optimized for trading
global:
  scrape_interval: 15s           // High-frequency scraping
  evaluation_interval: 15s       // Real-time alert evaluation

scrape_configs:
  - job_name: 'trading-system'
    scrape_interval: 5s          // Ultra-fast for trading metrics
    static_configs:
      - targets: ['localhost:9090']
    
  - job_name: 'analysis-execution-bridge'
    scrape_interval: 10s         // Bridge monitoring
    static_configs:
      - targets: ['localhost:8000']
```

### 5. **Grafana Dashboard**
```json
// Professional trading dashboard panels
{
  "panels": [
    "Trading Overview",          // Total trades, win rate stats
    "Portfolio Value",           // Real-time portfolio tracking
    "Trade P&L Distribution",    // Profit/loss histogram
    "System Latency",           // 95th percentile latencies
    "Risk Metrics",             // Drawdown and risk scores
    "ML Model Performance",      // Accuracy and prediction rates
    "System Errors",            // Error rates by component
    "Data Quality Heatmap"      // Quality scores by symbol/timeframe
  ]
}
```

## 📈 Usage Examples

### Initialize Monitoring System
```typescript
import { PerformanceMonitoringSystem } from './services/PerformanceMonitoringSystem';

// Create monitoring system with custom configuration
const monitoring = new PerformanceMonitoringSystem({
  metricsPort: 9090,
  enableDefaultMetrics: true,
  scrapeInterval: 15000,         // 15 seconds
  alertCheckInterval: 30000,     // 30 seconds
  retentionPeriod: 30,          // 30 days
  enableGrafanaIntegration: true
});

// Initialize and start
await monitoring.initialize();
await monitoring.start();

console.log('📊 Performance Monitoring System operational!');
```

### Record Trading Metrics
```typescript
// Record trading decision
const decision = await decisionEngine.generateTradingDecision('BTCUSD');
const latency = 75; // ms

monitoring.recordTradingDecision(decision, latency);

// Record trade execution
monitoring.recordTradeExecution(
  'BTCUSD',           // symbol
  'buy',              // action
  'success',          // status
  150.50,             // P&L
  120                 // latency in ms
);

// Record API request
monitoring.recordApiRequest('POST', '/api/signals', 201, 45);

// Record system error
monitoring.recordError('trading_engine', 'connection_timeout');
```

### Monitor Performance Metrics
```typescript
// Get current performance metrics
const metrics = monitoring.getPerformanceMetrics();

console.log('📊 PERFORMANCE DASHBOARD:');
console.log(`   Total Trades: ${metrics.totalTrades}`);
console.log(`   Win Rate: ${(metrics.winRate * 100).toFixed(1)}%`);
console.log(`   Total P&L: $${metrics.totalPnL.toFixed(2)}`);
console.log(`   Average Latency: ${metrics.averageLatency.toFixed(2)}ms`);
console.log(`   Model Accuracy: ${(metrics.modelAccuracy * 100).toFixed(1)}%`);
console.log(`   Current Drawdown: ${(metrics.currentDrawdown * 100).toFixed(2)}%`);
console.log(`   Sharpe Ratio: ${metrics.sharpeRatio.toFixed(3)}`);
```

### Manage Alerts
```typescript
// Add custom alert rule
const ruleId = monitoring.addAlertRule({
  name: 'High Trade Volume',
  metric: 'trading_trades_total',
  condition: 'greater_than',
  threshold: 100,
  duration: 300,                 // 5 minutes
  severity: 'warning',
  enabled: true,
  description: 'Trading volume is unusually high'
});

// Get active alerts
const activeAlerts = monitoring.getActiveAlerts();
console.log(`🚨 Active Alerts: ${activeAlerts.length}`);

activeAlerts.forEach(alert => {
  console.log(`   - ${alert.name} (${alert.severity}): ${alert.message}`);
});

// Remove alert rule
monitoring.removeAlertRule(ruleId);
```

### Access API Endpoints
```typescript
// Health check
GET /health
// Response: { status: 'healthy', uptime: 86400000, timestamp: 1640995200000 }

// Prometheus metrics
GET /metrics
// Response: Prometheus-formatted metrics

// Performance metrics
GET /performance
// Response: PerformanceMetrics object

// Active alerts
GET /alerts
// Response: { active: Alert[], resolved: Alert[], total: number }

// Alert rules
GET /alert-rules
// Response: AlertRule[]
```

## 🔧 Configuration Options

### **Monitoring Configuration**
```typescript
interface MonitoringConfig {
  metricsPort: number;           // 9090 - Prometheus metrics port
  enableDefaultMetrics: boolean; // true - Enable Node.js metrics
  scrapeInterval: number;        // 15000ms - Metrics update interval
  alertCheckInterval: number;    // 30000ms - Alert evaluation interval
  retentionPeriod: number;       // 30 days - Data retention period
  enableGrafanaIntegration: boolean; // true - Enable Grafana support
}
```

### **Prometheus Configuration**
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    monitor: 'trading-system-monitor'
    environment: 'production'

scrape_configs:
  - job_name: 'trading-system'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 5s
    metrics_path: /metrics
```

### **Alert Rules Configuration**
```yaml
# trading_rules.yml
groups:
  - name: trading_performance
    rules:
      - alert: LowWinRate
        expr: (sum(rate(trading_trades_total{status="success"}[10m])) / sum(rate(trading_trades_total[10m]))) * 100 < 40
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Trading win rate below threshold"
```

## 📊 Metrics Reference

### **Trading Metrics**
```prometheus
# Trade counters
trading_trades_total{symbol, action, status}

# P&L distribution
trading_trade_pnl{symbol, action}

# Execution latency
trading_trade_latency_seconds{symbol, action}
```

### **ML Model Metrics**
```prometheus
# Model accuracy
ml_model_accuracy{model_type, symbol}

# Prediction counts
ml_model_predictions_total{model_type, symbol, prediction}

# Inference latency
ml_model_latency_seconds{model_type, symbol}
```

### **Risk Metrics**
```prometheus
# Portfolio value
risk_portfolio_value

# Drawdown percentage
risk_drawdown_percentage

# Risk scores
risk_overall_score{risk_type}
```

### **System Metrics**
```prometheus
# Error tracking
system_errors_total{component, error_type}

# API metrics
api_requests_total{method, endpoint, status}
api_request_latency_seconds{method, endpoint}

# Data quality
data_quality_score{symbol, timeframe}
data_collection_latency_seconds{symbol, timeframe}
```

## 🚨 Alert Rules

### **Critical Alerts**
- **High Drawdown** (>15%): Immediate risk management action required
- **Critical Trade Latency** (>5s): System performance severely degraded
- **High Trade Failure Rate** (>20%): Trading system reliability issues
- **Critical Model Accuracy** (<50%): ML model requires immediate attention

### **Warning Alerts**
- **Low Win Rate** (<40%): Trading strategy performance declining
- **High API Latency** (>1s): System responsiveness issues
- **Low Data Quality** (<80%): Data integrity concerns
- **Model Accuracy Drop** (<70%): ML model performance degradation

### **Info Alerts**
- **No Trading Activity** (30min): System may be idle or paused
- **High Risk Score** (>80%): Elevated risk conditions detected

## 🧪 Testing

### Run Comprehensive Test
```bash
cd backend
npx ts-node src/scripts/test-performance-monitoring.ts
```

### Test Coverage
- ✅ Monitoring system initialization and startup
- ✅ Prometheus metrics collection and formatting
- ✅ Trading metrics recording and validation
- ✅ Alert system functionality and rule evaluation
- ✅ Performance metrics calculation and accuracy
- ✅ API endpoints and response validation
- ✅ Data quality monitoring and tracking
- ✅ System health monitoring and diagnostics

## 🔗 Integration Points

### **Enhanced Trading Decision Engine**
- Real-time decision latency tracking
- Model accuracy monitoring
- Prediction rate analysis

### **ML Position Manager**
- Trade execution metrics
- Position performance tracking
- P&L distribution analysis

### **Enhanced Risk Management System**
- Risk score monitoring
- Drawdown tracking
- Portfolio value updates

### **Analysis-Execution Bridge**
- API request monitoring
- System throughput tracking
- Error rate analysis

## 🎯 Summary

The Performance Monitoring System provides:

- **📊 Comprehensive Metrics**: 15+ metric types covering trading, ML, risk, and system performance
- **🚨 Intelligent Alerting**: 12+ default alert rules with customizable thresholds and severity levels
- **📈 Real-Time Analytics**: Sub-second performance tracking with historical trend analysis
- **🔧 Prometheus Integration**: Enterprise-grade metrics collection with 5-second scrape intervals
- **📱 Grafana Dashboards**: Professional visualization with 8 specialized panels
- **🔌 RESTful APIs**: 5 endpoints for programmatic access to metrics and alerts
- **⚡ High Performance**: Optimized for high-frequency trading with minimal overhead
- **🧪 Comprehensive Testing**: 8-step validation with performance benchmarking

This system transforms trading operations into a fully observable, monitored, and optimized platform with enterprise-grade reliability and performance insights!
