# 🧪 SmartMarketOOPS Testing Suite

Comprehensive testing framework for the ML trading system with unit tests, load testing, stress testing, and acceptance criteria validation.

## 📋 **Test Categories**

### 🔬 **Unit Tests** (`test_ml_models.py`)
- **ML Model Validation**: Tests model accuracy, prediction consistency, and edge cases
- **Feature Engineering**: Validates technical indicators and data transformations
- **Confidence Scoring**: Tests prediction confidence calculations
- **Edge Case Handling**: Tests system behavior with extreme/invalid inputs

### ⚡ **Load Testing** (`load_testing/locustfile.py`)
- **Normal Load**: 50 users, 5-minute duration
- **High Load**: 200 users, 10-minute duration  
- **Stress Test**: 500 users, 15-minute duration
- **WebSocket Load**: 100 concurrent WebSocket connections
- **HFT Simulation**: High-frequency trading scenarios

### 🌪️ **Stress Testing** (`stress_testing/market_volatility_test.py`)
- **Market Data Bursts**: 5000+ rapid price updates
- **Flash Crash Simulation**: 30% price drop scenarios
- **Extreme Volatility**: 20% volatility simulation
- **Sustained Load**: 50 users for 3 minutes continuous

### ✅ **Acceptance Criteria** (`acceptance/acceptance_criteria_validator.py`)
- **Performance**: API response times, throughput, error rates
- **Trading**: Win rate, Sharpe ratio, drawdown limits
- **ML Models**: Prediction accuracy, confidence scores
- **Risk Management**: Position sizing, stop-loss execution
- **Data Quality**: Completeness, accuracy, latency

## 🚀 **Quick Start**

### Prerequisites
```bash
pip install pytest pytest-cov locust aiohttp websockets numpy pandas
```

### Run All Tests
```bash
# Unit tests with coverage
pytest tests/test_ml_models.py -v --cov=ml_models --cov-report=html

# Load testing (requires running system)
locust -f tests/load_testing/locustfile.py --host=http://localhost:8000

# Stress testing
python tests/stress_testing/market_volatility_test.py

# Acceptance criteria validation
python tests/acceptance/acceptance_criteria_validator.py
```

## 📊 **Acceptance Criteria Thresholds**

### **Performance Requirements**
- **API Response Time P95**: ≤ 500ms
- **API Response Time P99**: ≤ 1000ms
- **Throughput**: ≥ 100 requests/second
- **Error Rate**: ≤ 1.0%
- **Uptime**: ≥ 99.9%

### **Trading Performance**
- **Win Rate**: ≥ 60%
- **Sharpe Ratio**: ≥ 1.5
- **Max Drawdown**: ≤ 20%
- **Daily Trades**: 3-5 trades
- **Position Accuracy**: ≥ 85%

### **ML Model Performance**
- **Prediction Accuracy**: ≥ 75%
- **Model Confidence**: ≥ 70%
- **Training Time**: ≤ 300 seconds
- **Inference Time**: ≤ 100ms

### **Risk Management**
- **Position Size Accuracy**: ≥ 95%
- **Stop Loss Execution**: ≥ 99%
- **Take Profit Execution**: ≥ 95%
- **Leverage Compliance**: 100%

### **Data Quality**
- **Data Completeness**: ≥ 99.5%
- **Data Accuracy**: ≥ 99.9%
- **Latency**: ≤ 100ms
- **Missing Data**: ≤ 0.1%

## 🔧 **Test Configuration**

### **Load Test Scenarios**
```python
scenarios = {
    "normal_load": {"users": 50, "duration": "5m"},
    "high_load": {"users": 200, "duration": "10m"},
    "stress_test": {"users": 500, "duration": "15m"},
    "websocket_load": {"users": 100, "duration": "5m"}
}
```

### **Stress Test Parameters**
```python
volatility_scenarios = {
    "normal": 0.05,      # 5% volatility
    "high": 0.10,        # 10% volatility  
    "extreme": 0.20,     # 20% volatility
    "flash_crash": 0.30  # 30% price drop
}
```

## 📈 **Test Reports**

### **Generated Reports**
- `htmlcov/index.html` - Unit test coverage report
- `stress_test_report_*.json` - Stress test results
- `acceptance_validation_report_*.json` - Acceptance criteria results
- Locust generates real-time web dashboard at `http://localhost:8089`

### **Report Metrics**
- **Response Times**: Mean, median, P95, P99, max
- **Throughput**: Requests per second
- **Error Analysis**: Error types and frequencies
- **Success Rates**: Percentage of successful operations
- **Performance Trends**: Time-series performance data

## 🎯 **Test Execution Strategy**

### **Development Testing**
```bash
# Quick unit tests during development
pytest tests/test_ml_models.py::TestEnhancedMLModel::test_model_training -v

# Fast load test
locust -f tests/load_testing/locustfile.py --host=http://localhost:8000 -u 10 -r 2 -t 60s --headless
```

### **Pre-Production Testing**
```bash
# Full test suite
pytest tests/ -v --cov=. --cov-report=html
python tests/stress_testing/market_volatility_test.py
python tests/acceptance/acceptance_criteria_validator.py
```

### **Production Monitoring**
```bash
# Continuous acceptance criteria validation
python tests/acceptance/acceptance_criteria_validator.py --continuous --interval=3600
```

## 🔍 **Test Data Management**

### **Mock Data Generation**
- **Market Data**: Realistic OHLCV data with configurable volatility
- **Portfolio Data**: Simulated portfolio metrics and positions
- **ML Predictions**: Mock model outputs with confidence scores
- **Risk Metrics**: Simulated risk calculations and alerts

### **Test Environment Setup**
```bash
# Set test environment variables
export TESTING=true
export API_BASE_URL=http://localhost:8000
export WS_URL=ws://localhost:8000/ws
export LOG_LEVEL=INFO
```

## 🚨 **Failure Handling**

### **Test Failure Scenarios**
- **API Timeouts**: Graceful degradation testing
- **Database Failures**: Resilience validation
- **Network Issues**: Connection retry testing
- **Memory Pressure**: Resource constraint testing

### **Recovery Testing**
- **Auto-Reconnection**: WebSocket recovery validation
- **Failover**: Backup system activation
- **Data Recovery**: State restoration testing
- **Performance Recovery**: System optimization validation

## 📝 **Test Maintenance**

### **Regular Updates**
- Update acceptance criteria thresholds based on system improvements
- Add new test scenarios for new features
- Maintain test data relevance with market conditions
- Review and optimize test execution times

### **Continuous Integration**
```yaml
# Example CI pipeline
test_pipeline:
  - unit_tests: pytest tests/test_ml_models.py
  - load_tests: locust --headless -u 50 -r 5 -t 300s
  - acceptance: python tests/acceptance/acceptance_criteria_validator.py
  - reports: Generate and archive test reports
```

---

**🎯 Testing ensures SmartMarketOOPS meets professional trading standards with institutional-grade reliability and performance.**
