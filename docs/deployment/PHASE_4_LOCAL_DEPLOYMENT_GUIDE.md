# Phase 4: Local MacBook Air M2 Deployment Guide

## Overview
This guide provides complete instructions for deploying SmartMarketOOPS as a personal trading bot on your MacBook Air M2 with 8GB RAM. The system is optimized for personal use with real money trading on Delta Exchange.

## 🚀 Quick Start

### Prerequisites
1. **MacBook Air M2** with 8GB RAM
2. **Docker Desktop** for Mac (latest version)
3. **Delta Exchange Account** with API access
4. **Node.js 18+** and **npm**
5. **Git** for version control

### Installation Steps

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd smartmarket-oops
   ```

2. **Configure Environment**
   ```bash
   cp .env.local.template .env.local
   # Edit .env.local with your Delta Exchange API credentials
   ```

3. **Start the System**
   ```bash
   ./start-local.sh
   ```

4. **Access Dashboards**
   - Trading Dashboard: http://localhost:3000
   - Monitoring: http://localhost:3002 (admin/admin123)
   - API Documentation: http://localhost:3001/api/docs

## 📊 System Architecture

### Memory Allocation (8GB Total)
```
┌─────────────────────────────────────────────────────────────┐
│                    MacBook Air M2 - 8GB RAM                │
├─────────────────────────────────────────────────────────────┤
│ macOS System + Other Apps           │ 1GB                   │
│ PostgreSQL Database                 │ 1GB                   │
│ Backend API Server                  │ 2GB                   │
│ ML Service (Enhanced + Fibonacci)   │ 1.5GB                 │
│ Frontend Dashboard                  │ 1GB                   │
│ Redis Cache                         │ 512MB                 │
│ Monitoring (Prometheus + Grafana)   │ 1GB                   │
│ System Buffer                       │ 512MB                 │
└─────────────────────────────────────────────────────────────┘
```

### Service Architecture
```
Frontend (Next.js) ──┐
                     ├── Backend API ──┬── PostgreSQL
ML Service ──────────┘                 ├── Redis
                                       ├── Delta Exchange API
                                       └── Monitoring Stack
```

## 🔧 Configuration

### Personal Trading Settings
- **Initial Capital**: $1,000 (configurable)
- **Max Position Size**: $100 per trade
- **Risk per Trade**: 2% of capital
- **Max Positions**: 3 simultaneous positions
- **Daily Loss Limit**: $50

### ML Model Configuration
- **Enhanced ML Model**: 60% weight, 65% confidence threshold
- **Fibonacci ML Model**: 40% weight, 60% confidence threshold
- **Ensemble Predictions**: Weighted voting system

### Risk Management
- **Stop Loss**: 2% automatic stop loss
- **Take Profit**: 4% automatic take profit
- **Trailing Stop**: 1.5% trailing stop
- **Emergency Stop**: 5% daily loss or 15% drawdown

## 🛠 Management Scripts

### Start System
```bash
./start-local.sh
```
- Starts all services in correct order
- Runs health checks
- Displays access URLs

### Stop System
```bash
./stop-local.sh
```
- Gracefully stops all services
- Preserves data and logs
- Warns about open positions

### Restart System
```bash
./restart-local.sh
```
- Quick restart without rebuilding
- Maintains data persistence

### Backup Data
```bash
./backup-local.sh
```
- Creates compressed backup
- Includes database, logs, and config
- Automatic cleanup of old backups

## 📈 Monitoring & Dashboards

### Trading Dashboard (Port 3000)
- Real-time portfolio value
- Active positions and P&L
- ML model predictions
- Trading activity charts
- Risk metrics

### Grafana Monitoring (Port 3002)
- System resource usage
- Trading performance metrics
- ML model accuracy
- Alert notifications
- Custom dashboards

### Prometheus Metrics (Port 9090)
- Raw metrics collection
- Custom trading metrics
- System health metrics
- Performance data

## 🔒 Security Features

### API Security
- JWT token authentication
- Rate limiting (1000 req/15min)
- CORS protection
- Input validation and sanitization

### Data Protection
- Encrypted API keys
- Secure password hashing
- Local data storage only
- Automated backups

### Trading Security
- Position size limits
- Daily loss limits
- Emergency stop mechanisms
- Real-time risk monitoring

## 💰 Trading Features

### Automated Trading
- ML-powered signal generation
- Risk-adjusted position sizing
- Automatic stop loss/take profit
- Multi-timeframe analysis

### Manual Override
- Manual trade execution
- Position management
- Risk parameter adjustment
- Emergency stop controls

### Paper Trading Mode
- Test strategies without real money
- 7-day paper trading period
- Performance validation
- Risk-free testing

## 🔧 Troubleshooting

### Common Issues

1. **High Memory Usage**
   ```bash
   # Check memory usage
   docker stats
   # Restart services if needed
   ./restart-local.sh
   ```

2. **API Connection Issues**
   ```bash
   # Check Delta Exchange connectivity
   curl -s https://api.delta.exchange/v2/products
   # Verify API credentials in .env.local
   ```

3. **Database Issues**
   ```bash
   # Check database health
   docker-compose -f docker-compose.local.yml exec postgres pg_isready
   # View database logs
   docker-compose -f docker-compose.local.yml logs postgres
   ```

4. **ML Service Issues**
   ```bash
   # Check ML service health
   curl -s http://localhost:8000/health
   # View ML service logs
   docker-compose -f docker-compose.local.yml logs ml-service
   ```

### Performance Optimization

1. **Memory Optimization**
   - Close unnecessary applications
   - Monitor Activity Monitor
   - Restart services if memory usage high

2. **CPU Optimization**
   - Ensure good ventilation
   - Monitor thermal throttling
   - Reduce ML inference frequency if needed

3. **Network Optimization**
   - Use stable internet connection
   - Monitor API latency
   - Check for network interruptions

## 📊 Performance Targets

### System Performance
- **API Response Time**: <100ms
- **ML Prediction Time**: <500ms
- **Database Query Time**: <50ms
- **Memory Usage**: <6GB total
- **CPU Usage**: <80% average

### Trading Performance
- **Daily Return Target**: 0.5%
- **Weekly Return Target**: 2.0%
- **Monthly Return Target**: 8.0%
- **Max Drawdown**: <10%
- **Win Rate Target**: >60%
- **Sharpe Ratio Target**: >1.5

## 🔄 Maintenance

### Daily Tasks
- [ ] Check system health
- [ ] Review trading performance
- [ ] Monitor open positions
- [ ] Check error logs

### Weekly Tasks
- [ ] Run backup script
- [ ] Review ML model performance
- [ ] Analyze trading metrics
- [ ] Update risk parameters if needed

### Monthly Tasks
- [ ] Update dependencies
- [ ] Review and optimize configuration
- [ ] Performance benchmarking
- [ ] Security review

## 🚨 Safety Reminders

### Important Warnings
- ⚠️ **Real Money Trading**: This system trades with real money on Delta Exchange
- ⚠️ **Risk Management**: Always monitor your positions and risk exposure
- ⚠️ **System Monitoring**: Keep an eye on system health and performance
- ⚠️ **Backup Strategy**: Regular backups are essential for data protection

### Best Practices
1. **Start with Paper Trading**: Test the system thoroughly before live trading
2. **Monitor Closely**: Watch the system especially in the first few days
3. **Set Conservative Limits**: Start with lower risk parameters
4. **Regular Backups**: Run backups before making any changes
5. **Stay Informed**: Keep up with market conditions and system updates

## 📞 Support

### Log Files
- **Application Logs**: `./logs/`
- **Backend Logs**: `./backend/logs/`
- **ML Service Logs**: `./ml/logs/`
- **Docker Logs**: `docker-compose -f docker-compose.local.yml logs`

### Health Checks
- **System Health**: http://localhost:3001/api/health
- **ML Service Health**: http://localhost:8000/health
- **Database Health**: `docker-compose -f docker-compose.local.yml exec postgres pg_isready`

### Configuration Files
- **Environment**: `.env.local`
- **Trading Config**: `config/personal-trading.json`
- **Docker Config**: `docker-compose.local.yml`
- **Monitoring Config**: `monitoring/prometheus-local.yml`

## 🎯 Success Metrics

Your SmartMarketOOPS personal trading bot is successfully deployed when:

- ✅ All services are running and healthy
- ✅ Trading dashboard is accessible
- ✅ ML models are making predictions
- ✅ Risk management is active
- ✅ Monitoring is collecting metrics
- ✅ Backups are working
- ✅ API connectivity to Delta Exchange is stable

**Happy Trading! 🚀💰**

Remember: This is your personal trading bot optimized for your MacBook Air M2. Start conservatively, monitor closely, and adjust parameters based on your risk tolerance and performance goals.