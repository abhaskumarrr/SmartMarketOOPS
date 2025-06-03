-- QuestDB Optimized Schema for SmartMarketOOPS
-- High-performance time-series database schema for financial data
-- Designed for 10-100x query performance improvement

-- ============================================================================
-- METRICS TABLE
-- Optimized for high-frequency metric ingestion and time-series queries
-- ============================================================================

-- Note: QuestDB tables are created automatically when data is inserted
-- This file documents the expected schema structure

-- Metrics table structure (created via line protocol)
-- Table: metrics
-- Columns:
--   timestamp TIMESTAMP (designated timestamp column)
--   name SYMBOL (indexed tag)
--   value DOUBLE (metric value)
--   Additional tags as SYMBOL columns (automatically indexed)

-- Example line protocol format:
-- metrics,name=cpu_usage,host=server1,region=us-east value=75.5 1640995200000000

-- ============================================================================
-- TRADING SIGNALS TABLE
-- Optimized for trading signal storage and retrieval
-- ============================================================================

-- Trading signals table structure
-- Table: trading_signals
-- Columns:
--   timestamp TIMESTAMP (designated timestamp column)
--   id SYMBOL (signal ID)
--   symbol SYMBOL (trading symbol, indexed)
--   type SYMBOL (signal type: ENTRY, EXIT, etc.)
--   direction SYMBOL (LONG, SHORT, NEUTRAL)
--   strength SYMBOL (signal strength)
--   timeframe SYMBOL (timeframe)
--   source SYMBOL (signal source)
--   price DOUBLE (signal price)
--   target_price DOUBLE (target price, nullable)
--   stop_loss DOUBLE (stop loss price, nullable)
--   confidence_score INT (0-100)
--   expected_return DOUBLE
--   expected_risk DOUBLE
--   risk_reward_ratio DOUBLE

-- Example line protocol format:
-- trading_signals,id=sig_001,symbol=BTCUSD,type=ENTRY,direction=LONG,strength=STRONG,timeframe=1h,source=ml_model_v1 price=45000.0,confidence_score=85,expected_return=0.05,expected_risk=0.02,risk_reward_ratio=2.5 1640995200000000

-- ============================================================================
-- ML PREDICTIONS TABLE
-- Optimized for machine learning prediction storage
-- ============================================================================

-- ML predictions table structure
-- Table: ml_predictions
-- Columns:
--   timestamp TIMESTAMP (designated timestamp column)
--   id SYMBOL (prediction ID)
--   model_id SYMBOL (ML model ID, indexed)
--   symbol SYMBOL (trading symbol, indexed)
--   timeframe SYMBOL (prediction timeframe)
--   prediction_type SYMBOL (PRICE, DIRECTION, PROBABILITY)
--   values STRING (JSON array of prediction values)
--   confidence_scores STRING (JSON array of confidence scores)

-- Example line protocol format:
-- ml_predictions,id=pred_001,model_id=transformer_v1,symbol=BTCUSD,timeframe=1h,prediction_type=PRICE values="[45100.0,45200.0,45150.0]",confidence_scores="[0.85,0.82,0.88]" 1640995200000000

-- ============================================================================
-- PERFORMANCE METRICS TABLE
-- Optimized for system performance monitoring
-- ============================================================================

-- Performance metrics table structure
-- Table: performance_metrics
-- Columns:
--   timestamp TIMESTAMP (designated timestamp column)
--   system SYMBOL (system name: API, ML, TRADING, DATABASE)
--   component SYMBOL (component name)
--   metric SYMBOL (metric name)
--   unit SYMBOL (metric unit)
--   value DOUBLE (metric value)
--   Additional tags as SYMBOL columns

-- Example line protocol format:
-- performance_metrics,system=API,component=auth,metric=response_time,unit=ms,endpoint=/api/login value=125.5 1640995200000000

-- ============================================================================
-- MARKET DATA TABLE (OHLCV)
-- Optimized for high-frequency market data storage
-- ============================================================================

-- Market data table structure
-- Table: market_data
-- Columns:
--   timestamp TIMESTAMP (designated timestamp column)
--   symbol SYMBOL (trading symbol, indexed)
--   exchange SYMBOL (exchange name, indexed)
--   timeframe SYMBOL (candle timeframe: 1m, 5m, 1h, etc.)
--   open DOUBLE (opening price)
--   high DOUBLE (highest price)
--   low DOUBLE (lowest price)
--   close DOUBLE (closing price)
--   volume DOUBLE (trading volume)
--   trades INT (number of trades, nullable)

-- Example line protocol format:
-- market_data,symbol=BTCUSD,exchange=binance,timeframe=1m open=45000.0,high=45100.0,low=44950.0,close=45050.0,volume=1250.5,trades=125 1640995200000000

-- ============================================================================
-- ORDER EXECUTION TABLE
-- Optimized for order tracking and execution analysis
-- ============================================================================

-- Order execution table structure
-- Table: order_executions
-- Columns:
--   timestamp TIMESTAMP (designated timestamp column)
--   order_id SYMBOL (order ID, indexed)
--   user_id SYMBOL (user ID, indexed)
--   bot_id SYMBOL (bot ID, indexed, nullable)
--   symbol SYMBOL (trading symbol, indexed)
--   side SYMBOL (BUY, SELL)
--   type SYMBOL (MARKET, LIMIT, STOP, etc.)
--   status SYMBOL (PENDING, FILLED, CANCELLED, etc.)
--   exchange SYMBOL (exchange name)
--   quantity DOUBLE (order quantity)
--   price DOUBLE (order price, nullable for market orders)
--   filled_quantity DOUBLE (filled quantity)
--   avg_fill_price DOUBLE (average fill price, nullable)
--   fee DOUBLE (trading fee, nullable)
--   latency_ms INT (execution latency in milliseconds)

-- Example line protocol format:
-- order_executions,order_id=ord_001,user_id=user_123,symbol=BTCUSD,side=BUY,type=MARKET,status=FILLED,exchange=binance quantity=0.1,filled_quantity=0.1,avg_fill_price=45050.0,fee=2.25,latency_ms=150 1640995200000000

-- ============================================================================
-- PORTFOLIO SNAPSHOTS TABLE
-- Optimized for portfolio tracking and performance analysis
-- ============================================================================

-- Portfolio snapshots table structure
-- Table: portfolio_snapshots
-- Columns:
--   timestamp TIMESTAMP (designated timestamp column)
--   user_id SYMBOL (user ID, indexed)
--   bot_id SYMBOL (bot ID, indexed, nullable)
--   total_value DOUBLE (total portfolio value)
--   cash_balance DOUBLE (available cash)
--   unrealized_pnl DOUBLE (unrealized profit/loss)
--   realized_pnl DOUBLE (realized profit/loss)
--   total_positions INT (number of open positions)
--   daily_pnl DOUBLE (daily profit/loss)
--   drawdown DOUBLE (current drawdown percentage)

-- Example line protocol format:
-- portfolio_snapshots,user_id=user_123,bot_id=bot_456 total_value=10500.0,cash_balance=2500.0,unrealized_pnl=500.0,realized_pnl=1000.0,total_positions=3,daily_pnl=150.0,drawdown=0.05 1640995200000000

-- ============================================================================
-- RISK METRICS TABLE
-- Optimized for risk management and monitoring
-- ============================================================================

-- Risk metrics table structure
-- Table: risk_metrics
-- Columns:
--   timestamp TIMESTAMP (designated timestamp column)
--   user_id SYMBOL (user ID, indexed)
--   bot_id SYMBOL (bot ID, indexed, nullable)
--   symbol SYMBOL (trading symbol, indexed, nullable)
--   metric_type SYMBOL (risk metric type)
--   value DOUBLE (metric value)
--   threshold DOUBLE (risk threshold)
--   severity SYMBOL (LOW, MEDIUM, HIGH, CRITICAL)

-- Example line protocol format:
-- risk_metrics,user_id=user_123,bot_id=bot_456,symbol=BTCUSD,metric_type=position_risk,severity=MEDIUM value=0.15,threshold=0.20 1640995200000000

-- ============================================================================
-- INDEXING AND OPTIMIZATION NOTES
-- ============================================================================

-- QuestDB automatically creates indexes for:
-- 1. SYMBOL columns (tags) - used for filtering and grouping
-- 2. Designated timestamp columns - used for time-based queries
-- 3. Partition keys - for efficient data organization

-- Optimization recommendations:
-- 1. Use SYMBOL type for frequently filtered columns (low cardinality)
-- 2. Use STRING type for high cardinality text data
-- 3. Partition tables by time for better query performance
-- 4. Use appropriate data types (DOUBLE for numbers, INT for integers)
-- 5. Leverage QuestDB's columnar storage for analytical queries

-- ============================================================================
-- SAMPLE QUERIES FOR PERFORMANCE TESTING
-- ============================================================================

-- Query 1: Get latest metrics for specific names
-- SELECT * FROM metrics WHERE name IN ('cpu_usage', 'memory_usage') AND timestamp > dateadd('h', -1, now()) ORDER BY timestamp DESC;

-- Query 2: Get trading signals for a symbol in the last 24 hours
-- SELECT * FROM trading_signals WHERE symbol = 'BTCUSD' AND timestamp > dateadd('d', -1, now()) ORDER BY timestamp DESC;

-- Query 3: Aggregate performance metrics by hour
-- SELECT timestamp, avg(value) as avg_value FROM performance_metrics WHERE metric = 'response_time' AND timestamp > dateadd('d', -7, now()) SAMPLE BY 1h;

-- Query 4: Get ML prediction accuracy over time
-- SELECT timestamp, symbol, confidence_scores FROM ml_predictions WHERE model_id = 'transformer_v1' AND timestamp > dateadd('d', -30, now()) ORDER BY timestamp DESC;

-- Query 5: Portfolio performance analysis
-- SELECT timestamp, total_value, daily_pnl, drawdown FROM portfolio_snapshots WHERE user_id = 'user_123' AND timestamp > dateadd('d', -30, now()) ORDER BY timestamp ASC;

-- ============================================================================
-- MIGRATION CONSIDERATIONS
-- ============================================================================

-- 1. Data Migration Strategy:
--    - Migrate historical data in batches
--    - Use parallel processing for large datasets
--    - Validate data integrity after migration

-- 2. Dual-Write Period:
--    - Write to both PostgreSQL and QuestDB during transition
--    - Compare query results for validation
--    - Gradually shift read traffic to QuestDB

-- 3. Rollback Plan:
--    - Keep PostgreSQL as backup during initial phase
--    - Monitor QuestDB performance and stability
--    - Have procedures to switch back if needed

-- 4. Performance Monitoring:
--    - Track query execution times
--    - Monitor ingestion rates
--    - Set up alerts for performance degradation
