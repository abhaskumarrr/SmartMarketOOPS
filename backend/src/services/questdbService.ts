/**
 * QuestDB Service
 * High-performance time-series database service for SmartMarketOOPS
 * Provides optimized operations for financial time-series data
 */

import { Sender } from '@questdb/nodejs-client';
import { questdbConnection } from '../config/questdb';
import { logger } from '../utils/logger';

export interface TimeSeriesDataPoint {
  timestamp: Date | number;
  symbol?: string;
  value?: number;
  tags?: Record<string, string | number>;
  fields?: Record<string, number | string | boolean>;
}

export interface QueryOptions {
  limit?: number;
  offset?: number;
  orderBy?: string;
  orderDirection?: 'ASC' | 'DESC';
  where?: string;
  groupBy?: string;
  having?: string;
}

export interface MetricData extends TimeSeriesDataPoint {
  name: string;
  value: number;
  tags?: Record<string, string | number>;
}

export interface TradingSignalData extends TimeSeriesDataPoint {
  id: string;
  symbol: string;
  type: string;
  direction: string;
  strength: string;
  timeframe: string;
  price: number;
  targetPrice?: number;
  stopLoss?: number;
  confidenceScore: number;
  expectedReturn: number;
  expectedRisk: number;
  riskRewardRatio: number;
  source: string;
  metadata?: Record<string, any>;
}

export interface MLPredictionData extends TimeSeriesDataPoint {
  id: string;
  modelId: string;
  symbol: string;
  timeframe: string;
  predictionType: string;
  values: number[];
  confidenceScores: number[];
  metadata?: Record<string, any>;
}

export interface PerformanceMetricData extends TimeSeriesDataPoint {
  system: string;
  component: string;
  metric: string;
  value: number;
  unit: string;
  tags?: Record<string, string | number>;
}

export class QuestDBService {
  private static instance: QuestDBService;
  private client: Sender | null = null;

  private constructor() {}

  public static getInstance(): QuestDBService {
    if (!QuestDBService.instance) {
      QuestDBService.instance = new QuestDBService();
    }
    return QuestDBService.instance;
  }

  public async initialize(): Promise<void> {
    try {
      await questdbConnection.connect();
      this.client = questdbConnection.getClient();
      logger.info('✅ QuestDB service initialized successfully');
    } catch (error) {
      logger.error('❌ Failed to initialize QuestDB service:', error);
      throw error;
    }
  }

  public async shutdown(): Promise<void> {
    try {
      await questdbConnection.disconnect();
      this.client = null;
      logger.info('🔌 QuestDB service shutdown completed');
    } catch (error) {
      logger.error('❌ Error during QuestDB service shutdown:', error);
      throw error;
    }
  }

  private ensureClient(): Sender {
    if (!this.client || !questdbConnection.isReady()) {
      throw new Error('QuestDB service not initialized. Call initialize() first.');
    }
    return this.client;
  }

  // Metric operations
  public async insertMetric(data: MetricData): Promise<void> {
    const client = this.ensureClient();

    try {
      // Use the Sender API to build the line protocol
      client
        .table('metrics')
        .symbol('name', data.name);

      // Add tags as symbols
      if (data.tags) {
        for (const [key, value] of Object.entries(data.tags)) {
          client.symbol(key, String(value));
        }
      }

      // Add value as float column
      client.floatColumn('value', data.value);

      // Set timestamp and send (don't await here, let auto-flush handle it)
      if (data.timestamp instanceof Date) {
        client.at(data.timestamp.getTime() * 1000000); // Convert to nanoseconds
      } else {
        client.atNow();
      }

    } catch (error) {
      logger.error('Error inserting metric:', error);
      throw error;
    }
  }

  public async insertMetrics(metrics: MetricData[]): Promise<void> {
    const client = this.ensureClient();
    
    try {
      for (const metric of metrics) {
        await this.insertMetric(metric);
      }
      await client.flush();
      
    } catch (error) {
      logger.error('Error inserting metrics batch:', error);
      throw error;
    }
  }

  // Trading Signal operations
  public async insertTradingSignal(data: TradingSignalData): Promise<void> {
    const client = this.ensureClient();

    try {
      client
        .table('trading_signals')
        .symbol('id', data.id)
        .symbol('symbol', data.symbol)
        .symbol('type', data.type)
        .symbol('direction', data.direction)
        .symbol('strength', data.strength)
        .symbol('timeframe', data.timeframe)
        .symbol('source', data.source)
        .floatColumn('price', data.price)
        .floatColumn('confidence_score', data.confidenceScore)
        .floatColumn('expected_return', data.expectedReturn)
        .floatColumn('expected_risk', data.expectedRisk)
        .floatColumn('risk_reward_ratio', data.riskRewardRatio);

      if (data.targetPrice !== undefined) {
        client.floatColumn('target_price', data.targetPrice);
      }

      if (data.stopLoss !== undefined) {
        client.floatColumn('stop_loss', data.stopLoss);
      }

      if (data.timestamp instanceof Date) {
        client.at(data.timestamp.getTime() * 1000000);
      } else {
        client.atNow();
      }

    } catch (error) {
      logger.error('Error inserting trading signal:', error);
      throw error;
    }
  }

  // ML Prediction operations
  public async insertMLPrediction(data: MLPredictionData): Promise<void> {
    const client = this.ensureClient();

    try {
      // Convert arrays to JSON strings for storage
      const valuesJson = JSON.stringify(data.values);
      const confidenceJson = JSON.stringify(data.confidenceScores);

      client
        .table('ml_predictions')
        .symbol('id', data.id)
        .symbol('model_id', data.modelId)
        .symbol('symbol', data.symbol)
        .symbol('timeframe', data.timeframe)
        .symbol('prediction_type', data.predictionType)
        .stringColumn('values', valuesJson)
        .stringColumn('confidence_scores', confidenceJson);

      if (data.timestamp instanceof Date) {
        client.at(data.timestamp.getTime() * 1000000);
      } else {
        client.atNow();
      }

    } catch (error) {
      logger.error('Error inserting ML prediction:', error);
      throw error;
    }
  }

  // Performance Metric operations
  public async insertPerformanceMetric(data: PerformanceMetricData): Promise<void> {
    const client = this.ensureClient();

    try {
      client
        .table('performance_metrics')
        .symbol('system', data.system)
        .symbol('component', data.component)
        .symbol('metric', data.metric)
        .symbol('unit', data.unit);

      // Add tags as symbols
      if (data.tags) {
        for (const [key, value] of Object.entries(data.tags)) {
          client.symbol(key, String(value));
        }
      }

      client.floatColumn('value', data.value);

      if (data.timestamp instanceof Date) {
        client.at(data.timestamp.getTime() * 1000000);
      } else {
        client.atNow();
      }

    } catch (error) {
      logger.error('Error inserting performance metric:', error);
      throw error;
    }
  }

  // Market Data operations
  public async insertMarketData(data: {
    timestamp: Date;
    symbol: string;
    exchange: string;
    timeframe: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
  }): Promise<void> {
    const client = this.ensureClient();

    try {
      client
        .table('market_data')
        .symbol('symbol', data.symbol)
        .symbol('exchange', data.exchange)
        .symbol('timeframe', data.timeframe)
        .floatColumn('open', data.open)
        .floatColumn('high', data.high)
        .floatColumn('low', data.low)
        .floatColumn('close', data.close)
        .floatColumn('volume', data.volume);

      client.at(data.timestamp.getTime() * 1000000);

    } catch (error) {
      logger.error('Error inserting market data:', error);
      throw error;
    }
  }

  public async insertMarketDataBatch(dataPoints: Array<{
    timestamp: Date;
    symbol: string;
    exchange: string;
    timeframe: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
  }>): Promise<void> {
    const client = this.ensureClient();

    try {
      for (const data of dataPoints) {
        client
          .table('market_data')
          .symbol('symbol', data.symbol)
          .symbol('exchange', data.exchange)
          .symbol('timeframe', data.timeframe)
          .floatColumn('open', data.open)
          .floatColumn('high', data.high)
          .floatColumn('low', data.low)
          .floatColumn('close', data.close)
          .floatColumn('volume', data.volume);

        client.at(data.timestamp.getTime() * 1000000);
      }

      await client.flush();

    } catch (error) {
      logger.error('Error inserting market data batch:', error);
      throw error;
    }
  }

  // Trade operations
  public async insertTrade(data: {
    timestamp: Date;
    id: string;
    symbol: string;
    side: string;
    entryPrice: number;
    exitPrice: number;
    quantity: number;
    entryTime: Date;
    exitTime: Date;
    pnl: number;
    pnlPercent: number;
    commission: number;
    strategy: string;
    reason: string;
    duration: number;
  }): Promise<void> {
    const client = this.ensureClient();

    try {
      client
        .table('trades')
        .symbol('id', data.id)
        .symbol('symbol', data.symbol)
        .symbol('side', data.side)
        .symbol('strategy', data.strategy)
        .symbol('reason', data.reason)
        .floatColumn('entry_price', data.entryPrice)
        .floatColumn('exit_price', data.exitPrice)
        .floatColumn('quantity', data.quantity)
        .floatColumn('pnl', data.pnl)
        .floatColumn('pnl_percent', data.pnlPercent)
        .floatColumn('commission', data.commission)
        .floatColumn('duration', data.duration)
        .timestampColumn('entry_time', data.entryTime.getTime() * 1000000)
        .timestampColumn('exit_time', data.exitTime.getTime() * 1000000);

      client.at(data.timestamp.getTime() * 1000000);

    } catch (error) {
      logger.error('Error inserting trade:', error);
      throw error;
    }
  }

  // Portfolio operations
  public async insertPortfolioSnapshot(data: {
    timestamp: Date;
    totalValue: number;
    cash: number;
    totalPnl: number;
    totalPnlPercent: number;
    drawdown: number;
    maxDrawdown: number;
    leverage: number;
    positionCount: number;
  }): Promise<void> {
    const client = this.ensureClient();

    try {
      client
        .table('portfolio_snapshots')
        .floatColumn('total_value', data.totalValue)
        .floatColumn('cash', data.cash)
        .floatColumn('total_pnl', data.totalPnl)
        .floatColumn('total_pnl_percent', data.totalPnlPercent)
        .floatColumn('drawdown', data.drawdown)
        .floatColumn('max_drawdown', data.maxDrawdown)
        .floatColumn('leverage', data.leverage)
        .floatColumn('position_count', data.positionCount);

      client.at(data.timestamp.getTime() * 1000000);

    } catch (error) {
      logger.error('Error inserting portfolio snapshot:', error);
      throw error;
    }
  }

  // Batch operations for high-performance ingestion
  public async batchInsert(
    tableName: string,
    data: TimeSeriesDataPoint[],
    formatter: (item: TimeSeriesDataPoint) => string
  ): Promise<void> {
    const client = this.ensureClient();

    try {
      // For batch operations, we'll use individual insertions
      // The formatter function is not used with the new Sender API
      for (const item of data) {
        // This is a generic method, so we'll handle it based on the item structure
        if ('name' in item && 'value' in item) {
          // Treat as metric
          await this.insertMetric(item as MetricData);
        }
      }
      await client.flush();

    } catch (error) {
      logger.error(`Error in batch insert for ${tableName}:`, error);
      throw error;
    }
  }

  // Query operations (using HTTP API for complex queries)
  public async executeQuery(query: string): Promise<any[]> {
    try {
      // For complex queries, we'll use HTTP API
      const response = await fetch(`http://${questdbConnection.getConfig().host}:9000/exec?query=${encodeURIComponent(query)}`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`Query failed: ${response.statusText}`);
      }

      const result = await response.json();
      return result.dataset || [];
      
    } catch (error) {
      logger.error('Error executing query:', error);
      throw error;
    }
  }

  // Optimized time-series queries
  public async getMetricsByTimeRange(
    metricName: string,
    startTime: Date,
    endTime: Date,
    options?: QueryOptions
  ): Promise<any[]> {
    const startTs = startTime.toISOString();
    const endTs = endTime.toISOString();
    
    let query = `SELECT * FROM metrics WHERE name = '${metricName}' AND timestamp >= '${startTs}' AND timestamp <= '${endTs}'`;
    
    if (options?.orderBy) {
      query += ` ORDER BY ${options.orderBy} ${options.orderDirection || 'ASC'}`;
    } else {
      query += ` ORDER BY timestamp ASC`;
    }
    
    if (options?.limit) {
      query += ` LIMIT ${options.limit}`;
    }
    
    return this.executeQuery(query);
  }

  public async getTradingSignalsBySymbol(
    symbol: string,
    startTime: Date,
    endTime: Date,
    options?: QueryOptions
  ): Promise<any[]> {
    const startTs = startTime.toISOString();
    const endTs = endTime.toISOString();
    
    let query = `SELECT * FROM trading_signals WHERE symbol = '${symbol}' AND timestamp >= '${startTs}' AND timestamp <= '${endTs}'`;
    
    if (options?.orderBy) {
      query += ` ORDER BY ${options.orderBy} ${options.orderDirection || 'ASC'}`;
    } else {
      query += ` ORDER BY timestamp DESC`;
    }
    
    if (options?.limit) {
      query += ` LIMIT ${options.limit}`;
    }
    
    return this.executeQuery(query);
  }

  public async getLatestMetrics(metricNames: string[], limit: number = 100): Promise<any[]> {
    const namesStr = metricNames.map(name => `'${name}'`).join(',');
    const query = `SELECT * FROM metrics WHERE name IN (${namesStr}) ORDER BY timestamp DESC LIMIT ${limit}`;
    
    return this.executeQuery(query);
  }

  // Aggregation queries for analytics
  public async getMetricAggregation(
    metricName: string,
    aggregation: 'AVG' | 'SUM' | 'MIN' | 'MAX' | 'COUNT',
    interval: string,
    startTime: Date,
    endTime: Date
  ): Promise<any[]> {
    const startTs = startTime.toISOString();
    const endTs = endTime.toISOString();
    
    const query = `
      SELECT 
        timestamp,
        ${aggregation}(value) as ${aggregation.toLowerCase()}_value
      FROM metrics 
      WHERE name = '${metricName}' 
        AND timestamp >= '${startTs}' 
        AND timestamp <= '${endTs}'
      SAMPLE BY ${interval}
      ORDER BY timestamp ASC
    `;
    
    return this.executeQuery(query);
  }

  // Health check and statistics
  public async healthCheck(): Promise<boolean> {
    try {
      await questdbConnection.healthCheck();
      return true;
    } catch (error) {
      logger.error('QuestDB service health check failed:', error);
      return false;
    }
  }

  public async getTableStats(tableName: string): Promise<any> {
    const query = `SELECT count() as row_count FROM ${tableName}`;
    const result = await this.executeQuery(query);
    return result[0] || { row_count: 0 };
  }

  // Utility methods
  public async flush(): Promise<void> {
    const client = this.ensureClient();
    await client.flush();
  }

  public isReady(): boolean {
    return questdbConnection.isReady() && this.client !== null;
  }
}

// Export singleton instance
export const questdbService = QuestDBService.getInstance();
