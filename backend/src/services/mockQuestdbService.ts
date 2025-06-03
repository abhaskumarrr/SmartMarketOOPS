/**
 * Mock QuestDB Service for Testing
 * Provides a mock implementation for testing without actual QuestDB
 */

import { logger } from '../utils/logger';

export interface MetricData {
  timestamp: Date | number;
  name: string;
  value: number;
  tags?: Record<string, string | number>;
}

export interface TradingSignalData {
  timestamp: Date | number;
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

export interface MLPredictionData {
  timestamp: Date | number;
  id: string;
  modelId: string;
  symbol: string;
  timeframe: string;
  predictionType: string;
  values: number[];
  confidenceScores: number[];
  metadata?: Record<string, any>;
}

export interface PerformanceMetricData {
  timestamp: Date | number;
  system: string;
  component: string;
  metric: string;
  unit: string;
  value: number;
  tags?: Record<string, string | number>;
}

export class MockQuestDBService {
  private static instance: MockQuestDBService;
  private isInitialized: boolean = false;
  private data: Map<string, any[]> = new Map();

  private constructor() {
    // Initialize mock data storage
    this.data.set('metrics', []);
    this.data.set('trading_signals', []);
    this.data.set('ml_predictions', []);
    this.data.set('performance_metrics', []);
  }

  public static getInstance(): MockQuestDBService {
    if (!MockQuestDBService.instance) {
      MockQuestDBService.instance = new MockQuestDBService();
    }
    return MockQuestDBService.instance;
  }

  public async initialize(): Promise<void> {
    try {
      logger.info('🔌 Initializing Mock QuestDB service...');
      this.isInitialized = true;
      logger.info('✅ Mock QuestDB service initialized successfully');
    } catch (error) {
      logger.error('❌ Failed to initialize Mock QuestDB service:', error);
      throw error;
    }
  }

  public async shutdown(): Promise<void> {
    try {
      this.isInitialized = false;
      this.data.clear();
      logger.info('🔌 Mock QuestDB service shutdown completed');
    } catch (error) {
      logger.error('❌ Error during Mock QuestDB service shutdown:', error);
      throw error;
    }
  }

  // Metric operations
  public async insertMetric(data: MetricData): Promise<void> {
    if (!this.isInitialized) {
      throw new Error('Mock QuestDB service not initialized');
    }

    try {
      const metrics = this.data.get('metrics') || [];
      metrics.push({
        ...data,
        timestamp: data.timestamp instanceof Date ? data.timestamp : new Date(data.timestamp),
        inserted_at: new Date(),
      });
      this.data.set('metrics', metrics);
      
      logger.debug(`📊 Mock: Inserted metric ${data.name} = ${data.value}`);
    } catch (error) {
      logger.error('Error inserting metric:', error);
      throw error;
    }
  }

  public async insertMetrics(metrics: MetricData[]): Promise<void> {
    for (const metric of metrics) {
      await this.insertMetric(metric);
    }
    logger.debug(`📊 Mock: Inserted ${metrics.length} metrics`);
  }

  // Trading Signal operations
  public async insertTradingSignal(data: TradingSignalData): Promise<void> {
    if (!this.isInitialized) {
      throw new Error('Mock QuestDB service not initialized');
    }

    try {
      const signals = this.data.get('trading_signals') || [];
      signals.push({
        ...data,
        timestamp: data.timestamp instanceof Date ? data.timestamp : new Date(data.timestamp),
        inserted_at: new Date(),
      });
      this.data.set('trading_signals', signals);
      
      logger.debug(`🎯 Mock: Inserted trading signal ${data.id} for ${data.symbol}`);
    } catch (error) {
      logger.error('Error inserting trading signal:', error);
      throw error;
    }
  }

  // ML Prediction operations
  public async insertMLPrediction(data: MLPredictionData): Promise<void> {
    if (!this.isInitialized) {
      throw new Error('Mock QuestDB service not initialized');
    }

    try {
      const predictions = this.data.get('ml_predictions') || [];
      predictions.push({
        ...data,
        timestamp: data.timestamp instanceof Date ? data.timestamp : new Date(data.timestamp),
        inserted_at: new Date(),
      });
      this.data.set('ml_predictions', predictions);
      
      logger.debug(`🤖 Mock: Inserted ML prediction ${data.id} for ${data.symbol}`);
    } catch (error) {
      logger.error('Error inserting ML prediction:', error);
      throw error;
    }
  }

  // Performance Metric operations
  public async insertPerformanceMetric(data: PerformanceMetricData): Promise<void> {
    if (!this.isInitialized) {
      throw new Error('Mock QuestDB service not initialized');
    }

    try {
      const metrics = this.data.get('performance_metrics') || [];
      metrics.push({
        ...data,
        timestamp: data.timestamp instanceof Date ? data.timestamp : new Date(data.timestamp),
        inserted_at: new Date(),
      });
      this.data.set('performance_metrics', metrics);
      
      logger.debug(`⚡ Mock: Inserted performance metric ${data.metric} = ${data.value}`);
    } catch (error) {
      logger.error('Error inserting performance metric:', error);
      throw error;
    }
  }

  // Query operations
  public async executeQuery(query: string): Promise<any[]> {
    logger.debug(`🔍 Mock: Executing query: ${query}`);
    
    // Simple mock query responses
    if (query.includes('count()')) {
      return [{ count: this.getTotalRecords() }];
    }
    
    if (query.includes('metrics')) {
      return this.data.get('metrics') || [];
    }
    
    if (query.includes('trading_signals')) {
      return this.data.get('trading_signals') || [];
    }
    
    if (query.includes('ml_predictions')) {
      return this.data.get('ml_predictions') || [];
    }
    
    if (query.includes('performance_metrics')) {
      return this.data.get('performance_metrics') || [];
    }
    
    return [];
  }

  // Health check and statistics
  public async healthCheck(): Promise<boolean> {
    return this.isInitialized;
  }

  public async getTableStats(tableName: string): Promise<any> {
    const data = this.data.get(tableName) || [];
    return { row_count: data.length };
  }

  public async flush(): Promise<void> {
    logger.debug('🔄 Mock: Flush called (no-op)');
  }

  public isReady(): boolean {
    return this.isInitialized;
  }

  // Mock-specific methods for testing
  public getStoredData(tableName: string): any[] {
    return this.data.get(tableName) || [];
  }

  public getTotalRecords(): number {
    let total = 0;
    for (const [, records] of this.data) {
      total += records.length;
    }
    return total;
  }

  public clearData(): void {
    this.data.clear();
    this.data.set('metrics', []);
    this.data.set('trading_signals', []);
    this.data.set('ml_predictions', []);
    this.data.set('performance_metrics', []);
    logger.info('🗑️ Mock: Cleared all data');
  }

  public getStats(): {
    isInitialized: boolean;
    totalRecords: number;
    tableStats: Record<string, number>;
  } {
    const tableStats: Record<string, number> = {};
    for (const [tableName, records] of this.data) {
      tableStats[tableName] = records.length;
    }

    return {
      isInitialized: this.isInitialized,
      totalRecords: this.getTotalRecords(),
      tableStats,
    };
  }
}

// Export singleton instance
export const mockQuestdbService = MockQuestDBService.getInstance();
