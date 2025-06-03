/**
 * ML Intelligence Service
 * Task #31: ML Trading Intelligence Integration
 * Advanced ML intelligence integration for real-time trading
 */

import { webSocketService } from './websocket';

export interface MLIntelligenceData {
  timestamp: string;
  symbol: string;
  signal: {
    id: string;
    signal_type: 'buy' | 'sell' | 'hold' | 'strong_buy' | 'strong_sell';
    confidence: number;
    quality: 'excellent' | 'good' | 'fair' | 'poor';
    price: number;
    transformer_prediction: number;
    ensemble_prediction: number;
    smc_score: number;
    technical_score: number;
    stop_loss?: number;
    take_profit?: number;
    position_size?: number;
    risk_reward_ratio?: number;
  };
  pipeline_prediction: number | number[];
  regime_analysis: {
    volatility_regime: 'low' | 'medium' | 'high';
    volatility_percentile: number;
    trend_regime: string;
    trend_strength: number;
    trend_direction: 'bullish' | 'bearish';
    volume_regime: 'low' | 'normal' | 'high';
    volume_ratio: number;
    market_condition: 'trending_stable' | 'trending_volatile' | 'consolidating' | 'choppy' | 'transitional';
  };
  risk_assessment: {
    var_95: number;
    var_99: number;
    maximum_adverse_excursion: number;
    kelly_fraction: number;
    risk_adjusted_position_size: number;
    risk_reward_ratio: number;
    confidence_adjusted_risk: number;
    risk_level: 'low' | 'medium' | 'high';
  };
  execution_strategy: {
    entry_method: 'market' | 'limit';
    exit_method: 'market' | 'limit';
    time_in_force: 'GTC' | 'IOC' | 'FOK';
    execution_urgency: 'urgent' | 'normal' | 'patient' | 'very_patient';
    entry_offset_pct?: number;
    partial_fill_allowed?: boolean;
    recommended_timing: 'immediate' | 'wait_for_volume' | 'normal';
    max_execution_time_minutes: number;
    slippage_tolerance_pct: number;
  };
  confidence_score: number;
  quality_rating: string;
  intelligence_version: string;
}

export interface MLPerformanceMetrics {
  overall_accuracy: number;
  transformer_accuracy: number;
  ensemble_accuracy: number;
  signal_quality_accuracy: number;
  prediction_latency_ms: number;
  throughput_predictions_per_second: number;
  memory_usage_gb: number;
  win_rate: number;
  profit_factor: number;
  sharpe_ratio: number;
  max_drawdown: number;
  model_confidence: number;
  prediction_consistency: number;
  error_rate: number;
  uptime_percentage: number;
  last_update: string;
}

export interface MLSystemStatus {
  is_initialized: boolean;
  is_running: boolean;
  active_predictions: number;
  cached_predictions: number;
  component_status: {
    transformer_pipeline: boolean;
    signal_quality_system: boolean;
    preprocessor: boolean;
  };
}

export interface MLIntelligenceSummary {
  system_status: MLSystemStatus;
  performance_metrics: MLPerformanceMetrics;
  configuration: any;
}

export class MLIntelligenceService {
  private baseUrl: string;
  private isConnected: boolean = false;
  private subscribers: Map<string, Set<(data: any) => void>> = new Map();
  private performanceCache: MLPerformanceMetrics | null = null;
  private lastIntelligenceUpdate: number = 0;

  constructor(baseUrl: string = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3002') {
    this.baseUrl = baseUrl;
    this.initializeWebSocketSubscriptions();
  }

  /**
   * Initialize WebSocket subscriptions for ML intelligence updates
   */
  private initializeWebSocketSubscriptions(): void {
    // Subscribe to ML intelligence updates
    webSocketService.subscribe('ml_intelligence', (data: MLIntelligenceData) => {
      this.handleIntelligenceUpdate(data);
    });

    // Subscribe to ML performance updates
    webSocketService.subscribe('ml_performance', (data: MLPerformanceMetrics) => {
      this.handlePerformanceUpdate(data);
    });

    // Subscribe to ML system status updates
    webSocketService.subscribe('ml_system_status', (data: MLSystemStatus) => {
      this.handleSystemStatusUpdate(data);
    });
  }

  /**
   * Request ML intelligence for a specific symbol
   */
  async requestIntelligence(
    symbol: string,
    marketData?: any,
    additionalContext?: any
  ): Promise<MLIntelligenceData | null> {
    try {
      const response = await fetch(`${this.baseUrl}/api/ml/intelligence`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`
        },
        body: JSON.stringify({
          symbol,
          market_data: marketData,
          additional_context: additionalContext
        })
      });

      if (!response.ok) {
        throw new Error(`ML Intelligence request failed: ${response.statusText}`);
      }

      const intelligence: MLIntelligenceData = await response.json();
      this.handleIntelligenceUpdate(intelligence);

      return intelligence;
    } catch (error) {
      console.error('Error requesting ML intelligence:', error);
      return null;
    }
  }

  /**
   * Get ML system performance metrics
   */
  async getPerformanceMetrics(): Promise<MLPerformanceMetrics | null> {
    try {
      const response = await fetch(`${this.baseUrl}/api/ml/performance`, {
        headers: {
          'Authorization': `Bearer ${this.getAuthToken()}`
        }
      });

      if (!response.ok) {
        throw new Error(`Performance metrics request failed: ${response.statusText}`);
      }

      const metrics: MLPerformanceMetrics = await response.json();
      this.performanceCache = metrics;

      return metrics;
    } catch (error) {
      console.error('Error getting performance metrics:', error);
      return this.performanceCache;
    }
  }

  /**
   * Get comprehensive ML intelligence summary
   */
  async getIntelligenceSummary(): Promise<MLIntelligenceSummary | null> {
    try {
      const response = await fetch(`${this.baseUrl}/api/ml/summary`, {
        headers: {
          'Authorization': `Bearer ${this.getAuthToken()}`
        }
      });

      if (!response.ok) {
        throw new Error(`Intelligence summary request failed: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error getting intelligence summary:', error);
      return null;
    }
  }

  /**
   * Subscribe to ML intelligence updates
   */
  subscribe(eventType: string, callback: (data: any) => void): () => void {
    if (!this.subscribers.has(eventType)) {
      this.subscribers.set(eventType, new Set());
    }

    this.subscribers.get(eventType)!.add(callback);

    // Return unsubscribe function
    return () => {
      const subscribers = this.subscribers.get(eventType);
      if (subscribers) {
        subscribers.delete(callback);
        if (subscribers.size === 0) {
          this.subscribers.delete(eventType);
        }
      }
    };
  }

  /**
   * Handle ML intelligence updates
   */
  private handleIntelligenceUpdate(data: MLIntelligenceData): void {
    this.lastIntelligenceUpdate = Date.now();
    this.notifySubscribers('intelligence_update', data);

    // Also notify symbol-specific subscribers
    this.notifySubscribers(`intelligence_${data.symbol}`, data);
  }

  /**
   * Handle ML performance updates
   */
  private handlePerformanceUpdate(data: MLPerformanceMetrics): void {
    this.performanceCache = data;
    this.notifySubscribers('performance_update', data);
  }

  /**
   * Handle ML system status updates
   */
  private handleSystemStatusUpdate(data: MLSystemStatus): void {
    this.isConnected = data.is_running;
    this.notifySubscribers('system_status_update', data);
  }

  /**
   * Notify subscribers of events
   */
  private notifySubscribers(eventType: string, data: any): void {
    const subscribers = this.subscribers.get(eventType);
    if (subscribers) {
      subscribers.forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error('Error in ML intelligence subscriber:', error);
        }
      });
    }
  }

  /**
   * Get authentication token
   */
  private getAuthToken(): string {
    // This would integrate with your auth system
    return localStorage.getItem('auth_token') || '';
  }

  /**
   * Get cached performance metrics
   */
  getCachedPerformanceMetrics(): MLPerformanceMetrics | null {
    return this.performanceCache;
  }

  /**
   * Check if ML intelligence is connected and running
   */
  isMLIntelligenceConnected(): boolean {
    return this.isConnected;
  }

  /**
   * Get time since last intelligence update
   */
  getTimeSinceLastUpdate(): number {
    return Date.now() - this.lastIntelligenceUpdate;
  }

  /**
   * Validate ML intelligence data
   */
  validateIntelligenceData(data: any): data is MLIntelligenceData {
    return (
      data &&
      typeof data.timestamp === 'string' &&
      typeof data.symbol === 'string' &&
      data.signal &&
      typeof data.signal.confidence === 'number' &&
      data.regime_analysis &&
      data.risk_assessment &&
      data.execution_strategy
    );
  }

  /**
   * Format intelligence data for display
   */
  formatIntelligenceForDisplay(data: MLIntelligenceData): any {
    return {
      signal: {
        type: data.signal.signal_type.toUpperCase(),
        confidence: `${(data.signal.confidence * 100).toFixed(1)}%`,
        quality: data.signal.quality.toUpperCase(),
        price: `$${data.signal.price.toFixed(2)}`
      },
      regime: {
        condition: data.regime_analysis.market_condition.replace('_', ' ').toUpperCase(),
        volatility: data.regime_analysis.volatility_regime.toUpperCase(),
        trend: `${data.regime_analysis.trend_direction.toUpperCase()} (${data.regime_analysis.trend_strength.toFixed(3)})`
      },
      risk: {
        level: data.risk_assessment.risk_level.toUpperCase(),
        position_size: `${(data.risk_assessment.risk_adjusted_position_size * 100).toFixed(1)}%`,
        risk_reward: data.risk_assessment.risk_reward_ratio.toFixed(1),
        var_95: `${(data.risk_assessment.var_95 * 100).toFixed(2)}%`
      },
      execution: {
        method: data.execution_strategy.entry_method.toUpperCase(),
        urgency: data.execution_strategy.execution_urgency.replace('_', ' ').toUpperCase(),
        timing: data.execution_strategy.recommended_timing.replace('_', ' ').toUpperCase(),
        max_time: `${data.execution_strategy.max_execution_time_minutes}min`
      }
    };
  }

  /**
   * Get intelligence quality score
   */
  getIntelligenceQualityScore(data: MLIntelligenceData): number {
    const weights = {
      confidence: 0.3,
      quality: 0.2,
      regime_clarity: 0.2,
      risk_assessment: 0.15,
      execution_strategy: 0.15
    };

    const qualityScores = {
      excellent: 1.0,
      good: 0.8,
      fair: 0.6,
      poor: 0.4
    };

    const confidenceScore = data.signal.confidence;
    const qualityScore = qualityScores[data.signal.quality] || 0.5;
    const regimeScore = data.regime_analysis.trend_strength > 0.05 ? 1.0 : 0.5;
    const riskScore = data.risk_assessment.risk_level === 'low' ? 1.0 :
                     data.risk_assessment.risk_level === 'medium' ? 0.7 : 0.4;
    const executionScore = data.execution_strategy.execution_urgency === 'urgent' ? 1.0 : 0.8;

    return (
      confidenceScore * weights.confidence +
      qualityScore * weights.quality +
      regimeScore * weights.regime_clarity +
      riskScore * weights.risk_assessment +
      executionScore * weights.execution_strategy
    );
  }

  /**
   * Get performance summary
   */
  getPerformanceSummary(): any {
    if (!this.performanceCache) {
      return null;
    }

    const metrics = this.performanceCache;

    return {
      overall: {
        accuracy: `${(metrics.overall_accuracy * 100).toFixed(1)}%`,
        win_rate: `${(metrics.win_rate * 100).toFixed(1)}%`,
        profit_factor: metrics.profit_factor.toFixed(2),
        sharpe_ratio: metrics.sharpe_ratio.toFixed(2)
      },
      performance: {
        latency: `${metrics.prediction_latency_ms.toFixed(0)}ms`,
        throughput: `${metrics.throughput_predictions_per_second.toFixed(1)}/s`,
        memory: `${metrics.memory_usage_gb.toFixed(2)}GB`,
        uptime: `${metrics.uptime_percentage.toFixed(1)}%`
      },
      models: {
        transformer: `${(metrics.transformer_accuracy * 100).toFixed(1)}%`,
        ensemble: `${(metrics.ensemble_accuracy * 100).toFixed(1)}%`,
        signal_quality: `${(metrics.signal_quality_accuracy * 100).toFixed(1)}%`,
        confidence: `${(metrics.model_confidence * 100).toFixed(1)}%`
      },
      health: {
        error_rate: `${(metrics.error_rate * 100).toFixed(2)}%`,
        consistency: `${(metrics.prediction_consistency * 100).toFixed(1)}%`,
        last_update: new Date(metrics.last_update).toLocaleString()
      }
    };
  }
}

// Singleton instance
export const mlIntelligenceService = new MLIntelligenceService();
