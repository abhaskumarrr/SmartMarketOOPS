/**
 * ML Intelligence Dashboard Component
 * Task #31: ML Trading Intelligence Integration
 * Advanced ML intelligence visualization and control center
 */

'use client';

import React, { useState, useEffect, useMemo } from 'react';
import { mlIntelligenceService, type MLIntelligenceData, type MLPerformanceMetrics } from '../../lib/services/mlIntelligenceService';
import { useTradingStore } from '../../lib/stores/tradingStore';

interface MLIntelligenceDashboardProps {
  symbol?: string;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

export const MLIntelligenceDashboard: React.FC<MLIntelligenceDashboardProps> = ({
  symbol,
  autoRefresh = true,
  refreshInterval = 30000 // 30 seconds
}) => {
  const [intelligenceData, setIntelligenceData] = useState<MLIntelligenceData | null>(null);
  const [performanceMetrics, setPerformanceMetrics] = useState<MLPerformanceMetrics | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedTab, setSelectedTab] = useState<'overview' | 'performance' | 'analysis' | 'execution'>('overview');

  const { selectedSymbol, marketData } = useTradingStore();
  const currentSymbol = symbol || selectedSymbol;

  // Subscribe to ML intelligence updates
  useEffect(() => {
    const unsubscribeIntelligence = mlIntelligenceService.subscribe('intelligence_update', (data: MLIntelligenceData) => {
      if (!currentSymbol || data.symbol === currentSymbol) {
        setIntelligenceData(data);
        setError(null);
      }
    });

    const unsubscribePerformance = mlIntelligenceService.subscribe('performance_update', (data: MLPerformanceMetrics) => {
      setPerformanceMetrics(data);
    });

    return () => {
      unsubscribeIntelligence();
      unsubscribePerformance();
    };
  }, [currentSymbol]);

  // Auto-refresh intelligence data
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(async () => {
      await requestIntelligence();
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval, currentSymbol]);

  // Initial data load
  useEffect(() => {
    requestIntelligence();
    loadPerformanceMetrics();
  }, [currentSymbol]);

  const requestIntelligence = async () => {
    if (!currentSymbol) return;

    setIsLoading(true);
    setError(null);

    try {
      const data = await mlIntelligenceService.requestIntelligence(
        currentSymbol,
        marketData[currentSymbol]
      );

      if (data) {
        setIntelligenceData(data);
      } else {
        setError('Failed to get ML intelligence data');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const loadPerformanceMetrics = async () => {
    try {
      const metrics = await mlIntelligenceService.getPerformanceMetrics();
      if (metrics) {
        setPerformanceMetrics(metrics);
      }
    } catch (err) {
      console.error('Failed to load performance metrics:', err);
    }
  };

  // Format intelligence data for display
  const formattedIntelligence = useMemo(() => {
    if (!intelligenceData) return null;
    return mlIntelligenceService.formatIntelligenceForDisplay(intelligenceData);
  }, [intelligenceData]);

  // Calculate intelligence quality score
  const qualityScore = useMemo(() => {
    if (!intelligenceData) return 0;
    return mlIntelligenceService.getIntelligenceQualityScore(intelligenceData);
  }, [intelligenceData]);

  // Performance summary
  const performanceSummary = useMemo(() => {
    return mlIntelligenceService.getPerformanceSummary();
  }, [performanceMetrics]);

  // Get quality color
  const getQualityColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600 bg-green-100';
    if (score >= 0.6) return 'text-blue-600 bg-blue-100';
    if (score >= 0.4) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  // Tab navigation
  const TabButton: React.FC<{ tab: string; label: string; isActive: boolean }> = ({ tab, label, isActive }) => (
    <button
      onClick={() => setSelectedTab(tab as any)}
      className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
        isActive
          ? 'bg-blue-600 text-white'
          : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
      }`}
    >
      {label}
    </button>
  );

  return (
    <div className="bg-white rounded-lg shadow-sm border">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">ML Intelligence Center</h3>
          <p className="text-sm text-gray-600">
            Advanced ML trading intelligence for {currentSymbol}
          </p>
        </div>
        
        <div className="flex items-center space-x-3">
          {/* Quality Score */}
          {intelligenceData && (
            <div className={`px-3 py-1 rounded-full text-sm font-medium ${getQualityColor(qualityScore)}`}>
              Quality: {(qualityScore * 100).toFixed(0)}%
            </div>
          )}
          
          {/* Connection Status */}
          <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm ${
            mlIntelligenceService.isMLIntelligenceConnected()
              ? 'bg-green-100 text-green-800'
              : 'bg-red-100 text-red-800'
          }`}>
            <div className={`w-2 h-2 rounded-full ${
              mlIntelligenceService.isMLIntelligenceConnected()
                ? 'bg-green-500 animate-pulse'
                : 'bg-red-500'
            }`}></div>
            <span>
              {mlIntelligenceService.isMLIntelligenceConnected() ? 'ML Active' : 'ML Offline'}
            </span>
          </div>
          
          {/* Refresh Button */}
          <button
            onClick={requestIntelligence}
            disabled={isLoading}
            className="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 disabled:opacity-50"
          >
            {isLoading ? 'Loading...' : 'Refresh'}
          </button>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex space-x-1 p-4 border-b border-gray-200 bg-gray-50">
        <TabButton tab="overview" label="Overview" isActive={selectedTab === 'overview'} />
        <TabButton tab="performance" label="Performance" isActive={selectedTab === 'performance'} />
        <TabButton tab="analysis" label="Analysis" isActive={selectedTab === 'analysis'} />
        <TabButton tab="execution" label="Execution" isActive={selectedTab === 'execution'} />
      </div>

      {/* Content */}
      <div className="p-4">
        {error && (
          <div className="mb-4 p-3 bg-red-100 border border-red-300 text-red-700 rounded">
            {error}
          </div>
        )}

        {selectedTab === 'overview' && (
          <OverviewTab 
            intelligenceData={intelligenceData}
            formattedIntelligence={formattedIntelligence}
            performanceSummary={performanceSummary}
            isLoading={isLoading}
          />
        )}

        {selectedTab === 'performance' && (
          <PerformanceTab 
            performanceMetrics={performanceMetrics}
            performanceSummary={performanceSummary}
          />
        )}

        {selectedTab === 'analysis' && (
          <AnalysisTab 
            intelligenceData={intelligenceData}
            formattedIntelligence={formattedIntelligence}
          />
        )}

        {selectedTab === 'execution' && (
          <ExecutionTab 
            intelligenceData={intelligenceData}
            formattedIntelligence={formattedIntelligence}
          />
        )}
      </div>
    </div>
  );
};

// Overview Tab Component
const OverviewTab: React.FC<{
  intelligenceData: MLIntelligenceData | null;
  formattedIntelligence: any;
  performanceSummary: any;
  isLoading: boolean;
}> = ({ intelligenceData, formattedIntelligence, performanceSummary, isLoading }) => {
  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2"></div>
          <p className="text-gray-600">Generating ML intelligence...</p>
        </div>
      </div>
    );
  }

  if (!intelligenceData || !formattedIntelligence) {
    return (
      <div className="text-center py-8 text-gray-500">
        <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-3">
          <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
          </svg>
        </div>
        <p>No ML intelligence data available</p>
        <p className="text-sm mt-1">Click refresh to generate intelligence</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Signal Summary */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-blue-50 p-4 rounded-lg">
          <div className="text-sm text-blue-600 font-medium">Signal Type</div>
          <div className="text-xl font-bold text-blue-900">{formattedIntelligence.signal.type}</div>
          <div className="text-sm text-blue-700">{formattedIntelligence.signal.confidence} confidence</div>
        </div>
        
        <div className="bg-green-50 p-4 rounded-lg">
          <div className="text-sm text-green-600 font-medium">Quality</div>
          <div className="text-xl font-bold text-green-900">{formattedIntelligence.signal.quality}</div>
          <div className="text-sm text-green-700">At {formattedIntelligence.signal.price}</div>
        </div>
        
        <div className="bg-purple-50 p-4 rounded-lg">
          <div className="text-sm text-purple-600 font-medium">Market Regime</div>
          <div className="text-xl font-bold text-purple-900">{formattedIntelligence.regime.condition}</div>
          <div className="text-sm text-purple-700">{formattedIntelligence.regime.volatility} volatility</div>
        </div>
        
        <div className="bg-orange-50 p-4 rounded-lg">
          <div className="text-sm text-orange-600 font-medium">Risk Level</div>
          <div className="text-xl font-bold text-orange-900">{formattedIntelligence.risk.level}</div>
          <div className="text-sm text-orange-700">{formattedIntelligence.risk.position_size} position</div>
        </div>
      </div>

      {/* Performance Summary */}
      {performanceSummary && (
        <div className="bg-gray-50 p-4 rounded-lg">
          <h4 className="text-md font-medium text-gray-900 mb-3">ML Performance Summary</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-gray-600">Accuracy:</span>
              <span className="font-medium ml-1">{performanceSummary.overall.accuracy}</span>
            </div>
            <div>
              <span className="text-gray-600">Win Rate:</span>
              <span className="font-medium ml-1">{performanceSummary.overall.win_rate}</span>
            </div>
            <div>
              <span className="text-gray-600">Latency:</span>
              <span className="font-medium ml-1">{performanceSummary.performance.latency}</span>
            </div>
            <div>
              <span className="text-gray-600">Memory:</span>
              <span className="font-medium ml-1">{performanceSummary.performance.memory}</span>
            </div>
          </div>
        </div>
      )}

      {/* Component Scores */}
      <div className="bg-white border rounded-lg p-4">
        <h4 className="text-md font-medium text-gray-900 mb-3">ML Component Analysis</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-3 bg-blue-50 rounded">
            <div className="text-lg font-bold text-blue-900">
              {(intelligenceData.signal.transformer_prediction * 100).toFixed(0)}%
            </div>
            <div className="text-sm text-blue-700">Transformer</div>
          </div>
          <div className="text-center p-3 bg-purple-50 rounded">
            <div className="text-lg font-bold text-purple-900">
              {(intelligenceData.signal.ensemble_prediction * 100).toFixed(0)}%
            </div>
            <div className="text-sm text-purple-700">Ensemble</div>
          </div>
          <div className="text-center p-3 bg-green-50 rounded">
            <div className="text-lg font-bold text-green-900">
              {(intelligenceData.signal.smc_score * 100).toFixed(0)}%
            </div>
            <div className="text-sm text-green-700">SMC</div>
          </div>
          <div className="text-center p-3 bg-orange-50 rounded">
            <div className="text-lg font-bold text-orange-900">
              {(intelligenceData.signal.technical_score * 100).toFixed(0)}%
            </div>
            <div className="text-sm text-orange-700">Technical</div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Performance Tab Component
const PerformanceTab: React.FC<{
  performanceMetrics: MLPerformanceMetrics | null;
  performanceSummary: any;
}> = ({ performanceMetrics, performanceSummary }) => {
  if (!performanceMetrics || !performanceSummary) {
    return (
      <div className="text-center py-8 text-gray-500">
        <p>Performance metrics not available</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Overall Performance */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-green-50 p-4 rounded-lg">
          <div className="text-sm text-green-600 font-medium">Overall Accuracy</div>
          <div className="text-2xl font-bold text-green-900">{performanceSummary.overall.accuracy}</div>
          <div className="text-sm text-green-700">Target: 75%</div>
        </div>
        
        <div className="bg-blue-50 p-4 rounded-lg">
          <div className="text-sm text-blue-600 font-medium">Win Rate</div>
          <div className="text-2xl font-bold text-blue-900">{performanceSummary.overall.win_rate}</div>
          <div className="text-sm text-blue-700">Target: 70%</div>
        </div>
        
        <div className="bg-purple-50 p-4 rounded-lg">
          <div className="text-sm text-purple-600 font-medium">Profit Factor</div>
          <div className="text-2xl font-bold text-purple-900">{performanceSummary.overall.profit_factor}</div>
          <div className="text-sm text-purple-700">Target: &gt;1.5</div>
        </div>
        
        <div className="bg-orange-50 p-4 rounded-lg">
          <div className="text-sm text-orange-600 font-medium">Sharpe Ratio</div>
          <div className="text-2xl font-bold text-orange-900">{performanceSummary.overall.sharpe_ratio}</div>
          <div className="text-sm text-orange-700">Target: &gt;1.0</div>
        </div>
      </div>

      {/* System Performance */}
      <div className="bg-gray-50 p-4 rounded-lg">
        <h4 className="text-md font-medium text-gray-900 mb-3">System Performance</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <span className="text-gray-600">Latency:</span>
            <span className="font-medium ml-1">{performanceSummary.performance.latency}</span>
          </div>
          <div>
            <span className="text-gray-600">Throughput:</span>
            <span className="font-medium ml-1">{performanceSummary.performance.throughput}</span>
          </div>
          <div>
            <span className="text-gray-600">Memory:</span>
            <span className="font-medium ml-1">{performanceSummary.performance.memory}</span>
          </div>
          <div>
            <span className="text-gray-600">Uptime:</span>
            <span className="font-medium ml-1">{performanceSummary.performance.uptime}</span>
          </div>
        </div>
      </div>

      {/* Model Performance */}
      <div className="bg-white border rounded-lg p-4">
        <h4 className="text-md font-medium text-gray-900 mb-3">Model Performance Breakdown</h4>
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Transformer Model</span>
            <span className="font-medium">{performanceSummary.models.transformer}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Ensemble Models</span>
            <span className="font-medium">{performanceSummary.models.ensemble}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Signal Quality</span>
            <span className="font-medium">{performanceSummary.models.signal_quality}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Model Confidence</span>
            <span className="font-medium">{performanceSummary.models.confidence}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

// Analysis Tab Component
const AnalysisTab: React.FC<{
  intelligenceData: MLIntelligenceData | null;
  formattedIntelligence: any;
}> = ({ intelligenceData, formattedIntelligence }) => {
  if (!intelligenceData) {
    return (
      <div className="text-center py-8 text-gray-500">
        <p>No analysis data available</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Market Regime Analysis */}
      <div className="bg-white border rounded-lg p-4">
        <h4 className="text-md font-medium text-gray-900 mb-3">Market Regime Analysis</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <div className="text-sm text-gray-600">Market Condition</div>
            <div className="text-lg font-medium">{formattedIntelligence.regime.condition}</div>
          </div>
          <div>
            <div className="text-sm text-gray-600">Volatility Regime</div>
            <div className="text-lg font-medium">{formattedIntelligence.regime.volatility}</div>
          </div>
          <div>
            <div className="text-sm text-gray-600">Trend Analysis</div>
            <div className="text-lg font-medium">{formattedIntelligence.regime.trend}</div>
          </div>
          <div>
            <div className="text-sm text-gray-600">Volume Regime</div>
            <div className="text-lg font-medium">{intelligenceData.regime_analysis.volume_regime.toUpperCase()}</div>
          </div>
        </div>
      </div>

      {/* Risk Assessment */}
      <div className="bg-white border rounded-lg p-4">
        <h4 className="text-md font-medium text-gray-900 mb-3">Risk Assessment</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <div className="text-sm text-gray-600">Risk Level</div>
            <div className="text-lg font-medium">{formattedIntelligence.risk.level}</div>
          </div>
          <div>
            <div className="text-sm text-gray-600">Position Size</div>
            <div className="text-lg font-medium">{formattedIntelligence.risk.position_size}</div>
          </div>
          <div>
            <div className="text-sm text-gray-600">Risk/Reward</div>
            <div className="text-lg font-medium">{formattedIntelligence.risk.risk_reward}</div>
          </div>
          <div>
            <div className="text-sm text-gray-600">VaR (95%)</div>
            <div className="text-lg font-medium">{formattedIntelligence.risk.var_95}</div>
          </div>
        </div>
      </div>

      {/* Detailed Metrics */}
      <div className="bg-gray-50 p-4 rounded-lg">
        <h4 className="text-md font-medium text-gray-900 mb-3">Detailed Analysis</h4>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-600">Volatility Percentile:</span>
            <span className="font-medium">{(intelligenceData.regime_analysis.volatility_percentile * 100).toFixed(1)}%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Trend Strength:</span>
            <span className="font-medium">{(intelligenceData.regime_analysis.trend_strength * 100).toFixed(2)}%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Volume Ratio:</span>
            <span className="font-medium">{intelligenceData.regime_analysis.volume_ratio.toFixed(2)}x</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Kelly Fraction:</span>
            <span className="font-medium">{(intelligenceData.risk_assessment.kelly_fraction * 100).toFixed(1)}%</span>
          </div>
        </div>
      </div>
    </div>
  );
};

// Execution Tab Component
const ExecutionTab: React.FC<{
  intelligenceData: MLIntelligenceData | null;
  formattedIntelligence: any;
}> = ({ intelligenceData, formattedIntelligence }) => {
  if (!intelligenceData) {
    return (
      <div className="text-center py-8 text-gray-500">
        <p>No execution data available</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Execution Strategy */}
      <div className="bg-white border rounded-lg p-4">
        <h4 className="text-md font-medium text-gray-900 mb-3">Execution Strategy</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <div className="text-sm text-gray-600">Entry Method</div>
            <div className="text-lg font-medium">{formattedIntelligence.execution.method}</div>
          </div>
          <div>
            <div className="text-sm text-gray-600">Execution Urgency</div>
            <div className="text-lg font-medium">{formattedIntelligence.execution.urgency}</div>
          </div>
          <div>
            <div className="text-sm text-gray-600">Recommended Timing</div>
            <div className="text-lg font-medium">{formattedIntelligence.execution.timing}</div>
          </div>
          <div>
            <div className="text-sm text-gray-600">Max Execution Time</div>
            <div className="text-lg font-medium">{formattedIntelligence.execution.max_time}</div>
          </div>
        </div>
      </div>

      {/* Risk Management */}
      {intelligenceData.signal.stop_loss && intelligenceData.signal.take_profit && (
        <div className="bg-white border rounded-lg p-4">
          <h4 className="text-md font-medium text-gray-900 mb-3">Risk Management</h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center p-3 bg-red-50 rounded">
              <div className="text-lg font-bold text-red-900">
                ${intelligenceData.signal.stop_loss.toFixed(2)}
              </div>
              <div className="text-sm text-red-700">Stop Loss</div>
            </div>
            <div className="text-center p-3 bg-blue-50 rounded">
              <div className="text-lg font-bold text-blue-900">
                ${intelligenceData.signal.price.toFixed(2)}
              </div>
              <div className="text-sm text-blue-700">Entry Price</div>
            </div>
            <div className="text-center p-3 bg-green-50 rounded">
              <div className="text-lg font-bold text-green-900">
                ${intelligenceData.signal.take_profit.toFixed(2)}
              </div>
              <div className="text-sm text-green-700">Take Profit</div>
            </div>
          </div>
        </div>
      )}

      {/* Execution Parameters */}
      <div className="bg-gray-50 p-4 rounded-lg">
        <h4 className="text-md font-medium text-gray-900 mb-3">Execution Parameters</h4>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-600">Time in Force:</span>
            <span className="font-medium">{intelligenceData.execution_strategy.time_in_force}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Slippage Tolerance:</span>
            <span className="font-medium">{(intelligenceData.execution_strategy.slippage_tolerance_pct * 100).toFixed(2)}%</span>
          </div>
          {intelligenceData.execution_strategy.entry_offset_pct && (
            <div className="flex justify-between">
              <span className="text-gray-600">Entry Offset:</span>
              <span className="font-medium">{(intelligenceData.execution_strategy.entry_offset_pct * 100).toFixed(2)}%</span>
            </div>
          )}
          {intelligenceData.execution_strategy.partial_fill_allowed && (
            <div className="flex justify-between">
              <span className="text-gray-600">Partial Fills:</span>
              <span className="font-medium">Allowed</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
