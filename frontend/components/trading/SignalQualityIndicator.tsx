/**
 * Real-Time Signal Quality Indicator Component
 * Task #30: Real-Time Trading Dashboard
 * Displays live signal quality metrics and confidence scores
 */

'use client';

import React, { useMemo } from 'react';
import { useTradingStore } from '../../lib/stores/tradingStore';

interface SignalQualityIndicatorProps {
  symbol?: string;
  showDetails?: boolean;
  compact?: boolean;
}

interface QualityMetrics {
  overall: number;
  transformer: number;
  ensemble: number;
  smc: number;
  technical: number;
  confidence: number;
  quality: 'excellent' | 'good' | 'fair' | 'poor';
}

export const SignalQualityIndicator: React.FC<SignalQualityIndicatorProps> = ({
  symbol,
  showDetails = true,
  compact = false
}) => {
  // Get latest signal for symbol or overall latest
  const latestSignal = useTradingStore((state) => 
    symbol ? state.latestSignals[symbol] : 
    Object.values(state.latestSignals)[0]
  );
  
  const performanceMetrics = useTradingStore((state) => state.performanceMetrics);
  const isConnected = useTradingStore((state) => state.isConnected);

  // Calculate quality metrics
  const qualityMetrics: QualityMetrics | null = useMemo(() => {
    if (!latestSignal) return null;

    const transformer = latestSignal.transformer_prediction;
    const ensemble = latestSignal.ensemble_prediction;
    const smc = latestSignal.smc_score;
    const technical = latestSignal.technical_score;
    const confidence = latestSignal.confidence;

    // Calculate overall quality score
    const overall = (transformer * 0.4 + ensemble * 0.3 + smc * 0.15 + technical * 0.15);

    return {
      overall,
      transformer,
      ensemble,
      smc,
      technical,
      confidence,
      quality: latestSignal.quality
    };
  }, [latestSignal]);

  // Quality color mapping
  const getQualityColor = (quality: string) => {
    switch (quality) {
      case 'excellent': return 'text-green-600 bg-green-100';
      case 'good': return 'text-blue-600 bg-blue-100';
      case 'fair': return 'text-yellow-600 bg-yellow-100';
      case 'poor': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  // Score color based on value
  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-blue-600';
    if (score >= 0.4) return 'text-yellow-600';
    return 'text-red-600';
  };

  // Progress bar component
  const ProgressBar: React.FC<{ value: number; label: string; color?: string }> = ({ 
    value, 
    label, 
    color = 'bg-blue-500' 
  }) => (
    <div className="flex items-center space-x-2">
      <span className="text-xs font-medium text-gray-600 w-16 text-right">{label}:</span>
      <div className="flex-1 bg-gray-200 rounded-full h-2">
        <div 
          className={`h-2 rounded-full transition-all duration-300 ${color}`}
          style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }}
        />
      </div>
      <span className={`text-xs font-medium w-12 ${getScoreColor(value)}`}>
        {(value * 100).toFixed(0)}%
      </span>
    </div>
  );

  // Compact view
  if (compact) {
    if (!qualityMetrics) {
      return (
        <div className="flex items-center space-x-2 text-sm text-gray-500">
          <div className="w-2 h-2 bg-gray-400 rounded-full"></div>
          <span>No signal</span>
        </div>
      );
    }

    return (
      <div className="flex items-center space-x-2">
        <div className={`px-2 py-1 rounded text-xs font-medium ${getQualityColor(qualityMetrics.quality)}`}>
          {qualityMetrics.quality.toUpperCase()}
        </div>
        <span className={`text-sm font-medium ${getScoreColor(qualityMetrics.confidence)}`}>
          {(qualityMetrics.confidence * 100).toFixed(0)}%
        </span>
        {!isConnected && (
          <div className="w-2 h-2 bg-red-500 rounded-full" title="Disconnected"></div>
        )}
      </div>
    );
  }

  // No signal state
  if (!qualityMetrics) {
    return (
      <div className="bg-white rounded-lg shadow-sm border p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-lg font-semibold text-gray-900">Signal Quality</h3>
          <div className={`flex items-center space-x-2 ${isConnected ? 'text-green-600' : 'text-red-600'}`}>
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></div>
            <span className="text-sm">{isConnected ? 'Live' : 'Disconnected'}</span>
          </div>
        </div>
        
        <div className="text-center py-8 text-gray-500">
          <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-3">
            <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <p>Waiting for trading signals...</p>
          <p className="text-sm mt-1">Connect to start receiving real-time signals</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">Signal Quality</h3>
        <div className="flex items-center space-x-3">
          {symbol && (
            <span className="text-sm font-medium text-gray-600">{symbol}</span>
          )}
          <div className={`flex items-center space-x-2 ${isConnected ? 'text-green-600' : 'text-red-600'}`}>
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></div>
            <span className="text-sm">{isConnected ? 'Live' : 'Disconnected'}</span>
          </div>
        </div>
      </div>

      {/* Overall Quality Badge */}
      <div className="flex items-center justify-between mb-4">
        <div className={`px-3 py-2 rounded-lg text-sm font-medium ${getQualityColor(qualityMetrics.quality)}`}>
          {qualityMetrics.quality.toUpperCase()} QUALITY
        </div>
        <div className="text-right">
          <div className={`text-2xl font-bold ${getScoreColor(qualityMetrics.confidence)}`}>
            {(qualityMetrics.confidence * 100).toFixed(0)}%
          </div>
          <div className="text-xs text-gray-500">Confidence</div>
        </div>
      </div>

      {/* Signal Type and Timestamp */}
      {latestSignal && (
        <div className="flex items-center justify-between mb-4 p-3 bg-gray-50 rounded-lg">
          <div>
            <div className={`text-sm font-medium ${
              latestSignal.signal_type.includes('buy') ? 'text-green-600' : 
              latestSignal.signal_type.includes('sell') ? 'text-red-600' : 'text-gray-600'
            }`}>
              {latestSignal.signal_type.toUpperCase()} SIGNAL
            </div>
            <div className="text-xs text-gray-500">
              {new Date(latestSignal.timestamp).toLocaleString()}
            </div>
          </div>
          <div className="text-right">
            <div className="text-sm font-medium">${latestSignal.price.toFixed(2)}</div>
            {latestSignal.risk_reward_ratio && (
              <div className="text-xs text-gray-500">
                R:R {latestSignal.risk_reward_ratio.toFixed(1)}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Detailed Metrics */}
      {showDetails && (
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-gray-700 mb-2">Component Scores</h4>
          
          <ProgressBar 
            value={qualityMetrics.transformer} 
            label="Transformer" 
            color="bg-blue-500"
          />
          
          <ProgressBar 
            value={qualityMetrics.ensemble} 
            label="Ensemble" 
            color="bg-purple-500"
          />
          
          <ProgressBar 
            value={qualityMetrics.smc} 
            label="SMC" 
            color="bg-green-500"
          />
          
          <ProgressBar 
            value={qualityMetrics.technical} 
            label="Technical" 
            color="bg-orange-500"
          />

          {/* Risk Metrics */}
          {latestSignal && (latestSignal.stop_loss || latestSignal.take_profit) && (
            <div className="mt-4 pt-3 border-t border-gray-200">
              <h4 className="text-sm font-medium text-gray-700 mb-2">Risk Management</h4>
              <div className="grid grid-cols-2 gap-3 text-sm">
                {latestSignal.stop_loss && (
                  <div>
                    <span className="text-gray-600">Stop Loss:</span>
                    <span className="font-medium text-red-600 ml-1">
                      ${latestSignal.stop_loss.toFixed(2)}
                    </span>
                  </div>
                )}
                {latestSignal.take_profit && (
                  <div>
                    <span className="text-gray-600">Take Profit:</span>
                    <span className="font-medium text-green-600 ml-1">
                      ${latestSignal.take_profit.toFixed(2)}
                    </span>
                  </div>
                )}
                {latestSignal.position_size && (
                  <div>
                    <span className="text-gray-600">Position:</span>
                    <span className="font-medium ml-1">
                      {(latestSignal.position_size * 100).toFixed(1)}%
                    </span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Performance Summary */}
          <div className="mt-4 pt-3 border-t border-gray-200">
            <h4 className="text-sm font-medium text-gray-700 mb-2">Performance Summary</h4>
            <div className="grid grid-cols-3 gap-3 text-sm">
              <div className="text-center">
                <div className="font-medium text-gray-900">{performanceMetrics.totalSignals}</div>
                <div className="text-gray-500">Total Signals</div>
              </div>
              <div className="text-center">
                <div className={`font-medium ${getScoreColor(performanceMetrics.winRate / 100)}`}>
                  {performanceMetrics.winRate.toFixed(1)}%
                </div>
                <div className="text-gray-500">Win Rate</div>
              </div>
              <div className="text-center">
                <div className={`font-medium ${performanceMetrics.totalReturn >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {performanceMetrics.totalReturn >= 0 ? '+' : ''}{performanceMetrics.totalReturn.toFixed(1)}%
                </div>
                <div className="text-gray-500">Total Return</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
