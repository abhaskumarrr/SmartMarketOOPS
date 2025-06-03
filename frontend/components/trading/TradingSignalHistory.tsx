/**
 * Trading Signal History Component
 * Task #30: Real-Time Trading Dashboard
 * Displays real-time trading signal history with performance tracking
 */

'use client';

import React, { useState, useMemo } from 'react';
import { useTradingStore } from '../../lib/stores/tradingStore';

interface TradingSignalHistoryProps {
  maxSignals?: number;
  showFilters?: boolean;
  compact?: boolean;
}

type SignalFilter = 'all' | 'buy' | 'sell' | 'excellent' | 'good';

export const TradingSignalHistory: React.FC<TradingSignalHistoryProps> = ({
  maxSignals = 50,
  showFilters = true,
  compact = false
}) => {
  const [filter, setFilter] = useState<SignalFilter>('all');
  const [selectedSymbol, setSelectedSymbol] = useState<string>('all');

  const tradingSignals = useTradingStore((state) => state.tradingSignals);
  const isConnected = useTradingStore((state) => state.isConnected);

  // Get unique symbols
  const symbols = useMemo(() => {
    const uniqueSymbols = [...new Set(tradingSignals.map(signal => signal.symbol))];
    return ['all', ...uniqueSymbols];
  }, [tradingSignals]);

  // Filter and sort signals
  const filteredSignals = useMemo(() => {
    let filtered = tradingSignals;

    // Filter by symbol
    if (selectedSymbol !== 'all') {
      filtered = filtered.filter(signal => signal.symbol === selectedSymbol);
    }

    // Filter by type/quality
    switch (filter) {
      case 'buy':
        filtered = filtered.filter(signal => 
          signal.signal_type === 'buy' || signal.signal_type === 'strong_buy'
        );
        break;
      case 'sell':
        filtered = filtered.filter(signal => 
          signal.signal_type === 'sell' || signal.signal_type === 'strong_sell'
        );
        break;
      case 'excellent':
        filtered = filtered.filter(signal => signal.quality === 'excellent');
        break;
      case 'good':
        filtered = filtered.filter(signal => 
          signal.quality === 'excellent' || signal.quality === 'good'
        );
        break;
    }

    // Sort by timestamp (newest first) and limit
    return filtered
      .sort((a, b) => b.timestamp - a.timestamp)
      .slice(0, maxSignals);
  }, [tradingSignals, filter, selectedSymbol, maxSignals]);

  // Signal type styling
  const getSignalTypeStyle = (signalType: string) => {
    switch (signalType) {
      case 'strong_buy':
        return 'bg-green-600 text-white';
      case 'buy':
        return 'bg-green-100 text-green-800';
      case 'strong_sell':
        return 'bg-red-600 text-white';
      case 'sell':
        return 'bg-red-100 text-red-800';
      case 'hold':
        return 'bg-gray-100 text-gray-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  // Quality styling
  const getQualityStyle = (quality: string) => {
    switch (quality) {
      case 'excellent':
        return 'bg-green-100 text-green-800';
      case 'good':
        return 'bg-blue-100 text-blue-800';
      case 'fair':
        return 'bg-yellow-100 text-yellow-800';
      case 'poor':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  // Format time
  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  // Compact view
  if (compact) {
    return (
      <div className="bg-white rounded-lg shadow-sm border p-3">
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-md font-medium text-gray-900">Recent Signals</h4>
          <span className="text-sm text-gray-500">{filteredSignals.length}</span>
        </div>
        
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {filteredSignals.slice(0, 5).map((signal) => (
            <div key={signal.id} className="flex items-center justify-between p-2 bg-gray-50 rounded">
              <div className="flex items-center space-x-2">
                <span className={`px-2 py-1 rounded text-xs font-medium ${getSignalTypeStyle(signal.signal_type)}`}>
                  {signal.signal_type.toUpperCase()}
                </span>
                <span className="text-sm font-medium">{signal.symbol}</span>
              </div>
              <div className="text-right">
                <div className="text-sm font-medium">${signal.price.toFixed(2)}</div>
                <div className="text-xs text-gray-500">{formatTime(signal.timestamp)}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900">Trading Signals</h3>
        <div className="flex items-center space-x-2">
          <span className="text-sm text-gray-500">{filteredSignals.length} signals</span>
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></div>
        </div>
      </div>

      {/* Filters */}
      {showFilters && (
        <div className="p-4 border-b border-gray-200 bg-gray-50">
          <div className="flex flex-wrap items-center gap-3">
            {/* Symbol Filter */}
            <select
              value={selectedSymbol}
              onChange={(e) => setSelectedSymbol(e.target.value)}
              className="text-sm border border-gray-300 rounded px-2 py-1 bg-white"
            >
              {symbols.map(symbol => (
                <option key={symbol} value={symbol}>
                  {symbol === 'all' ? 'All Symbols' : symbol}
                </option>
              ))}
            </select>

            {/* Type/Quality Filter */}
            <div className="flex space-x-1">
              {(['all', 'buy', 'sell', 'excellent', 'good'] as SignalFilter[]).map((filterType) => (
                <button
                  key={filterType}
                  onClick={() => setFilter(filterType)}
                  className={`px-3 py-1 text-sm rounded transition-colors ${
                    filter === filterType
                      ? 'bg-blue-600 text-white'
                      : 'bg-white text-gray-600 hover:bg-gray-100 border border-gray-300'
                  }`}
                >
                  {filterType.charAt(0).toUpperCase() + filterType.slice(1)}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Signals List */}
      <div className="max-h-96 overflow-y-auto">
        {filteredSignals.length > 0 ? (
          <div className="divide-y divide-gray-200">
            {filteredSignals.map((signal) => (
              <div key={signal.id} className="p-4 hover:bg-gray-50 transition-colors">
                <div className="flex items-start justify-between">
                  {/* Signal Info */}
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${getSignalTypeStyle(signal.signal_type)}`}>
                        {signal.signal_type.replace('_', ' ').toUpperCase()}
                      </span>
                      <span className={`px-2 py-1 rounded text-xs font-medium ${getQualityStyle(signal.quality)}`}>
                        {signal.quality.toUpperCase()}
                      </span>
                      <span className="text-sm font-medium text-gray-900">{signal.symbol}</span>
                    </div>

                    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 text-sm">
                      <div>
                        <span className="text-gray-500">Price:</span>
                        <span className="font-medium ml-1">${signal.price.toFixed(2)}</span>
                      </div>
                      <div>
                        <span className="text-gray-500">Confidence:</span>
                        <span className="font-medium ml-1">{(signal.confidence * 100).toFixed(0)}%</span>
                      </div>
                      {signal.stop_loss && (
                        <div>
                          <span className="text-gray-500">Stop Loss:</span>
                          <span className="font-medium ml-1 text-red-600">${signal.stop_loss.toFixed(2)}</span>
                        </div>
                      )}
                      {signal.take_profit && (
                        <div>
                          <span className="text-gray-500">Take Profit:</span>
                          <span className="font-medium ml-1 text-green-600">${signal.take_profit.toFixed(2)}</span>
                        </div>
                      )}
                    </div>

                    {/* Component Scores */}
                    <div className="mt-2 grid grid-cols-4 gap-2 text-xs">
                      <div className="text-center p-1 bg-blue-50 rounded">
                        <div className="font-medium text-blue-900">{(signal.transformer_prediction * 100).toFixed(0)}%</div>
                        <div className="text-blue-700">Transformer</div>
                      </div>
                      <div className="text-center p-1 bg-purple-50 rounded">
                        <div className="font-medium text-purple-900">{(signal.ensemble_prediction * 100).toFixed(0)}%</div>
                        <div className="text-purple-700">Ensemble</div>
                      </div>
                      <div className="text-center p-1 bg-green-50 rounded">
                        <div className="font-medium text-green-900">{(signal.smc_score * 100).toFixed(0)}%</div>
                        <div className="text-green-700">SMC</div>
                      </div>
                      <div className="text-center p-1 bg-orange-50 rounded">
                        <div className="font-medium text-orange-900">{(signal.technical_score * 100).toFixed(0)}%</div>
                        <div className="text-orange-700">Technical</div>
                      </div>
                    </div>
                  </div>

                  {/* Timestamp */}
                  <div className="text-right text-sm text-gray-500 ml-4">
                    <div>{formatTime(signal.timestamp)}</div>
                    <div className="text-xs">{new Date(signal.timestamp).toLocaleTimeString()}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-3">
              <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <p>No trading signals found</p>
            <p className="text-sm mt-1">
              {filter !== 'all' || selectedSymbol !== 'all' 
                ? 'Try adjusting your filters' 
                : 'Signals will appear here when generated'
              }
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
