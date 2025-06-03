/**
 * Real-Time Portfolio Monitor Component
 * Task #30: Real-Time Trading Dashboard
 * Displays live portfolio performance, P&L, and position tracking
 */

'use client';

import React, { useMemo } from 'react';
import { useTradingStore } from '../../lib/stores/tradingStore';

interface RealTimePortfolioMonitorProps {
  showPositions?: boolean;
  compact?: boolean;
}

interface PortfolioSummary {
  totalValue: number;
  totalPnL: number;
  totalPnLPercent: number;
  dayChange: number;
  dayChangePercent: number;
  positionCount: number;
  profitablePositions: number;
}

export const RealTimePortfolioMonitor: React.FC<RealTimePortfolioMonitorProps> = ({
  showPositions = true,
  compact = false
}) => {
  const portfolio = useTradingStore((state) => state.portfolio);
  const isConnected = useTradingStore((state) => state.isConnected);
  const performanceMetrics = useTradingStore((state) => state.performanceMetrics);

  // Calculate portfolio summary
  const portfolioSummary: PortfolioSummary = useMemo(() => {
    const positions = Object.values(portfolio.positions);
    const positionCount = positions.length;
    const profitablePositions = positions.filter(pos => pos.pnl > 0).length;

    // For demo purposes, calculate day change as a percentage of total P&L
    const dayChange = portfolio.totalPnL * 0.1; // Simulate daily change
    const dayChangePercent = portfolio.totalValue > 0 ? (dayChange / portfolio.totalValue) * 100 : 0;

    return {
      totalValue: portfolio.totalValue,
      totalPnL: portfolio.totalPnL,
      totalPnLPercent: portfolio.totalPnLPercent,
      dayChange,
      dayChangePercent,
      positionCount,
      profitablePositions
    };
  }, [portfolio]);

  // Format currency
  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(amount);
  };

  // Format percentage
  const formatPercentage = (percent: number) => {
    const sign = percent >= 0 ? '+' : '';
    return `${sign}${percent.toFixed(2)}%`;
  };

  // Get color for P&L values
  const getPnLColor = (value: number) => {
    if (value > 0) return 'text-green-600';
    if (value < 0) return 'text-red-600';
    return 'text-gray-600';
  };

  // Compact view
  if (compact) {
    return (
      <div className="bg-white rounded-lg shadow-sm border p-3">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-lg font-semibold text-gray-900">
              {formatCurrency(portfolioSummary.totalValue)}
            </div>
            <div className="text-sm text-gray-500">Portfolio Value</div>
          </div>
          <div className="text-right">
            <div className={`text-sm font-medium ${getPnLColor(portfolioSummary.totalPnL)}`}>
              {formatCurrency(portfolioSummary.totalPnL)}
            </div>
            <div className={`text-xs ${getPnLColor(portfolioSummary.totalPnLPercent)}`}>
              {formatPercentage(portfolioSummary.totalPnLPercent)}
            </div>
          </div>
          {!isConnected && (
            <div className="w-2 h-2 bg-red-500 rounded-full ml-2" title="Disconnected"></div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900">Portfolio Monitor</h3>
        <div className={`flex items-center space-x-2 ${isConnected ? 'text-green-600' : 'text-red-600'}`}>
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></div>
          <span className="text-sm">{isConnected ? 'Live' : 'Disconnected'}</span>
        </div>
      </div>

      {/* Portfolio Summary */}
      <div className="p-4">
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          {/* Total Value */}
          <div className="text-center p-3 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold text-gray-900">
              {formatCurrency(portfolioSummary.totalValue)}
            </div>
            <div className="text-sm text-gray-500">Total Value</div>
          </div>

          {/* Total P&L */}
          <div className="text-center p-3 bg-gray-50 rounded-lg">
            <div className={`text-2xl font-bold ${getPnLColor(portfolioSummary.totalPnL)}`}>
              {formatCurrency(portfolioSummary.totalPnL)}
            </div>
            <div className={`text-sm ${getPnLColor(portfolioSummary.totalPnLPercent)}`}>
              {formatPercentage(portfolioSummary.totalPnLPercent)}
            </div>
          </div>

          {/* Day Change */}
          <div className="text-center p-3 bg-gray-50 rounded-lg">
            <div className={`text-2xl font-bold ${getPnLColor(portfolioSummary.dayChange)}`}>
              {formatCurrency(portfolioSummary.dayChange)}
            </div>
            <div className={`text-sm ${getPnLColor(portfolioSummary.dayChangePercent)}`}>
              {formatPercentage(portfolioSummary.dayChangePercent)} Today
            </div>
          </div>

          {/* Positions */}
          <div className="text-center p-3 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold text-gray-900">
              {portfolioSummary.positionCount}
            </div>
            <div className="text-sm text-gray-500">
              {portfolioSummary.profitablePositions} Profitable
            </div>
          </div>
        </div>

        {/* Performance Metrics */}
        <div className="grid grid-cols-3 gap-4 mb-6 p-3 bg-blue-50 rounded-lg">
          <div className="text-center">
            <div className="text-lg font-semibold text-blue-900">
              {performanceMetrics.winRate.toFixed(1)}%
            </div>
            <div className="text-sm text-blue-700">Win Rate</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-semibold text-blue-900">
              {performanceMetrics.totalSignals}
            </div>
            <div className="text-sm text-blue-700">Total Signals</div>
          </div>
          <div className="text-center">
            <div className={`text-lg font-semibold ${getPnLColor(performanceMetrics.totalReturn)}`}>
              {formatPercentage(performanceMetrics.totalReturn)}
            </div>
            <div className="text-sm text-blue-700">Total Return</div>
          </div>
        </div>

        {/* Positions Table */}
        {showPositions && Object.keys(portfolio.positions).length > 0 && (
          <div>
            <h4 className="text-md font-medium text-gray-900 mb-3">Active Positions</h4>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Symbol
                    </th>
                    <th className="px-3 py-2 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Amount
                    </th>
                    <th className="px-3 py-2 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Avg Price
                    </th>
                    <th className="px-3 py-2 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Current Price
                    </th>
                    <th className="px-3 py-2 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                      P&L
                    </th>
                    <th className="px-3 py-2 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                      P&L %
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {Object.values(portfolio.positions).map((position) => (
                    <tr key={position.symbol} className="hover:bg-gray-50">
                      <td className="px-3 py-2 whitespace-nowrap text-sm font-medium text-gray-900">
                        {position.symbol}
                      </td>
                      <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-900 text-right">
                        {position.amount.toFixed(6)}
                      </td>
                      <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-900 text-right">
                        {formatCurrency(position.averagePrice)}
                      </td>
                      <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-900 text-right">
                        {formatCurrency(position.currentPrice)}
                      </td>
                      <td className={`px-3 py-2 whitespace-nowrap text-sm text-right font-medium ${getPnLColor(position.pnl)}`}>
                        {formatCurrency(position.pnl)}
                      </td>
                      <td className={`px-3 py-2 whitespace-nowrap text-sm text-right font-medium ${getPnLColor(position.pnlPercent)}`}>
                        {formatPercentage(position.pnlPercent)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* No Positions State */}
        {showPositions && Object.keys(portfolio.positions).length === 0 && (
          <div className="text-center py-8 text-gray-500">
            <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-3">
              <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1" />
              </svg>
            </div>
            <p>No active positions</p>
            <p className="text-sm mt-1">Start trading to see your positions here</p>
          </div>
        )}
      </div>
    </div>
  );
};
