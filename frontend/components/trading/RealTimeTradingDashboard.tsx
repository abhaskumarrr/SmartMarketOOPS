/**
 * Real-Time Trading Dashboard
 * Task #30: Real-Time Trading Dashboard
 * Main dashboard component integrating all real-time trading features
 */

'use client';

import React, { useEffect, useState } from 'react';
import { useTradingStore } from '../../lib/stores/tradingStore';
import { useAuthStore } from '../../lib/stores/authStore';
import { RealTimePriceChart } from './RealTimePriceChart';
import { SignalQualityIndicator } from './SignalQualityIndicator';
import { RealTimePortfolioMonitor } from './RealTimePortfolioMonitor';
import { TradingSignalHistory } from './TradingSignalHistory';
import { MLIntelligenceDashboard } from '../intelligence/MLIntelligenceDashboard';

interface RealTimeTradingDashboardProps {
  defaultSymbol?: string;
}

export const RealTimeTradingDashboard: React.FC<RealTimeTradingDashboardProps> = ({
  defaultSymbol = 'BTCUSD'
}) => {
  const [selectedView, setSelectedView] = useState<'overview' | 'detailed' | 'intelligence'>('overview');
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Store hooks
  const {
    selectedSymbol,
    setSelectedSymbol,
    isConnected,
    connectionStatus,
    initializeWebSocket,
    disconnectWebSocket,
    cleanup,
    settings,
    updateSettings,
    requestMLIntelligence,
    mlIntelligence
  } = useTradingStore();

  const { isAuthenticated, enableDemoMode } = useAuthStore();

  // Auto-enable demo mode for dashboard access
  useEffect(() => {
    if (!isAuthenticated) {
      enableDemoMode();
    }
  }, [isAuthenticated, enableDemoMode]);

  // Initialize WebSocket connection when authenticated
  useEffect(() => {
    if (isAuthenticated) {
      initializeWebSocket();
    } else {
      disconnectWebSocket();
    }

    // Cleanup on unmount
    return () => {
      disconnectWebSocket();
    };
  }, [isAuthenticated, initializeWebSocket, disconnectWebSocket]);

  // Set default symbol
  useEffect(() => {
    if (selectedSymbol !== defaultSymbol) {
      setSelectedSymbol(defaultSymbol);
    }
  }, [defaultSymbol, selectedSymbol, setSelectedSymbol]);

  // Auto cleanup old data
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      cleanup();
    }, 5 * 60 * 1000); // Cleanup every 5 minutes

    return () => clearInterval(interval);
  }, [autoRefresh, cleanup]);

  // Symbol options
  const symbolOptions = [
    'BTCUSD', 'ETHUSD', 'ADAUSD', 'SOLUSD', 'DOTUSD',
    'LINKUSD', 'MATICUSD', 'AVAXUSD', 'ATOMUSD', 'NEARUSD'
  ];

  // Connection status component
  const ConnectionStatus = () => (
    <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm ${
      isConnected
        ? 'bg-green-100 text-green-800'
        : connectionStatus.status === 'reconnecting'
        ? 'bg-yellow-100 text-yellow-800'
        : 'bg-red-100 text-red-800'
    }`}>
      <div className={`w-2 h-2 rounded-full ${
        isConnected
          ? 'bg-green-500 animate-pulse'
          : connectionStatus.status === 'reconnecting'
          ? 'bg-yellow-500 animate-pulse'
          : 'bg-red-500'
      }`}></div>
      <span className="font-medium">
        {isConnected ? 'Live' : connectionStatus.status === 'reconnecting' ? 'Reconnecting' : 'Disconnected'}
      </span>
    </div>
  );

  // Dashboard controls
  const DashboardControls = () => (
    <div className="flex flex-wrap items-center gap-4">
      {/* Symbol Selector */}
      <div className="flex items-center space-x-2">
        <label className="text-sm font-medium text-gray-700">Symbol:</label>
        <select
          value={selectedSymbol}
          onChange={(e) => setSelectedSymbol(e.target.value)}
          className="border border-gray-300 rounded px-3 py-1 text-sm bg-white focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        >
          {symbolOptions.map(symbol => (
            <option key={symbol} value={symbol}>{symbol}</option>
          ))}
        </select>
      </div>

      {/* View Toggle */}
      <div className="flex items-center space-x-1 bg-gray-100 rounded p-1">
        <button
          onClick={() => setSelectedView('overview')}
          className={`px-3 py-1 text-sm rounded transition-colors ${
            selectedView === 'overview'
              ? 'bg-white text-gray-900 shadow-sm'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          Overview
        </button>
        <button
          onClick={() => setSelectedView('detailed')}
          className={`px-3 py-1 text-sm rounded transition-colors ${
            selectedView === 'detailed'
              ? 'bg-white text-gray-900 shadow-sm'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          Detailed
        </button>
        <button
          onClick={() => setSelectedView('intelligence')}
          className={`px-3 py-1 text-sm rounded transition-colors ${
            selectedView === 'intelligence'
              ? 'bg-white text-gray-900 shadow-sm'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          ML Intelligence
        </button>
      </div>

      {/* Settings Toggle */}
      <div className="flex items-center space-x-2">
        <label className="text-sm font-medium text-gray-700">Real-time Signals:</label>
        <button
          onClick={() => updateSettings({ enableRealTimeSignals: !settings.enableRealTimeSignals })}
          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
            settings.enableRealTimeSignals ? 'bg-blue-600' : 'bg-gray-200'
          }`}
        >
          <span
            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
              settings.enableRealTimeSignals ? 'translate-x-6' : 'translate-x-1'
            }`}
          />
        </button>
      </div>

      {/* Auto Refresh Toggle */}
      <div className="flex items-center space-x-2">
        <label className="text-sm font-medium text-gray-700">Auto Refresh:</label>
        <button
          onClick={() => setAutoRefresh(!autoRefresh)}
          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
            autoRefresh ? 'bg-blue-600' : 'bg-gray-200'
          }`}
        >
          <span
            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
              autoRefresh ? 'translate-x-6' : 'translate-x-1'
            }`}
          />
        </button>
      </div>
    </div>
  );

  // Show loading state while initializing demo mode
  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          </div>
          <h2 className="text-xl font-semibold text-gray-900 mb-2">Loading Dashboard</h2>
          <p className="text-gray-600">Initializing SmartMarketOOPS Real-Time Trading Dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Real-Time Trading Dashboard</h1>
              <p className="text-sm text-gray-600">Enhanced with Transformer ML Intelligence</p>
            </div>
            <ConnectionStatus />
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
        <DashboardControls />
      </div>

      {/* Dashboard Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-8">
        {selectedView === 'overview' ? (
          /* Overview Layout */
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Main Chart - Takes 2 columns */}
            <div className="lg:col-span-2">
              <RealTimePriceChart
                symbol={selectedSymbol}
                height={500}
                showSignals={settings.enableRealTimeSignals}
                showVolume={true}
              />
            </div>

            {/* Right Sidebar */}
            <div className="space-y-6">
              <SignalQualityIndicator
                symbol={selectedSymbol}
                showDetails={true}
              />
              <RealTimePortfolioMonitor compact={true} />
            </div>

            {/* Bottom Row */}
            <div className="lg:col-span-2">
              <TradingSignalHistory
                maxSignals={20}
                showFilters={true}
                compact={false}
              />
            </div>

            <div>
              <RealTimePortfolioMonitor
                showPositions={true}
                compact={false}
              />
            </div>
          </div>
        ) : selectedView === 'detailed' ? (
          /* Detailed Layout */
          <div className="space-y-6">
            {/* Top Row - Charts */}
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              <RealTimePriceChart
                symbol={selectedSymbol}
                height={400}
                showSignals={settings.enableRealTimeSignals}
                showVolume={true}
              />
              <SignalQualityIndicator
                symbol={selectedSymbol}
                showDetails={true}
              />
            </div>

            {/* Middle Row - Portfolio and Performance */}
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              <RealTimePortfolioMonitor
                showPositions={true}
                compact={false}
              />
              <TradingSignalHistory
                maxSignals={30}
                showFilters={true}
                compact={false}
              />
            </div>
          </div>
        ) : (
          /* ML Intelligence Layout */
          <div className="space-y-6">
            {/* ML Intelligence Dashboard */}
            <MLIntelligenceDashboard
              symbol={selectedSymbol}
              autoRefresh={autoRefresh}
            />

            {/* Supporting Components */}
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              <RealTimePriceChart
                symbol={selectedSymbol}
                height={300}
                showSignals={settings.enableRealTimeSignals}
                showVolume={false}
              />
              <div className="space-y-4">
                <SignalQualityIndicator
                  symbol={selectedSymbol}
                  showDetails={false}
                  compact={true}
                />
                <RealTimePortfolioMonitor
                  showPositions={false}
                  compact={true}
                />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
