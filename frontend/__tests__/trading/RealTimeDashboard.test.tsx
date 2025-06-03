/**
 * Real-Time Trading Dashboard Tests
 * Task #30: Real-Time Trading Dashboard
 * Comprehensive tests for real-time functionality and components
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import { RealTimeTradingDashboard } from '../../components/trading/RealTimeTradingDashboard';
import { SignalQualityIndicator } from '../../components/trading/SignalQualityIndicator';
import { RealTimePortfolioMonitor } from '../../components/trading/RealTimePortfolioMonitor';
import { TradingSignalHistory } from '../../components/trading/TradingSignalHistory';
import { useTradingStore } from '../../lib/stores/tradingStore';
import { useAuthStore } from '../../lib/stores/authStore';

// Mock the stores
jest.mock('../../lib/stores/tradingStore');
jest.mock('../../lib/stores/authStore');
jest.mock('../../lib/services/websocket');

// Mock Chart.js
jest.mock('react-chartjs-2', () => ({
  Line: ({ data, options }: any) => (
    <div data-testid="price-chart">
      <div data-testid="chart-data">{JSON.stringify(data)}</div>
      <div data-testid="chart-options">{JSON.stringify(options)}</div>
    </div>
  )
}));

const mockTradingStore = {
  selectedSymbol: 'BTCUSD',
  setSelectedSymbol: jest.fn(),
  isConnected: true,
  connectionStatus: { status: 'connected' },
  initializeWebSocket: jest.fn(),
  disconnectWebSocket: jest.fn(),
  cleanup: jest.fn(),
  settings: {
    enableRealTimeSignals: true,
    signalQualityThreshold: 'good'
  },
  updateSettings: jest.fn(),
  marketData: {
    BTCUSD: {
      symbol: 'BTCUSD',
      price: 50000,
      change: 1000,
      changePercent: 2.0,
      volume: 1000000,
      high24h: 51000,
      low24h: 49000,
      timestamp: Date.now()
    }
  },
  tradingSignals: [
    {
      id: '1',
      symbol: 'BTCUSD',
      signal_type: 'buy',
      confidence: 0.85,
      quality: 'excellent',
      price: 50000,
      timestamp: Date.now(),
      transformer_prediction: 0.8,
      ensemble_prediction: 0.75,
      smc_score: 0.9,
      technical_score: 0.7,
      stop_loss: 49000,
      take_profit: 52000,
      position_size: 0.1,
      risk_reward_ratio: 2.0
    }
  ],
  latestSignals: {},
  portfolio: {
    totalValue: 10000,
    totalPnL: 500,
    totalPnLPercent: 5.0,
    positions: {
      BTCUSD: {
        symbol: 'BTCUSD',
        amount: 0.1,
        averagePrice: 48000,
        currentPrice: 50000,
        pnl: 200,
        pnlPercent: 4.17
      }
    }
  },
  performanceMetrics: {
    totalSignals: 10,
    successfulSignals: 7,
    averageConfidence: 0.75,
    winRate: 70,
    totalReturn: 15.5
  }
};

const mockAuthStore = {
  isAuthenticated: true,
  user: { id: '1', email: 'test@example.com' },
  token: 'mock-token'
};

describe('RealTimeTradingDashboard', () => {
  beforeEach(() => {
    (useTradingStore as jest.Mock).mockReturnValue(mockTradingStore);
    (useAuthStore as jest.Mock).mockReturnValue(mockAuthStore);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('renders dashboard when authenticated', () => {
    render(<RealTimeTradingDashboard />);
    
    expect(screen.getByText('Real-Time Trading Dashboard')).toBeInTheDocument();
    expect(screen.getByText('Enhanced with Transformer ML Intelligence')).toBeInTheDocument();
  });

  it('shows authentication required when not authenticated', () => {
    (useAuthStore as jest.Mock).mockReturnValue({
      ...mockAuthStore,
      isAuthenticated: false
    });

    render(<RealTimeTradingDashboard />);
    
    expect(screen.getByText('Authentication Required')).toBeInTheDocument();
    expect(screen.getByText('Please log in to access the real-time trading dashboard.')).toBeInTheDocument();
  });

  it('initializes WebSocket connection when authenticated', () => {
    render(<RealTimeTradingDashboard />);
    
    expect(mockTradingStore.initializeWebSocket).toHaveBeenCalled();
  });

  it('displays connection status correctly', () => {
    render(<RealTimeTradingDashboard />);
    
    expect(screen.getByText('Live')).toBeInTheDocument();
  });

  it('allows symbol selection', () => {
    render(<RealTimeTradingDashboard />);
    
    const symbolSelect = screen.getByDisplayValue('BTCUSD');
    fireEvent.change(symbolSelect, { target: { value: 'ETHUSD' } });
    
    expect(mockTradingStore.setSelectedSymbol).toHaveBeenCalledWith('ETHUSD');
  });

  it('toggles real-time signals setting', () => {
    render(<RealTimeTradingDashboard />);
    
    const signalsToggle = screen.getByRole('button', { name: /real-time signals/i });
    fireEvent.click(signalsToggle);
    
    expect(mockTradingStore.updateSettings).toHaveBeenCalledWith({
      enableRealTimeSignals: false
    });
  });

  it('switches between overview and detailed views', () => {
    render(<RealTimeTradingDashboard />);
    
    const detailedButton = screen.getByText('Detailed');
    fireEvent.click(detailedButton);
    
    // Should show detailed layout
    expect(screen.getByText('Detailed')).toHaveClass('bg-white');
  });
});

describe('SignalQualityIndicator', () => {
  beforeEach(() => {
    (useTradingStore as jest.Mock).mockReturnValue({
      ...mockTradingStore,
      latestSignals: {
        BTCUSD: mockTradingStore.tradingSignals[0]
      }
    });
  });

  it('displays signal quality metrics', () => {
    render(<SignalQualityIndicator symbol="BTCUSD" />);
    
    expect(screen.getByText('EXCELLENT QUALITY')).toBeInTheDocument();
    expect(screen.getByText('85%')).toBeInTheDocument();
    expect(screen.getByText('BUY SIGNAL')).toBeInTheDocument();
  });

  it('shows component scores', () => {
    render(<SignalQualityIndicator symbol="BTCUSD" showDetails={true} />);
    
    expect(screen.getByText('Transformer:')).toBeInTheDocument();
    expect(screen.getByText('Ensemble:')).toBeInTheDocument();
    expect(screen.getByText('SMC:')).toBeInTheDocument();
    expect(screen.getByText('Technical:')).toBeInTheDocument();
  });

  it('displays risk management metrics', () => {
    render(<SignalQualityIndicator symbol="BTCUSD" showDetails={true} />);
    
    expect(screen.getByText('Stop Loss:')).toBeInTheDocument();
    expect(screen.getByText('$49,000.00')).toBeInTheDocument();
    expect(screen.getByText('Take Profit:')).toBeInTheDocument();
    expect(screen.getByText('$52,000.00')).toBeInTheDocument();
  });

  it('shows compact view correctly', () => {
    render(<SignalQualityIndicator symbol="BTCUSD" compact={true} />);
    
    expect(screen.getByText('EXCELLENT')).toBeInTheDocument();
    expect(screen.getByText('85%')).toBeInTheDocument();
  });

  it('handles no signal state', () => {
    (useTradingStore as jest.Mock).mockReturnValue({
      ...mockTradingStore,
      latestSignals: {}
    });

    render(<SignalQualityIndicator symbol="BTCUSD" />);
    
    expect(screen.getByText('Waiting for trading signals...')).toBeInTheDocument();
  });
});

describe('RealTimePortfolioMonitor', () => {
  it('displays portfolio summary', () => {
    render(<RealTimePortfolioMonitor />);
    
    expect(screen.getByText('$10,000.00')).toBeInTheDocument();
    expect(screen.getByText('$500.00')).toBeInTheDocument();
    expect(screen.getByText('+5.00%')).toBeInTheDocument();
  });

  it('shows performance metrics', () => {
    render(<RealTimePortfolioMonitor />);
    
    expect(screen.getByText('70.0%')).toBeInTheDocument(); // Win Rate
    expect(screen.getByText('10')).toBeInTheDocument(); // Total Signals
  });

  it('displays active positions', () => {
    render(<RealTimePortfolioMonitor showPositions={true} />);
    
    expect(screen.getByText('Active Positions')).toBeInTheDocument();
    expect(screen.getByText('BTCUSD')).toBeInTheDocument();
    expect(screen.getByText('0.100000')).toBeInTheDocument();
  });

  it('shows compact view', () => {
    render(<RealTimePortfolioMonitor compact={true} />);
    
    expect(screen.getByText('Portfolio Value')).toBeInTheDocument();
    expect(screen.getByText('$10,000.00')).toBeInTheDocument();
  });

  it('handles empty positions', () => {
    (useTradingStore as jest.Mock).mockReturnValue({
      ...mockTradingStore,
      portfolio: {
        ...mockTradingStore.portfolio,
        positions: {}
      }
    });

    render(<RealTimePortfolioMonitor showPositions={true} />);
    
    expect(screen.getByText('No active positions')).toBeInTheDocument();
  });
});

describe('TradingSignalHistory', () => {
  it('displays signal history', () => {
    render(<TradingSignalHistory />);
    
    expect(screen.getByText('Trading Signals')).toBeInTheDocument();
    expect(screen.getByText('1 signals')).toBeInTheDocument();
    expect(screen.getByText('BUY')).toBeInTheDocument();
  });

  it('filters signals by type', async () => {
    render(<TradingSignalHistory showFilters={true} />);
    
    const buyFilter = screen.getByText('Buy');
    fireEvent.click(buyFilter);
    
    await waitFor(() => {
      expect(screen.getByText('BUY')).toBeInTheDocument();
    });
  });

  it('filters signals by symbol', async () => {
    render(<TradingSignalHistory showFilters={true} />);
    
    const symbolSelect = screen.getByDisplayValue('All Symbols');
    fireEvent.change(symbolSelect, { target: { value: 'BTCUSD' } });
    
    await waitFor(() => {
      expect(screen.getByText('BTCUSD')).toBeInTheDocument();
    });
  });

  it('shows component scores for signals', () => {
    render(<TradingSignalHistory />);
    
    expect(screen.getByText('80%')).toBeInTheDocument(); // Transformer
    expect(screen.getByText('75%')).toBeInTheDocument(); // Ensemble
    expect(screen.getByText('90%')).toBeInTheDocument(); // SMC
    expect(screen.getByText('70%')).toBeInTheDocument(); // Technical
  });

  it('displays compact view', () => {
    render(<TradingSignalHistory compact={true} />);
    
    expect(screen.getByText('Recent Signals')).toBeInTheDocument();
    expect(screen.getByText('BUY')).toBeInTheDocument();
  });

  it('handles empty signal history', () => {
    (useTradingStore as jest.Mock).mockReturnValue({
      ...mockTradingStore,
      tradingSignals: []
    });

    render(<TradingSignalHistory />);
    
    expect(screen.getByText('No trading signals found')).toBeInTheDocument();
  });
});

describe('Real-Time Data Integration', () => {
  it('updates components when market data changes', async () => {
    const { rerender } = render(<RealTimeTradingDashboard />);
    
    // Update market data
    const updatedStore = {
      ...mockTradingStore,
      marketData: {
        BTCUSD: {
          ...mockTradingStore.marketData.BTCUSD,
          price: 51000,
          change: 2000,
          changePercent: 4.0
        }
      }
    };
    
    (useTradingStore as jest.Mock).mockReturnValue(updatedStore);
    
    rerender(<RealTimeTradingDashboard />);
    
    await waitFor(() => {
      expect(screen.getByText('$51,000.00')).toBeInTheDocument();
    });
  });

  it('handles connection status changes', async () => {
    const { rerender } = render(<RealTimeTradingDashboard />);
    
    // Simulate disconnection
    const disconnectedStore = {
      ...mockTradingStore,
      isConnected: false,
      connectionStatus: { status: 'disconnected' }
    };
    
    (useTradingStore as jest.Mock).mockReturnValue(disconnectedStore);
    
    rerender(<RealTimeTradingDashboard />);
    
    await waitFor(() => {
      expect(screen.getByText('Disconnected')).toBeInTheDocument();
    });
  });

  it('performs cleanup on unmount', () => {
    const { unmount } = render(<RealTimeTradingDashboard />);
    
    unmount();
    
    expect(mockTradingStore.disconnectWebSocket).toHaveBeenCalled();
  });
});
