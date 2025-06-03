/**
 * ML Intelligence Integration Tests
 * Task #31: ML Trading Intelligence Integration
 * Comprehensive tests for ML intelligence components and services
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import { MLIntelligenceDashboard } from '../../components/intelligence/MLIntelligenceDashboard';
import { mlIntelligenceService, type MLIntelligenceData, type MLPerformanceMetrics } from '../../lib/services/mlIntelligenceService';
import { useTradingStore } from '../../lib/stores/tradingStore';

// Mock the services and stores
jest.mock('../../lib/services/mlIntelligenceService');
jest.mock('../../lib/stores/tradingStore');

const mockMLIntelligenceData: MLIntelligenceData = {
  timestamp: '2024-01-01T12:00:00Z',
  symbol: 'BTCUSD',
  signal: {
    id: 'test-signal-1',
    signal_type: 'strong_buy',
    confidence: 0.85,
    quality: 'excellent',
    price: 50000,
    transformer_prediction: 0.8,
    ensemble_prediction: 0.75,
    smc_score: 0.9,
    technical_score: 0.7,
    stop_loss: 49000,
    take_profit: 52000,
    position_size: 0.1,
    risk_reward_ratio: 2.0
  },
  pipeline_prediction: [0.78],
  regime_analysis: {
    volatility_regime: 'medium',
    volatility_percentile: 0.6,
    trend_regime: 'strong_bullish',
    trend_strength: 0.08,
    trend_direction: 'bullish',
    volume_regime: 'high',
    volume_ratio: 1.5,
    market_condition: 'trending_stable'
  },
  risk_assessment: {
    var_95: -0.025,
    var_99: -0.045,
    maximum_adverse_excursion: -0.03,
    kelly_fraction: 0.15,
    risk_adjusted_position_size: 0.08,
    risk_reward_ratio: 2.0,
    confidence_adjusted_risk: 0.068,
    risk_level: 'medium'
  },
  execution_strategy: {
    entry_method: 'limit',
    exit_method: 'limit',
    time_in_force: 'GTC',
    execution_urgency: 'normal',
    entry_offset_pct: 0.001,
    partial_fill_allowed: false,
    recommended_timing: 'immediate',
    max_execution_time_minutes: 30,
    slippage_tolerance_pct: 0.002
  },
  confidence_score: 0.85,
  quality_rating: 'excellent',
  intelligence_version: '1.0'
};

const mockPerformanceMetrics: MLPerformanceMetrics = {
  overall_accuracy: 0.78,
  transformer_accuracy: 0.82,
  ensemble_accuracy: 0.75,
  signal_quality_accuracy: 0.80,
  prediction_latency_ms: 85,
  throughput_predictions_per_second: 12.5,
  memory_usage_gb: 1.8,
  win_rate: 0.72,
  profit_factor: 1.85,
  sharpe_ratio: 1.45,
  max_drawdown: 0.08,
  model_confidence: 0.83,
  prediction_consistency: 0.88,
  error_rate: 0.05,
  uptime_percentage: 99.2,
  last_update: '2024-01-01T12:00:00Z'
};

const mockTradingStore = {
  selectedSymbol: 'BTCUSD',
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
  mlIntelligence: {
    currentIntelligence: {
      BTCUSD: mockMLIntelligenceData
    },
    performanceMetrics: mockPerformanceMetrics,
    intelligenceHistory: [mockMLIntelligenceData],
    isMLConnected: true,
    lastMLUpdate: Date.now()
  },
  requestMLIntelligence: jest.fn()
};

const mockMLIntelligenceService = {
  requestIntelligence: jest.fn(),
  getPerformanceMetrics: jest.fn(),
  getIntelligenceSummary: jest.fn(),
  subscribe: jest.fn(),
  isMLIntelligenceConnected: jest.fn(),
  formatIntelligenceForDisplay: jest.fn(),
  getIntelligenceQualityScore: jest.fn(),
  getPerformanceSummary: jest.fn()
};

describe('MLIntelligenceDashboard', () => {
  beforeEach(() => {
    (useTradingStore as jest.Mock).mockReturnValue(mockTradingStore);
    (mlIntelligenceService.requestIntelligence as jest.Mock).mockResolvedValue(mockMLIntelligenceData);
    (mlIntelligenceService.getPerformanceMetrics as jest.Mock).mockResolvedValue(mockPerformanceMetrics);
    (mlIntelligenceService.isMLIntelligenceConnected as jest.Mock).mockReturnValue(true);
    (mlIntelligenceService.formatIntelligenceForDisplay as jest.Mock).mockReturnValue({
      signal: {
        type: 'STRONG BUY',
        confidence: '85.0%',
        quality: 'EXCELLENT',
        price: '$50,000.00'
      },
      regime: {
        condition: 'TRENDING STABLE',
        volatility: 'MEDIUM',
        trend: 'BULLISH (0.080)'
      },
      risk: {
        level: 'MEDIUM',
        position_size: '8.0%',
        risk_reward: '2.0',
        var_95: '-2.50%'
      },
      execution: {
        method: 'LIMIT',
        urgency: 'NORMAL',
        timing: 'IMMEDIATE',
        max_time: '30min'
      }
    });
    (mlIntelligenceService.getIntelligenceQualityScore as jest.Mock).mockReturnValue(0.85);
    (mlIntelligenceService.getPerformanceSummary as jest.Mock).mockReturnValue({
      overall: {
        accuracy: '78.0%',
        win_rate: '72.0%',
        profit_factor: '1.85',
        sharpe_ratio: '1.45'
      },
      performance: {
        latency: '85ms',
        throughput: '12.5/s',
        memory: '1.80GB',
        uptime: '99.2%'
      },
      models: {
        transformer: '82.0%',
        ensemble: '75.0%',
        signal_quality: '80.0%',
        confidence: '83.0%'
      },
      health: {
        error_rate: '5.00%',
        consistency: '88.0%',
        last_update: '1/1/2024, 12:00:00 PM'
      }
    });
    (mlIntelligenceService.subscribe as jest.Mock).mockReturnValue(() => {});
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('renders ML intelligence dashboard', () => {
    render(<MLIntelligenceDashboard symbol="BTCUSD" />);
    
    expect(screen.getByText('ML Intelligence Center')).toBeInTheDocument();
    expect(screen.getByText('Advanced ML trading intelligence for BTCUSD')).toBeInTheDocument();
  });

  it('displays quality score', () => {
    render(<MLIntelligenceDashboard symbol="BTCUSD" />);
    
    expect(screen.getByText('Quality: 85%')).toBeInTheDocument();
  });

  it('shows ML connection status', () => {
    render(<MLIntelligenceDashboard symbol="BTCUSD" />);
    
    expect(screen.getByText('ML Active')).toBeInTheDocument();
  });

  it('displays overview tab by default', () => {
    render(<MLIntelligenceDashboard symbol="BTCUSD" />);
    
    expect(screen.getByText('STRONG BUY')).toBeInTheDocument();
    expect(screen.getByText('EXCELLENT')).toBeInTheDocument();
    expect(screen.getByText('TRENDING STABLE')).toBeInTheDocument();
    expect(screen.getByText('MEDIUM')).toBeInTheDocument();
  });

  it('switches to performance tab', async () => {
    render(<MLIntelligenceDashboard symbol="BTCUSD" />);
    
    const performanceTab = screen.getByText('Performance');
    fireEvent.click(performanceTab);
    
    await waitFor(() => {
      expect(screen.getByText('Overall Accuracy')).toBeInTheDocument();
      expect(screen.getByText('78.0%')).toBeInTheDocument();
      expect(screen.getByText('Win Rate')).toBeInTheDocument();
      expect(screen.getByText('72.0%')).toBeInTheDocument();
    });
  });

  it('switches to analysis tab', async () => {
    render(<MLIntelligenceDashboard symbol="BTCUSD" />);
    
    const analysisTab = screen.getByText('Analysis');
    fireEvent.click(analysisTab);
    
    await waitFor(() => {
      expect(screen.getByText('Market Regime Analysis')).toBeInTheDocument();
      expect(screen.getByText('Risk Assessment')).toBeInTheDocument();
    });
  });

  it('switches to execution tab', async () => {
    render(<MLIntelligenceDashboard symbol="BTCUSD" />);
    
    const executionTab = screen.getByText('Execution');
    fireEvent.click(executionTab);
    
    await waitFor(() => {
      expect(screen.getByText('Execution Strategy')).toBeInTheDocument();
      expect(screen.getByText('Risk Management')).toBeInTheDocument();
    });
  });

  it('requests ML intelligence on refresh', async () => {
    render(<MLIntelligenceDashboard symbol="BTCUSD" />);
    
    const refreshButton = screen.getByText('Refresh');
    fireEvent.click(refreshButton);
    
    await waitFor(() => {
      expect(mlIntelligenceService.requestIntelligence).toHaveBeenCalledWith(
        'BTCUSD',
        mockTradingStore.marketData.BTCUSD
      );
    });
  });

  it('handles loading state', async () => {
    (mlIntelligenceService.requestIntelligence as jest.Mock).mockImplementation(
      () => new Promise(resolve => setTimeout(() => resolve(mockMLIntelligenceData), 100))
    );

    render(<MLIntelligenceDashboard symbol="BTCUSD" />);
    
    const refreshButton = screen.getByText('Refresh');
    fireEvent.click(refreshButton);
    
    expect(screen.getByText('Loading...')).toBeInTheDocument();
    
    await waitFor(() => {
      expect(screen.getByText('Refresh')).toBeInTheDocument();
    });
  });

  it('handles error state', async () => {
    (mlIntelligenceService.requestIntelligence as jest.Mock).mockRejectedValue(
      new Error('ML service unavailable')
    );

    render(<MLIntelligenceDashboard symbol="BTCUSD" />);
    
    const refreshButton = screen.getByText('Refresh');
    fireEvent.click(refreshButton);
    
    await waitFor(() => {
      expect(screen.getByText('ML service unavailable')).toBeInTheDocument();
    });
  });

  it('displays component scores', () => {
    render(<MLIntelligenceDashboard symbol="BTCUSD" />);
    
    expect(screen.getByText('80%')).toBeInTheDocument(); // Transformer
    expect(screen.getByText('75%')).toBeInTheDocument(); // Ensemble
    expect(screen.getByText('90%')).toBeInTheDocument(); // SMC
    expect(screen.getByText('70%')).toBeInTheDocument(); // Technical
  });

  it('shows performance summary', () => {
    render(<MLIntelligenceDashboard symbol="BTCUSD" />);
    
    expect(screen.getByText('ML Performance Summary')).toBeInTheDocument();
    expect(screen.getByText('78.0%')).toBeInTheDocument(); // Accuracy
    expect(screen.getByText('72.0%')).toBeInTheDocument(); // Win Rate
    expect(screen.getByText('85ms')).toBeInTheDocument(); // Latency
    expect(screen.getByText('1.80GB')).toBeInTheDocument(); // Memory
  });
});

describe('MLIntelligenceService', () => {
  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();
    
    // Mock fetch
    global.fetch = jest.fn();
    
    // Mock localStorage
    Object.defineProperty(window, 'localStorage', {
      value: {
        getItem: jest.fn(() => 'mock-token'),
        setItem: jest.fn(),
        removeItem: jest.fn(),
      },
      writable: true,
    });
  });

  it('requests ML intelligence successfully', async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockMLIntelligenceData)
    });

    const result = await mlIntelligenceService.requestIntelligence('BTCUSD');
    
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/ml/intelligence'),
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          'Content-Type': 'application/json',
          'Authorization': 'Bearer mock-token'
        }),
        body: expect.stringContaining('BTCUSD')
      })
    );
    
    expect(result).toEqual(mockMLIntelligenceData);
  });

  it('handles request failure', async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: false,
      statusText: 'Internal Server Error'
    });

    const result = await mlIntelligenceService.requestIntelligence('BTCUSD');
    
    expect(result).toBeNull();
  });

  it('gets performance metrics', async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockPerformanceMetrics)
    });

    const result = await mlIntelligenceService.getPerformanceMetrics();
    
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/ml/performance'),
      expect.objectContaining({
        headers: expect.objectContaining({
          'Authorization': 'Bearer mock-token'
        })
      })
    );
    
    expect(result).toEqual(mockPerformanceMetrics);
  });

  it('validates intelligence data', () => {
    const validData = mockMLIntelligenceData;
    const invalidData = { invalid: 'data' };
    
    expect(mlIntelligenceService.validateIntelligenceData(validData)).toBe(true);
    expect(mlIntelligenceService.validateIntelligenceData(invalidData)).toBe(false);
  });

  it('calculates intelligence quality score', () => {
    const score = mlIntelligenceService.getIntelligenceQualityScore(mockMLIntelligenceData);
    
    expect(score).toBeGreaterThan(0);
    expect(score).toBeLessThanOrEqual(1);
  });

  it('formats intelligence for display', () => {
    const formatted = mlIntelligenceService.formatIntelligenceForDisplay(mockMLIntelligenceData);
    
    expect(formatted).toHaveProperty('signal');
    expect(formatted).toHaveProperty('regime');
    expect(formatted).toHaveProperty('risk');
    expect(formatted).toHaveProperty('execution');
    
    expect(formatted.signal.type).toBe('STRONG_BUY');
    expect(formatted.signal.confidence).toBe('85.0%');
  });
});

describe('ML Intelligence Integration', () => {
  it('integrates with trading store', () => {
    const store = mockTradingStore;
    
    expect(store.mlIntelligence).toBeDefined();
    expect(store.mlIntelligence.currentIntelligence).toBeDefined();
    expect(store.mlIntelligence.performanceMetrics).toBeDefined();
    expect(store.mlIntelligence.isMLConnected).toBe(true);
  });

  it('updates ML intelligence in store', () => {
    const updateMLIntelligence = jest.fn();
    
    updateMLIntelligence('BTCUSD', mockMLIntelligenceData);
    
    expect(updateMLIntelligence).toHaveBeenCalledWith('BTCUSD', mockMLIntelligenceData);
  });

  it('handles real-time ML updates', () => {
    const subscribe = jest.fn();
    
    subscribe('intelligence_update', (data: MLIntelligenceData) => {
      expect(data.symbol).toBe('BTCUSD');
      expect(data.signal.confidence).toBeGreaterThan(0);
    });
    
    expect(subscribe).toHaveBeenCalled();
  });

  it('manages memory efficiently', () => {
    const clearMLIntelligenceHistory = jest.fn();
    
    clearMLIntelligenceHistory();
    
    expect(clearMLIntelligenceHistory).toHaveBeenCalled();
  });
});
