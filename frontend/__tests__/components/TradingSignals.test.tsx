import React from 'react';
import { render, screen } from '@testing-library/react';
import TradingSignals from '../../components/TradingSignals';
import WebSocketService, { ConnectionStatus } from '../../lib/websocketService';

// Get the mock from jest.setup.js
const mockWebSocketService = WebSocketService.getInstance() as any;

// Mock signal data
const mockSignal = {
  id: '1',
  symbol: 'BTCUSD',
  type: 'buy',
  price: 45000,
  timestamp: new Date().toISOString(),
  confidence: 0.85,
  reason: 'Strong upward trend detected'
};

describe('TradingSignals component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Reset the mock service state
    mockWebSocketService.mockChangeStatus(ConnectionStatus.DISCONNECTED);
  });

  test('renders title correctly', () => {
    render(<TradingSignals />);
    
    const titleElement = screen.getByText('Trading Signals');
    expect(titleElement).toBeInTheDocument();
  });

  test('shows connecting state', () => {
    mockWebSocketService.mockChangeStatus(ConnectionStatus.CONNECTING);
    
    render(<TradingSignals />);
    
    const connectingText = screen.getByText('Connecting...');
    expect(connectingText).toBeInTheDocument();
  });

  test('shows connected state with live indicator', () => {
    mockWebSocketService.mockChangeStatus(ConnectionStatus.CONNECTED);
    
    render(<TradingSignals />);
    
    const liveIndicator = screen.getByText('Live');
    expect(liveIndicator).toBeInTheDocument();
  });

  test('shows error state', () => {
    mockWebSocketService.mockChangeStatus(ConnectionStatus.ERROR);
    
    render(<TradingSignals />);
    
    const errorIndicator = screen.getByText('Connection Error');
    expect(errorIndicator).toBeInTheDocument();
  });

  test('shows empty state message when connected but no signals', () => {
    mockWebSocketService.mockChangeStatus(ConnectionStatus.CONNECTED);
    
    render(<TradingSignals />);
    
    const emptyMessage = screen.getByText(/No trading signals yet/i);
    expect(emptyMessage).toBeInTheDocument();
  });

  test('displays signal when received from WebSocket', () => {
    // Start with connected status
    mockWebSocketService.mockChangeStatus(ConnectionStatus.CONNECTED);
    
    render(<TradingSignals />);
    
    // Simulate receiving a signal
    mockWebSocketService.mockReceiveData('signal:new', mockSignal);
    
    // Check for signal type (BUY)
    const signalType = screen.getByText('BUY');
    expect(signalType).toBeInTheDocument();
    
    // Check for price
    const price = screen.getByText(/\$45,000/);
    expect(price).toBeInTheDocument();
    
    // Check for reason
    const reason = screen.getByText('Strong upward trend detected');
    expect(reason).toBeInTheDocument();
    
    // Check for confidence chip
    const confidence = screen.getByText(/Confidence: 85%/);
    expect(confidence).toBeInTheDocument();
  });

  test('limits the number of signals to maxSignals prop', async () => {
    // Start with connected status
    mockWebSocketService.mockChangeStatus(ConnectionStatus.CONNECTED);
    
    // Render with maxSignals set to 2
    render(<TradingSignals maxSignals={2} />);
    
    // Simulate receiving multiple signals
    mockWebSocketService.mockReceiveData('signal:new', {
      ...mockSignal,
      id: '1',
      type: 'buy',
      price: 45000,
    });
    
    mockWebSocketService.mockReceiveData('signal:new', {
      ...mockSignal,
      id: '2',
      type: 'sell',
      price: 46000,
    });
    
    mockWebSocketService.mockReceiveData('signal:new', {
      ...mockSignal,
      id: '3',
      type: 'hold',
      price: 45500,
    });
    
    // Should only show the latest 2 signals (id 2 and 3)
    expect(screen.queryByText('$45,000')).not.toBeInTheDocument(); // First signal should be removed
    expect(screen.getByText('$46,000')).toBeInTheDocument(); // Second signal should be visible
    expect(screen.getByText('$45,500')).toBeInTheDocument(); // Third signal should be visible
  });

  test('respects the symbol prop for filtering signals', () => {
    // Start with connected status
    mockWebSocketService.mockChangeStatus(ConnectionStatus.CONNECTED);
    
    // Render with specific symbol
    render(<TradingSignals symbol="ETHUSD" />);
    
    // Simulate receiving signals for different symbols
    mockWebSocketService.mockReceiveData('signal:new', {
      ...mockSignal,
      symbol: 'BTCUSD',
      price: 45000,
    });
    
    mockWebSocketService.mockReceiveData('signal:new', {
      ...mockSignal,
      symbol: 'ETHUSD',
      price: 3000,
    });
    
    // Should only show the ETHUSD signal
    expect(screen.queryByText('$45,000')).not.toBeInTheDocument(); // BTCUSD signal should not be shown
    expect(screen.getByText('$3,000')).toBeInTheDocument(); // ETHUSD signal should be visible
  });
}); 