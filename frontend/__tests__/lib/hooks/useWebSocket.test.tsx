import { renderHook, act } from '@testing-library/react';
import useWebSocket from '../../../lib/hooks/useWebSocket';
import WebSocketService, { ConnectionStatus } from '../../../lib/websocketService';

// Get the mock from jest.setup.js and add type assertion for TypeScript
const mockWebSocketService = WebSocketService.getInstance() as any;

describe('useWebSocket hook', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Reset the mock service state
    mockWebSocketService.mockChangeStatus(ConnectionStatus.DISCONNECTED);
  });

  test('returns initial state correctly', () => {
    const { result } = renderHook(() => useWebSocket('market', 'market:data'));
    
    expect(result.current.data).toBeNull();
    expect(result.current.status).toBe(ConnectionStatus.DISCONNECTED);
    expect(result.current.error).toBeNull();
    expect(result.current.isConnected).toBe(false);
    expect(result.current.isConnecting).toBe(false);
    expect(result.current.isDisconnected).toBe(true);
    expect(result.current.hasError).toBe(false);
  });

  test('subscribes to channel on mount', () => {
    // Spy on the subscribe method
    const subscribeSpy = jest.spyOn(mockWebSocketService, 'subscribe');
    
    renderHook(() => useWebSocket('market', 'market:data', { symbol: 'BTCUSD' }));
    
    expect(subscribeSpy).toHaveBeenCalledWith('market', { symbol: 'BTCUSD' });
  });

  test('unsubscribes from channel on unmount', () => {
    // Spy on the unsubscribe method
    const unsubscribeSpy = jest.spyOn(mockWebSocketService, 'unsubscribe');
    
    const { unmount } = renderHook(() => 
      useWebSocket('market', 'market:data', { symbol: 'BTCUSD' })
    );
    
    unmount();
    
    expect(unsubscribeSpy).toHaveBeenCalledWith('market', { symbol: 'BTCUSD' });
  });

  test('updates status when WebSocket status changes', () => {
    const { result } = renderHook(() => useWebSocket('market', 'market:data'));
    
    // Initial status
    expect(result.current.status).toBe(ConnectionStatus.DISCONNECTED);
    
    // Change status to CONNECTING
    act(() => {
      mockWebSocketService.mockChangeStatus(ConnectionStatus.CONNECTING);
    });
    
    expect(result.current.status).toBe(ConnectionStatus.CONNECTING);
    expect(result.current.isConnecting).toBe(true);
    expect(result.current.isConnected).toBe(false);
    
    // Change status to CONNECTED
    act(() => {
      mockWebSocketService.mockChangeStatus(ConnectionStatus.CONNECTED);
    });
    
    expect(result.current.status).toBe(ConnectionStatus.CONNECTED);
    expect(result.current.isConnected).toBe(true);
    expect(result.current.isConnecting).toBe(false);
  });

  test('updates data when event data is received', () => {
    const { result } = renderHook(() => useWebSocket('market', 'market:data'));
    
    const testData = { symbol: 'BTCUSD', price: 45000 };
    
    // Simulate receiving data
    act(() => {
      mockWebSocketService.mockReceiveData('market:data', testData);
    });
    
    expect(result.current.data).toEqual(testData);
  });

  test('sets error when WebSocket status is ERROR', () => {
    const { result } = renderHook(() => useWebSocket('market', 'market:data'));
    
    // Change status to ERROR
    act(() => {
      mockWebSocketService.mockChangeStatus(ConnectionStatus.ERROR);
    });
    
    expect(result.current.status).toBe(ConnectionStatus.ERROR);
    expect(result.current.hasError).toBe(true);
    expect(result.current.error).toBeInstanceOf(Error);
  });

  test('connect method calls WebSocketService connect', async () => {
    const connectSpy = jest.spyOn(mockWebSocketService, 'connect');
    
    const { result } = renderHook(() => useWebSocket('market', 'market:data', null, { autoConnect: false }));
    
    await act(async () => {
      await result.current.connect();
    });
    
    expect(connectSpy).toHaveBeenCalled();
  });

  test('disconnect method calls WebSocketService disconnect', () => {
    const disconnectSpy = jest.spyOn(mockWebSocketService, 'disconnect');
    
    const { result } = renderHook(() => useWebSocket('market', 'market:data'));
    
    act(() => {
      result.current.disconnect();
    });
    
    expect(disconnectSpy).toHaveBeenCalled();
  });
}); 