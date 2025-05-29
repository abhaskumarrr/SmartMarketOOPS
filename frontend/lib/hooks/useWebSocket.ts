import { useState, useEffect, useCallback } from 'react';
import WebSocketService, { ConnectionStatus, WebSocketEventType } from '../websocketService';

interface UseWebSocketOptions {
  autoConnect?: boolean;
}

/**
 * React hook for using the WebSocket service
 * @param channel Channel to subscribe to (e.g. 'market', 'trades')
 * @param event Event type to listen for (e.g. 'market:data', 'trade:executed')
 * @param options Additional options
 */
function useWebSocket<T = any>(
  channel: string,
  event: WebSocketEventType,
  channelData?: any,
  options: UseWebSocketOptions = {}
) {
  const [data, setData] = useState<T | null>(null);
  const [status, setStatus] = useState<ConnectionStatus>(ConnectionStatus.DISCONNECTED);
  const [error, setError] = useState<Error | null>(null);
  
  // Initialize WebSocket service
  const wsService = WebSocketService.getInstance();
  
  // Handle incoming data
  const handleData = useCallback((newData: T) => {
    setData(newData);
    setError(null);
  }, []);
  
  // Handle status changes
  const handleStatus = useCallback((newStatus: ConnectionStatus) => {
    setStatus(newStatus);
    if (newStatus === ConnectionStatus.ERROR) {
      setError(new Error('WebSocket connection error'));
    } else {
      setError(null);
    }
  }, []);
  
  // Connect to WebSocket on mount
  useEffect(() => {
    // Subscribe to status updates
    wsService.on('status', handleStatus);
    
    // Get initial status
    setStatus(wsService.getStatus());
    
    // Auto-connect if specified
    if (options.autoConnect !== false) {
      wsService.connect().catch(err => {
        setError(err);
      });
    }
    
    // Subscribe to data events
    wsService.on(event, handleData);
    
    // Subscribe to the channel
    wsService.subscribe(channel, channelData);
    
    // Cleanup on unmount
    return () => {
      wsService.removeListener(event, handleData);
      wsService.removeListener('status', handleStatus);
      wsService.unsubscribe(channel, channelData);
    };
  }, [channel, event, channelData, handleData, handleStatus, options.autoConnect, wsService]);
  
  // Provide methods to manually connect/disconnect
  const connect = useCallback(() => {
    return wsService.connect();
  }, [wsService]);
  
  const disconnect = useCallback(() => {
    wsService.disconnect();
  }, [wsService]);
  
  return {
    data,
    status,
    error,
    connect,
    disconnect,
    isConnected: status === ConnectionStatus.CONNECTED,
    isConnecting: status === ConnectionStatus.CONNECTING,
    isDisconnected: status === ConnectionStatus.DISCONNECTED,
    hasError: status === ConnectionStatus.ERROR
  };
}

export default useWebSocket;