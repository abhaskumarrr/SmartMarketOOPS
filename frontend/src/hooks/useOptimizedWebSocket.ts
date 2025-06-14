import { useState, useEffect, useRef, useCallback, useMemo } from 'react';

// Defines the structure of messages stored in the buffer
interface BufferedMessage<T = unknown> {
  data: T;
  timestamp: number;
}

// Configuration options for the optimized WebSocket
interface WebSocketOptions {
  // How often to flush the buffer (in ms)
  bufferInterval?: number;
  // Maximum number of messages to process per flush
  batchSize?: number;
  // Enable/disable buffering mechanism
  enableBuffering?: boolean;
  // Auto reconnect when connection is lost
  autoReconnect?: boolean;
  // Number of reconnect attempts
  maxReconnectAttempts?: number;
  // Delay between reconnect attempts (in ms)
  reconnectDelay?: number;
  // Initial backoff delay for reconnection (ms)
  initialBackoff?: number;
  // Maximum backoff delay for reconnection (ms)
  maxBackoff?: number;
  // Optional protocol strings
  protocols?: string | string[];
  // Debug mode to log WebSocket events
  debug?: boolean;
}

// Default options
const DEFAULT_OPTIONS: WebSocketOptions = {
  bufferInterval: 100, // 100ms buffer flush interval
  batchSize: 10, // Process 10 messages at a time
  enableBuffering: true,
  autoReconnect: true,
  maxReconnectAttempts: 5,
  reconnectDelay: 1000,
  initialBackoff: 300,
  maxBackoff: 10000,
  debug: false
};

// WebSocket connection pool to reuse connections
const connectionPool: Record<string, {
  connection: WebSocket;
  refCount: number;
  lastUsed: number;
}> = {};

// Periodic cleanup of idle connections (runs every 5 minutes)
const CLEANUP_INTERVAL = 5 * 60 * 1000; // 5 minutes
const MAX_IDLE_TIME = 10 * 60 * 1000; // 10 minutes

// Setup connection pool cleanup
if (typeof window !== 'undefined') {
  setInterval(() => {
    const now = Date.now();
    Object.keys(connectionPool).forEach(url => {
      const pooledConnection = connectionPool[url];
      if (pooledConnection.refCount === 0 && now - pooledConnection.lastUsed > MAX_IDLE_TIME) {
        if (pooledConnection.connection.readyState === WebSocket.OPEN) {
          pooledConnection.connection.close();
        }
        delete connectionPool[url];
      }
    });
  }, CLEANUP_INTERVAL);
}

/**
 * An optimized WebSocket hook with performance enhancements:
 * - Connection pooling to reuse WebSocket connections
 * - Message buffering to batch updates and reduce render cycles
 * - Automatic reconnection with exponential backoff
 * - Rate limiting for high-frequency data
 */
export function useOptimizedWebSocket<T = unknown>(
  url: string | null,
  options: WebSocketOptions = {}
) {
  // Use useMemo to prevent recreation of mergedOptions on every render
  const mergedOptions = useMemo(() => ({ ...DEFAULT_OPTIONS, ...options }), [options]);
  
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<Event | null>(null);
  const [data, setData] = useState<T | null>(null);
  const [bufferedData, setBufferedData] = useState<T[]>([]);
  
  // References
  const ws = useRef<WebSocket | null>(null);
  const messageBuffer = useRef<BufferedMessage<T>[]>([]);
  const reconnectAttempts = useRef(0);
  const reconnectTimeout = useRef<NodeJS.Timeout | null>(null);
  const bufferTimeout = useRef<NodeJS.Timeout | null>(null);
  const currentBackoff = useRef(mergedOptions.initialBackoff || 300);
  
  // Schedule a reconnection attempt with exponential backoff
  const scheduleReconnect = useCallback(() => {
    if (
      reconnectAttempts.current < (mergedOptions.maxReconnectAttempts || 5)
    ) {
      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current);
      }
      
      // Calculate backoff with jitter
      const jitter = Math.random() * 0.3 + 0.85; // 0.85-1.15 range for jitter
      const delay = Math.min(
        currentBackoff.current * jitter,
        mergedOptions.maxBackoff || 10000
      );
      
      if (mergedOptions.debug) {
        console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts.current + 1})`);
      }
      
      reconnectTimeout.current = setTimeout(() => {
        reconnectAttempts.current++;
        currentBackoff.current = Math.min(
          currentBackoff.current * 2,
          mergedOptions.maxBackoff || 10000
        );
        connect();
      }, delay);
    } else {
      console.error(`Max reconnect attempts (${mergedOptions.maxReconnectAttempts}) exceeded`);
    }
  }, [mergedOptions]);
  
  // Close any previous connection when the component unmounts
  const cleanup = useCallback(() => {
    if (reconnectTimeout.current) {
      clearTimeout(reconnectTimeout.current);
      reconnectTimeout.current = null;
    }
    
    if (bufferTimeout.current) {
      clearTimeout(bufferTimeout.current);
      bufferTimeout.current = null;
    }
    
    if (url && connectionPool[url]) {
      connectionPool[url].refCount--;
      connectionPool[url].lastUsed = Date.now();
    }
  }, [url]);
  
  // Process messages from the buffer
  const processBuffer = useCallback(() => {
    if (messageBuffer.current.length === 0) return;
    
    // Sort messages by timestamp (oldest first)
    messageBuffer.current.sort((a, b) => a.timestamp - b.timestamp);
    
    // Take only the most recent messages up to batchSize
    const batch = messageBuffer.current.slice(-mergedOptions.batchSize!);
    
    // Get the most recent message for the primary data state
    const latestMessage = batch[batch.length - 1];
    setData(latestMessage.data);
    
    // Set all messages for the buffered data state
    setBufferedData(batch.map(msg => msg.data));
    
    // Clear the processed messages
    messageBuffer.current = messageBuffer.current.slice(0, -mergedOptions.batchSize!);
    
    // Schedule the next buffer processing if there are more messages
    if (messageBuffer.current.length > 0 && mergedOptions.enableBuffering) {
      bufferTimeout.current = setTimeout(processBuffer, mergedOptions.bufferInterval);
    }
  }, [mergedOptions.batchSize, mergedOptions.bufferInterval, mergedOptions.enableBuffering]);
  
  // Connect to the WebSocket server
  const connect = useCallback(() => {
    if (!url) return;
    
    // Check if we already have a pooled connection
    if (connectionPool[url]) {
      // Reuse existing connection
      const pooledConnection = connectionPool[url];
      pooledConnection.refCount++;
      pooledConnection.lastUsed = Date.now();
      ws.current = pooledConnection.connection;
      
      // If the connection is already open, set connected state
      if (ws.current.readyState === WebSocket.OPEN) {
        setIsConnected(true);
        setError(null);
        return;
      }
    } else {
      // Create a new connection
      try {
        const newWs = new WebSocket(url, mergedOptions.protocols);
        ws.current = newWs;
        
        // Add to connection pool
        connectionPool[url] = {
          connection: newWs,
          refCount: 1,
          lastUsed: Date.now()
        };
      } catch (err) {
        console.error('WebSocket connection error:', err);
        setError(err as Event);
        setIsConnected(false);
        
        // Try to reconnect if enabled
        if (mergedOptions.autoReconnect) {
          scheduleReconnect();
        }
        return;
      }
    }
    
    const socket = ws.current;
    
    // Set up event handlers
    socket.onopen = () => {
      if (mergedOptions.debug) console.log('WebSocket connected:', url);
      setIsConnected(true);
      setError(null);
      reconnectAttempts.current = 0;
      currentBackoff.current = mergedOptions.initialBackoff || 300;
    };
    
    socket.onclose = () => {
      if (mergedOptions.debug) console.log('WebSocket closed:', url);
      setIsConnected(false);
      
      // If not unmounting and auto reconnect is enabled, try to reconnect
      if (mergedOptions.autoReconnect) {
        scheduleReconnect();
      }
    };
    
    socket.onerror = (event) => {
      console.error('WebSocket error:', event);
      setError(event);
    };
    
    socket.onmessage = (event) => {
      try {
        const parsedData = JSON.parse(event.data) as T;
        
        if (mergedOptions.enableBuffering) {
          // Add to buffer
          messageBuffer.current.push({
            data: parsedData,
            timestamp: Date.now()
          });
          
          // If no buffer processing is scheduled, schedule one
          if (!bufferTimeout.current) {
            bufferTimeout.current = setTimeout(processBuffer, mergedOptions.bufferInterval);
          }
        } else {
          // Immediately update state if buffering is disabled
          setData(parsedData);
          setBufferedData([parsedData]);
        }
      } catch (err) {
        console.error('Error parsing WebSocket message:', err, event.data);
      }
    };
  }, [url, mergedOptions, processBuffer, scheduleReconnect]);
  
  // Send a message through the WebSocket
  const sendMessage = useCallback((message: string | Record<string, unknown>) => {
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) {
      console.error('WebSocket is not connected');
      return false;
    }
    
    try {
      const data = typeof message === 'string' ? message : JSON.stringify(message);
      ws.current.send(data);
      return true;
    } catch (err) {
      console.error('Error sending message:', err);
      return false;
    }
  }, []);
  
  // Manually reconnect the WebSocket
  const reconnect = useCallback(() => {
    cleanup();
    reconnectAttempts.current = 0;
    currentBackoff.current = mergedOptions.initialBackoff || 300;
    connect();
  }, [cleanup, connect, mergedOptions.initialBackoff]);
  
  // Manually disconnect the WebSocket
  const disconnect = useCallback(() => {
    if (ws.current) {
      if (ws.current.readyState === WebSocket.OPEN) {
        ws.current.close();
      }
      ws.current = null;
    }
    cleanup();
  }, [cleanup]);
  
  // Connect when the URL changes
  useEffect(() => {
    if (url) {
      connect();
    } else {
      disconnect();
    }
    
    return cleanup;
  }, [url, connect, disconnect, cleanup]);
  
  return {
    isConnected,
    error,
    data, // Latest message
    bufferedData, // Batch of messages for advanced rendering
    sendMessage,
    reconnect,
    disconnect
  };
}

export default useOptimizedWebSocket; 