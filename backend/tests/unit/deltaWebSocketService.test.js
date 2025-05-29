/**
 * Unit tests for Delta Exchange WebSocket Service
 */
const WebSocket = require('ws');
const EventEmitter = require('events');

// Mock WebSocket
jest.mock('ws');

// Mock the WebSocket class
WebSocket.mockImplementation(() => {
  const mockWs = new EventEmitter();
  mockWs.send = jest.fn();
  mockWs.close = jest.fn();
  return mockWs;
});

// Import the service (adjust the path if needed)
const DeltaWebSocketService = require('../../src/services/deltaWebSocketService');

// Mock the logger
jest.mock('../../src/utils/logger', () => ({
  createLogger: jest.fn().mockReturnValue({
    info: jest.fn(),
    debug: jest.fn(),
    warn: jest.fn(),
    error: jest.fn()
  })
}));

describe('DeltaWebSocketService', () => {
  let deltaWs;
  let mockSocket;
  
  beforeEach(() => {
    // Clear all mocks
    jest.clearAllMocks();
    
    // Create a new instance for each test
    deltaWs = new DeltaWebSocketService({ testnet: true });
    
    // Save reference to the mocked WebSocket
    mockSocket = WebSocket.mock.results[0]?.value;
  });
  
  afterEach(() => {
    // Clean up
    if (deltaWs) {
      deltaWs.disconnect();
    }
  });
  
  describe('Connection Handling', () => {
    test('should connect to WebSocket server', () => {
      // Act
      deltaWs.connect();
      
      // Assert
      expect(WebSocket).toHaveBeenCalledWith(expect.stringContaining('delta.exchange'), expect.any(Object));
    });
    
    test('should use testnet URL when testnet is true', () => {
      // Act
      deltaWs.connect();
      
      // Assert
      expect(WebSocket).toHaveBeenCalledWith(expect.stringContaining('testnet'), expect.any(Object));
    });
    
    test('should use mainnet URL when testnet is false', () => {
      // Arrange
      deltaWs = new DeltaWebSocketService({ testnet: false });
      
      // Act
      deltaWs.connect();
      
      // Assert
      expect(WebSocket).toHaveBeenCalledWith(expect.not.stringContaining('testnet'), expect.any(Object));
    });
    
    test('should emit connected event when connection opens', () => {
      // Arrange
      const onConnectedMock = jest.fn();
      deltaWs.on('connected', onConnectedMock);
      
      // Act
      deltaWs.connect();
      mockSocket.emit('open');
      
      // Assert
      expect(onConnectedMock).toHaveBeenCalled();
    });
    
    test('should emit error event when connection fails', () => {
      // Arrange
      const onErrorMock = jest.fn();
      deltaWs.on('error', onErrorMock);
      
      // Act
      deltaWs.connect();
      mockSocket.emit('error', new Error('Connection failed'));
      
      // Assert
      expect(onErrorMock).toHaveBeenCalled();
    });
    
    test('should emit close event when connection closes', () => {
      // Arrange
      const onCloseMock = jest.fn();
      deltaWs.on('close', onCloseMock);
      
      // Act
      deltaWs.connect();
      mockSocket.emit('close');
      
      // Assert
      expect(onCloseMock).toHaveBeenCalled();
    });
    
    test('should handle connection close with reconnect', () => {
      // Arrange
      deltaWs.autoReconnect = true;
      jest.spyOn(deltaWs, 'connect');
      
      // Act
      deltaWs.connect();
      WebSocket.mockClear(); // Clear the first connect call
      mockSocket.emit('close');
      
      // Fast-forward timers to trigger reconnect
      jest.advanceTimersByTime(5000);
      
      // Assert
      expect(WebSocket).toHaveBeenCalledTimes(1);
    });
  });
  
  describe('Subscription Handling', () => {
    beforeEach(() => {
      // Setup connection
      deltaWs.connect();
      mockSocket.emit('open');
    });
    
    test('should subscribe to ticker channel', () => {
      // Act
      deltaWs.subscribeTicker('BTCUSDT');
      
      // Assert
      expect(mockSocket.send).toHaveBeenCalledWith(
        expect.stringContaining('"name":"subscribe"')
      );
      expect(mockSocket.send).toHaveBeenCalledWith(
        expect.stringContaining('"symbols":["BTCUSDT"]')
      );
      expect(mockSocket.send).toHaveBeenCalledWith(
        expect.stringContaining('"channels":["v2/ticker"]')
      );
    });
    
    test('should subscribe to orderbook channel', () => {
      // Act
      deltaWs.subscribeOrderbook('BTCUSDT');
      
      // Assert
      expect(mockSocket.send).toHaveBeenCalledWith(
        expect.stringContaining('"channels":["v2/orderbook"]')
      );
    });
    
    test('should subscribe to trades channel', () => {
      // Act
      deltaWs.subscribeTrades('BTCUSDT');
      
      // Assert
      expect(mockSocket.send).toHaveBeenCalledWith(
        expect.stringContaining('"channels":["v2/trades"]')
      );
    });
    
    test('should unsubscribe from channel', () => {
      // Act
      deltaWs.unsubscribe('v2/ticker', 'BTCUSDT');
      
      // Assert
      expect(mockSocket.send).toHaveBeenCalledWith(
        expect.stringContaining('"name":"unsubscribe"')
      );
    });
  });
  
  describe('Message Handling', () => {
    beforeEach(() => {
      // Setup connection
      deltaWs.connect();
      mockSocket.emit('open');
    });
    
    test('should handle ticker message', () => {
      // Arrange
      const tickerHandler = jest.fn();
      deltaWs.on('ticker', tickerHandler);
      
      // Act
      mockSocket.emit('message', JSON.stringify({
        type: 'update',
        channel: 'v2/ticker',
        symbol: 'BTCUSDT',
        data: {
          mark_price: '45000',
          last_price: '44999',
          timestamp: Date.now()
        }
      }));
      
      // Assert
      expect(tickerHandler).toHaveBeenCalledWith({
        symbol: 'BTCUSDT',
        mark_price: '45000',
        last_price: '44999',
        timestamp: expect.any(Number)
      });
    });
    
    test('should handle orderbook message', () => {
      // Arrange
      const orderbookHandler = jest.fn();
      deltaWs.on('orderbook', orderbookHandler);
      
      // Act
      mockSocket.emit('message', JSON.stringify({
        type: 'update',
        channel: 'v2/orderbook',
        symbol: 'BTCUSDT',
        data: {
          asks: [['45000', '1.2']],
          bids: [['44999', '0.8']],
          timestamp: Date.now()
        }
      }));
      
      // Assert
      expect(orderbookHandler).toHaveBeenCalledWith({
        symbol: 'BTCUSDT',
        asks: [['45000', '1.2']],
        bids: [['44999', '0.8']],
        timestamp: expect.any(Number)
      });
    });
    
    test('should handle trades message', () => {
      // Arrange
      const tradesHandler = jest.fn();
      deltaWs.on('trades', tradesHandler);
      
      // Act
      mockSocket.emit('message', JSON.stringify({
        type: 'update',
        channel: 'v2/trades',
        symbol: 'BTCUSDT',
        data: [
          {
            price: '45000',
            size: '0.1',
            side: 'buy',
            timestamp: Date.now()
          }
        ]
      }));
      
      // Assert
      expect(tradesHandler).toHaveBeenCalledWith({
        symbol: 'BTCUSDT',
        trades: [
          {
            price: '45000',
            size: '0.1',
            side: 'buy',
            timestamp: expect.any(Number)
          }
        ]
      });
    });
    
    test('should handle ping message with pong response', () => {
      // Act
      mockSocket.emit('message', JSON.stringify({
        type: 'ping',
        timestamp: Date.now()
      }));
      
      // Assert
      expect(mockSocket.send).toHaveBeenCalledWith(
        expect.stringContaining('"type":"pong"')
      );
    });
    
    test('should handle error message', () => {
      // Arrange
      const errorHandler = jest.fn();
      deltaWs.on('error', errorHandler);
      
      // Act
      mockSocket.emit('message', JSON.stringify({
        type: 'error',
        message: 'Subscription failed',
        code: 1001
      }));
      
      // Assert
      expect(errorHandler).toHaveBeenCalledWith(
        expect.objectContaining({
          message: 'Subscription failed',
          code: 1001
        })
      );
    });
  });
  
  describe('Authentication', () => {
    beforeEach(() => {
      // Setup connection
      deltaWs.connect();
      mockSocket.emit('open');
    });
    
    test('should send authentication message with API keys', () => {
      // Act
      deltaWs.authenticate('test-api-key', 'test-api-secret');
      
      // Assert
      expect(mockSocket.send).toHaveBeenCalledWith(
        expect.stringContaining('"name":"auth"')
      );
      expect(mockSocket.send).toHaveBeenCalledWith(
        expect.stringContaining('"api-key":"test-api-key"')
      );
      expect(mockSocket.send).toHaveBeenCalledWith(
        expect.stringContaining('"signature"')
      );
    });
    
    test('should emit authenticated event when auth succeeds', () => {
      // Arrange
      const authHandler = jest.fn();
      deltaWs.on('authenticated', authHandler);
      
      // Act
      deltaWs.authenticate('test-api-key', 'test-api-secret');
      mockSocket.emit('message', JSON.stringify({
        type: 'response',
        name: 'auth',
        success: true
      }));
      
      // Assert
      expect(authHandler).toHaveBeenCalled();
    });
    
    test('should emit error when auth fails', () => {
      // Arrange
      const errorHandler = jest.fn();
      deltaWs.on('error', errorHandler);
      
      // Act
      deltaWs.authenticate('test-api-key', 'test-api-secret');
      mockSocket.emit('message', JSON.stringify({
        type: 'response',
        name: 'auth',
        success: false,
        message: 'Invalid API key'
      }));
      
      // Assert
      expect(errorHandler).toHaveBeenCalledWith(
        expect.objectContaining({
          message: 'Authentication failed: Invalid API key'
        })
      );
    });
  });
  
  describe('Heartbeat and Reconnect', () => {
    beforeEach(() => {
      // Setup fake timers
      jest.useFakeTimers();
      
      // Setup connection
      deltaWs.connect();
      mockSocket.emit('open');
    });
    
    afterEach(() => {
      jest.useRealTimers();
    });
    
    test('should handle heartbeat mechanism', () => {
      // Arrange
      jest.spyOn(deltaWs, '_startHeartbeat');
      
      // Act
      deltaWs._startHeartbeat();
      jest.advanceTimersByTime(30000); // 30 seconds
      
      // Assert
      expect(mockSocket.send).toHaveBeenCalledWith(
        expect.stringContaining('"type":"ping"')
      );
    });
    
    test('should reconnect if no pong received', () => {
      // Arrange
      deltaWs.lastPongAt = Date.now() - 60000; // 60 seconds ago
      jest.spyOn(deltaWs, 'reconnect');
      
      // Act
      deltaWs._checkConnection();
      
      // Assert
      expect(deltaWs.reconnect).toHaveBeenCalled();
    });
    
    test('should resubscribe to previous channels after reconnect', () => {
      // Arrange
      deltaWs.subscribeTicker('BTCUSDT');
      mockSocket.send.mockClear();
      
      // Act
      deltaWs.reconnect();
      mockSocket = WebSocket.mock.results[1]?.value;
      mockSocket.emit('open');
      
      // Assert
      expect(mockSocket.send).toHaveBeenCalledWith(
        expect.stringContaining('"channels":["v2/ticker"]')
      );
    });
  });
}); 