/**
 * Jest Setup File
 * Configures testing environment and global mocks
 */

import '@testing-library/jest-dom';
import 'jest-fetch-mock';

// Enable fetch mocks
require('jest-fetch-mock').enableMocks();

// Global test utilities
global.testUtils = {
  // Mock user data
  mockUser: {
    id: 'test-user-id',
    email: 'test@example.com',
    name: 'Test User',
    role: 'user',
  },

  // Mock bot data
  mockBot: {
    id: 'test-bot-id',
    name: 'Test Bot',
    symbol: 'BTCUSD',
    strategy: 'ML_PREDICTION',
    timeframe: '1h',
    isActive: false,
    parameters: {},
  },

  // Mock market data
  mockMarketData: {
    symbol: 'BTCUSD',
    price: 50000,
    change24h: 2.5,
    volume24h: 1000000,
    timestamp: Date.now(),
  },

  // Mock API responses
  mockApiResponse: (data, success = true) => ({
    success,
    data,
    message: success ? 'Success' : 'Error',
  }),
};

// Mock next/router
jest.mock('next/router', () => ({
  useRouter: () => ({
    route: '/',
    pathname: '',
    query: {},
    asPath: '',
    push: jest.fn(),
    replace: jest.fn(),
    reload: jest.fn(),
    back: jest.fn(),
    prefetch: jest.fn(() => Promise.resolve()),
    beforePopState: jest.fn(),
    events: {
      on: jest.fn(),
      off: jest.fn(),
      emit: jest.fn(),
    },
    isFallback: false,
  }),
}));

// Mock next/image
jest.mock('next/image', () => ({
  __esModule: true,
  default: (props) => {
    // eslint-disable-next-line jsx-a11y/alt-text
    return <img {...props} />;
  },
}));

// Mock WebSocket service
jest.mock('./lib/websocketService', () => {
  const EventEmitter = require('events');
  
  const ConnectionStatus = {
    DISCONNECTED: 'disconnected',
    CONNECTING: 'connecting',
    CONNECTED: 'connected',
    ERROR: 'error'
  };
  
  class MockWebSocketService extends EventEmitter {
    constructor() {
      super();
      this.status = ConnectionStatus.DISCONNECTED;
      this.subscriptions = new Set();
    }
    
    static getInstance() {
      if (!MockWebSocketService.instance) {
        MockWebSocketService.instance = new MockWebSocketService();
      }
      return MockWebSocketService.instance;
    }
    
    connect() {
      this.status = ConnectionStatus.CONNECTED;
      this.emit('status', this.status);
      return Promise.resolve();
    }
    
    disconnect() {
      this.status = ConnectionStatus.DISCONNECTED;
      this.emit('status', this.status);
    }
    
    subscribe(channel) {
      this.subscriptions.add(channel);
    }
    
    unsubscribe(channel) {
      this.subscriptions.delete(channel);
    }
    
    getStatus() {
      return this.status;
    }
    
    // Helper to simulate receiving data
    mockReceiveData(event, data) {
      this.emit(event, data);
    }
    
    // Helper to simulate status change
    mockChangeStatus(status) {
      this.status = status;
      this.emit('status', status);
    }
  }
  
  return {
    __esModule: true,
    default: MockWebSocketService,
    ConnectionStatus,
  };
}); 