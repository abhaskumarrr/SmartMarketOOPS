// Optional: configure or set up a testing framework before each test.
// If you delete this file, remove `setupFilesAfterEnv` from `jest.config.js`

// Used for __tests__/testing-library.js
// Learn more: https://github.com/testing-library/jest-dom
import '@testing-library/jest-dom';

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