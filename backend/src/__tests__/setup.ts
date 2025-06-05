/**
 * Jest Test Setup
 * Global setup for all tests
 */

import { config } from 'dotenv';
import { initializeTestHelpers, resetTestDatabase } from './helpers/testHelpers';

// Load test environment variables
config({ path: '.env.test' });

// Set test environment
process.env.NODE_ENV = 'test';
process.env.JWT_SECRET = 'test-jwt-secret';
process.env.BCRYPT_ROUNDS = '4'; // Faster hashing for tests

// Global test timeout
jest.setTimeout(60000);

// Initialize test helpers
beforeAll(async () => {
  initializeTestHelpers();
});

// Clean up after each test
afterEach(async () => {
  // Clear all mocks
  jest.clearAllMocks();
});

// Global cleanup
afterAll(async () => {
  // Reset database to clean state
  await resetTestDatabase();
});

// Mock console methods to reduce noise in tests
const originalConsole = { ...console };

beforeAll(() => {
  // Mock console.log, console.warn, etc. but keep console.error for debugging
  console.log = jest.fn();
  console.warn = jest.fn();
  console.info = jest.fn();
  console.debug = jest.fn();
});

afterAll(() => {
  // Restore original console methods
  Object.assign(console, originalConsole);
});

// Global error handler for unhandled promises
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

// Mock external services
jest.mock('ioredis', () => {
  return jest.fn().mockImplementation(() => ({
    on: jest.fn(),
    get: jest.fn(),
    set: jest.fn(),
    setex: jest.fn(),
    del: jest.fn(),
    exists: jest.fn(),
    ttl: jest.fn(),
    mget: jest.fn(),
    mset: jest.fn(),
    sadd: jest.fn(),
    smembers: jest.fn(),
    keys: jest.fn(),
    flushdb: jest.fn(),
    info: jest.fn(),
    quit: jest.fn(),
    pipeline: jest.fn(() => ({
      del: jest.fn(),
      set: jest.fn(),
      setex: jest.fn(),
      exec: jest.fn().mockResolvedValue([]),
    })),
  }));
});

// Mock WebSocket
jest.mock('ws', () => {
  return {
    WebSocketServer: jest.fn().mockImplementation(() => ({
      on: jest.fn(),
      close: jest.fn(),
    })),
    WebSocket: jest.fn().mockImplementation(() => ({
      on: jest.fn(),
      send: jest.fn(),
      close: jest.fn(),
      readyState: 1,
    })),
  };
});

// Mock external APIs
jest.mock('axios', () => ({
  default: {
    get: jest.fn(),
    post: jest.fn(),
    put: jest.fn(),
    delete: jest.fn(),
    patch: jest.fn(),
    create: jest.fn(() => ({
      get: jest.fn(),
      post: jest.fn(),
      put: jest.fn(),
      delete: jest.fn(),
      patch: jest.fn(),
    })),
  },
  get: jest.fn(),
  post: jest.fn(),
  put: jest.fn(),
  delete: jest.fn(),
  patch: jest.fn(),
}));

// Mock file system operations
jest.mock('fs', () => ({
  ...jest.requireActual('fs'),
  writeFileSync: jest.fn(),
  readFileSync: jest.fn(),
  existsSync: jest.fn(() => true),
  mkdirSync: jest.fn(),
  createWriteStream: jest.fn(() => ({
    write: jest.fn(),
    end: jest.fn(),
    on: jest.fn(),
  })),
}));

// Global test utilities
global.testUtils = {
  // Mock data generators
  generateMockUser: () => ({
    id: 'test-user-id',
    email: 'test@example.com',
    name: 'Test User',
    role: 'user',
    createdAt: new Date(),
    updatedAt: new Date(),
  }),

  generateMockBot: () => ({
    id: 'test-bot-id',
    name: 'Test Bot',
    symbol: 'BTCUSD',
    strategy: 'ML_PREDICTION',
    timeframe: '1h',
    isActive: false,
    parameters: {},
    userId: 'test-user-id',
    createdAt: new Date(),
    updatedAt: new Date(),
  }),

  generateMockMarketData: () => ({
    symbol: 'BTCUSD',
    timestamp: new Date(),
    open: 50000,
    high: 51000,
    low: 49000,
    close: 50500,
    volume: 1000000,
  }),

  // Test helpers
  sleep: (ms: number) => new Promise(resolve => setTimeout(resolve, ms)),
  
  randomString: (length: number = 10) => {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let result = '';
    for (let i = 0; i < length; i++) {
      result += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return result;
  },

  randomEmail: () => `test-${Math.random().toString(36).substr(2, 9)}@example.com`,
  
  // Mock API responses
  mockApiSuccess: (data: any) => ({
    success: true,
    data,
    message: 'Success',
  }),

  mockApiError: (message: string = 'Error') => ({
    success: false,
    message,
    error: message,
  }),
};

// Extend Jest matchers
expect.extend({
  toBeValidDate(received) {
    const pass = received instanceof Date && !isNaN(received.getTime());
    if (pass) {
      return {
        message: () => `expected ${received} not to be a valid date`,
        pass: true,
      };
    } else {
      return {
        message: () => `expected ${received} to be a valid date`,
        pass: false,
      };
    }
  },

  toBeValidUUID(received) {
    const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
    const pass = typeof received === 'string' && uuidRegex.test(received);
    if (pass) {
      return {
        message: () => `expected ${received} not to be a valid UUID`,
        pass: true,
      };
    } else {
      return {
        message: () => `expected ${received} to be a valid UUID`,
        pass: false,
      };
    }
  },

  toBeWithinRange(received, floor, ceiling) {
    const pass = received >= floor && received <= ceiling;
    if (pass) {
      return {
        message: () => `expected ${received} not to be within range ${floor} - ${ceiling}`,
        pass: true,
      };
    } else {
      return {
        message: () => `expected ${received} to be within range ${floor} - ${ceiling}`,
        pass: false,
      };
    }
  },
});

// Type declarations for global utilities
declare global {
  namespace jest {
    interface Matchers<R> {
      toBeValidDate(): R;
      toBeValidUUID(): R;
      toBeWithinRange(floor: number, ceiling: number): R;
    }
  }

  var testUtils: {
    generateMockUser: () => any;
    generateMockBot: () => any;
    generateMockMarketData: () => any;
    sleep: (ms: number) => Promise<void>;
    randomString: (length?: number) => string;
    randomEmail: () => string;
    mockApiSuccess: (data: any) => any;
    mockApiError: (message?: string) => any;
  };
}
