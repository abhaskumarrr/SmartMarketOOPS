import WebSocketService, { ConnectionStatus } from '../../lib/websocketService';
import { Socket } from 'socket.io-client';

// Mock socket.io-client
jest.mock('socket.io-client', () => {
  const mockSocketOn = jest.fn();
  const mockSocketEmit = jest.fn();
  const mockSocketDisconnect = jest.fn();
  const mockSocketRemoveAllListeners = jest.fn();

  const mockSocket = {
    on: mockSocketOn,
    emit: mockSocketEmit,
    disconnect: mockSocketDisconnect,
    removeAllListeners: mockSocketRemoveAllListeners,
  };

  const mockIO = jest.fn(() => mockSocket);
  
  return {
    io: mockIO,
    Socket: jest.fn(),
  };
});

describe('WebSocketService', () => {
  let service: WebSocketService;
  let originalSetTimeout: typeof setTimeout;
  let originalClearTimeout: typeof clearTimeout;
  let originalSetInterval: typeof setInterval;
  let originalClearInterval: typeof clearInterval;
  let mockSetTimeout: jest.Mock;
  let mockClearTimeout: jest.Mock;
  let mockSetInterval: jest.Mock;
  let mockClearInterval: jest.Mock;
  const mockIO = require('socket.io-client').io;
  
  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();
    
    // Mock timers
    originalSetTimeout = global.setTimeout;
    originalClearTimeout = global.clearTimeout;
    originalSetInterval = global.setInterval;
    originalClearInterval = global.clearInterval;
    
    mockSetTimeout = jest.fn().mockImplementation((fn, delay) => {
      return 123 as unknown as NodeJS.Timeout; // Mock timer id
    });
    mockClearTimeout = jest.fn();
    mockSetInterval = jest.fn().mockImplementation((fn, delay) => {
      return 456 as unknown as NodeJS.Timeout; // Mock timer id
    });
    mockClearInterval = jest.fn();
    
    global.setTimeout = mockSetTimeout as unknown as typeof setTimeout;
    global.clearTimeout = mockClearTimeout as unknown as typeof clearTimeout;
    global.setInterval = mockSetInterval as unknown as typeof setInterval;
    global.clearInterval = mockClearInterval as unknown as typeof clearInterval;
    
    // Create a new instance before each test
    (WebSocketService as any).instance = undefined;
    service = WebSocketService.getInstance({ autoConnect: false });
  });
  
  afterEach(() => {
    // Restore original timers
    global.setTimeout = originalSetTimeout;
    global.clearTimeout = originalClearTimeout;
    global.setInterval = originalSetInterval;
    global.clearInterval = originalClearInterval;
  });
  
  test('getInstance returns the same instance', () => {
    const instance1 = WebSocketService.getInstance();
    const instance2 = WebSocketService.getInstance();
    
    expect(instance1).toBe(instance2);
  });
  
  test('connect initializes socket with correct options', async () => {
    await service.connect();
    
    expect(mockIO).toHaveBeenCalledWith(expect.any(String), {
      transports: ['websocket'],
      reconnection: false,
      timeout: 10000,
      forceNew: true
    });
  });
  
  test('connect sets up event handlers', async () => {
    await service.connect();
    
    const mockSocket = mockIO.mock.results[0].value;
    
    expect(mockSocket.on).toHaveBeenCalledWith('connect', expect.any(Function));
    expect(mockSocket.on).toHaveBeenCalledWith('disconnect', expect.any(Function));
    expect(mockSocket.on).toHaveBeenCalledWith('connect_error', expect.any(Function));
    expect(mockSocket.on).toHaveBeenCalledWith('error', expect.any(Function));
    expect(mockSocket.on).toHaveBeenCalledWith('pong', expect.any(Function));
    
    // Check for data event handlers
    expect(mockSocket.on).toHaveBeenCalledWith('market:data', expect.any(Function));
    expect(mockSocket.on).toHaveBeenCalledWith('trade:executed', expect.any(Function));
    expect(mockSocket.on).toHaveBeenCalledWith('prediction:new', expect.any(Function));
    expect(mockSocket.on).toHaveBeenCalledWith('signal:new', expect.any(Function));
    expect(mockSocket.on).toHaveBeenCalledWith('alert:triggered', expect.any(Function));
    expect(mockSocket.on).toHaveBeenCalledWith('status:update', expect.any(Function));
  });
  
  test('subscribe sends subscription to socket', async () => {
    await service.connect();
    
    const mockSocket = mockIO.mock.results[0].value;
    service.subscribe('market', { symbol: 'BTCUSD' });
    
    expect(mockSocket.emit).toHaveBeenCalledWith('subscribe:market', { symbol: 'BTCUSD' });
  });
  
  test('unsubscribe sends unsubscription to socket', async () => {
    await service.connect();
    
    const mockSocket = mockIO.mock.results[0].value;
    service.unsubscribe('market', { symbol: 'BTCUSD' });
    
    expect(mockSocket.emit).toHaveBeenCalledWith('unsubscribe:market', { symbol: 'BTCUSD' });
  });
  
  test('disconnect closes socket and clears state', async () => {
    await service.connect();
    
    const mockSocket = mockIO.mock.results[0].value;
    service.disconnect();
    
    expect(mockSocket.removeAllListeners).toHaveBeenCalled();
    expect(mockSocket.disconnect).toHaveBeenCalled();
    expect(service.getStatus()).toBe(ConnectionStatus.DISCONNECTED);
    expect(mockClearInterval).toHaveBeenCalled(); // Should clear ping timer
  });
  
  test('getStatus returns current connection status', () => {
    expect(service.getStatus()).toBe(ConnectionStatus.DISCONNECTED);
  });
  
  test('startPingPong sets up ping interval', async () => {
    await service.connect();
    
    // Simulate connect event to trigger startPingPong
    const connectHandler = mockIO.mock.results[0].value.on.mock.calls.find(
      call => call[0] === 'connect'
    )[1];
    
    connectHandler();
    
    // Should set up ping interval
    expect(mockSetInterval).toHaveBeenCalled();
    expect(mockSetInterval.mock.calls[0][1]).toBe(30000); // Default pingInterval
  });
  
  test('forceReconnect resets connection and attempts reconnection', async () => {
    // Set up a mock implementation for connect
    const connectSpy = jest.spyOn(service, 'connect').mockResolvedValue();
    
    await service.forceReconnect();
    
    expect(connectSpy).toHaveBeenCalled();
  });
  
  test('handleReconnect uses exponential backoff', () => {
    // Access private method using type casting
    const handleReconnect = (service as any).handleReconnect.bind(service);
    
    // First attempt
    handleReconnect();
    expect(mockSetTimeout).toHaveBeenCalledTimes(1);
    
    // Should be called with a delay
    expect(mockSetTimeout.mock.calls[0][1]).toBeGreaterThan(0);
    
    // Simulate more reconnection attempts
    (service as any).reconnectAttempts = 3;
    handleReconnect();
    
    // Second delay should be longer due to exponential backoff
    const firstDelay = mockSetTimeout.mock.calls[0][1];
    const secondDelay = mockSetTimeout.mock.calls[1][1];
    expect(secondDelay).toBeGreaterThan(firstDelay);
  });
  
  test('checkConnection detects stale connections and triggers reconnect', () => {
    // Mock private properties and methods
    (service as any).status = ConnectionStatus.CONNECTED;
    (service as any).lastPongTime = Date.now() - 100000; // Set last pong to be very old
    (service as any).pingTimer = true; // Simulate active ping timer
    (service as any).handleReconnect = jest.fn();
    
    service.checkConnection();
    
    // Should call handleReconnect for stale connection
    expect((service as any).handleReconnect).toHaveBeenCalled();
  });
}); 