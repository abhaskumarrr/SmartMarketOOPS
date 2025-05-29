/**
 * Order Execution Service Tests
 */

import { OrderExecutionService } from '../../../src/services/trading/orderExecutionService';
import { 
  OrderExecutionRequest, 
  OrderExecutionStatus, 
  OrderType, 
  OrderSide, 
  ExecutionSource 
} from '../../../src/types/orderExecution';

// Mock dependencies
jest.mock('@prisma/client', () => {
  const mockCreate = jest.fn().mockResolvedValue({ id: 'mocked-id' });
  return {
    PrismaClient: jest.fn().mockImplementation(() => ({
      order: {
        create: mockCreate,
        findMany: jest.fn().mockResolvedValue([]),
        findUnique: jest.fn().mockResolvedValue(null)
      }
    }))
  };
});

jest.mock('../../../src/services/trading/riskManagementService', () => {
  return {
    RiskManagementService: jest.fn().mockImplementation(() => ({
      getRiskSettings: jest.fn().mockResolvedValue({
        maxPositionSize: 10,
        maxLeverage: 5
      }),
      calculatePositionSize: jest.fn().mockResolvedValue({
        recommendedSize: 1.0,
        maxPositionSize: 2.0,
        leverage: 3,
        accountBalance: 10000,
        riskAmount: 100
      })
    }))
  };
});

jest.mock('../../../src/utils/logger', () => {
  return {
    createLogger: jest.fn().mockReturnValue({
      info: jest.fn(),
      warn: jest.fn(),
      error: jest.fn(),
      debug: jest.fn()
    })
  };
});

// Mock exchange connector
const mockExchangeConnector = {
  createOrder: jest.fn().mockResolvedValue({
    id: 'exchange-order-id',
    status: 'filled',
    symbol: 'BTC/USDT',
    type: 'limit',
    side: 'buy',
    amount: 1.0,
    price: 50000,
    filled: 1.0,
    remaining: 0,
    fee: { cost: 2.5, currency: 'USDT' },
    timestamp: Date.now()
  }),
  cancelOrder: jest.fn().mockResolvedValue({ id: 'exchange-order-id', status: 'canceled' }),
  fetchOrder: jest.fn().mockResolvedValue({
    id: 'exchange-order-id',
    status: 'filled',
    symbol: 'BTC/USDT',
    type: 'limit',
    side: 'buy',
    amount: 1.0,
    price: 50000,
    filled: 1.0,
    remaining: 0,
    fee: { cost: 2.5, currency: 'USDT' },
    timestamp: Date.now()
  }),
  fetchOpenOrders: jest.fn().mockResolvedValue([]),
  fetchOrderHistory: jest.fn().mockResolvedValue([]),
  fetchTicker: jest.fn().mockResolvedValue({
    symbol: 'BTC/USDT',
    last: 50000,
    bid: 49900,
    ask: 50100,
    volume: 100
  })
};

// Mock the Delta Exchange service
jest.mock('../../../src/services/deltaExchangeService', () => {
  return {
    createDefaultService: jest.fn().mockReturnValue(mockExchangeConnector)
  };
});

describe('OrderExecutionService', () => {
  let service: OrderExecutionService;
  
  beforeEach(() => {
    jest.clearAllMocks();
    service = new OrderExecutionService();
  });
  
  it('should initialize properly', () => {
    expect(service).toBeDefined();
  });
  
  describe('executeOrder', () => {
    it('should execute a valid market order successfully', async () => {
      const request: OrderExecutionRequest = {
        symbol: 'BTC/USDT',
        type: OrderType.MARKET,
        side: OrderSide.BUY,
        quantity: 1.0,
        source: ExecutionSource.MANUAL,
        userId: 'user-123',
        exchangeId: 'delta'
      };
      
      const result = await service.executeOrder(request);
      
      expect(result).toBeDefined();
      expect(result.status).toBe(OrderExecutionStatus.FILLED);
      expect(result.symbol).toBe('BTC/USDT');
      expect(result.type).toBe(OrderType.MARKET);
      expect(result.side).toBe(OrderSide.BUY);
      expect(result.filledQuantity).toBe(1.0);
      expect(result.remainingQuantity).toBe(0);
      expect(result.fee).toBe(2.5);
      expect(result.feeCurrency).toBe('USDT');
      expect(result.exchangeOrderId).toBe('exchange-order-id');
      expect(mockExchangeConnector.createOrder).toHaveBeenCalledTimes(1);
    });
    
    it('should validate order requests and reject invalid ones', async () => {
      // Missing quantity
      const request: Partial<OrderExecutionRequest> = {
        symbol: 'BTC/USDT',
        type: OrderType.MARKET,
        side: OrderSide.BUY,
        quantity: 0, // Invalid quantity
        source: ExecutionSource.MANUAL,
        userId: 'user-123',
        exchangeId: 'delta'
      };
      
      const result = await service.executeOrder(request as OrderExecutionRequest);
      
      expect(result).toBeDefined();
      expect(result.status).toBe(OrderExecutionStatus.REJECTED);
      expect(result.error).toBeDefined();
      expect(result.error!.code).toBe('VALIDATION_FAILED');
      expect(mockExchangeConnector.createOrder).not.toHaveBeenCalled();
    });
    
    it('should perform risk checks and reject orders that exceed limits', async () => {
      const request: OrderExecutionRequest = {
        symbol: 'BTC/USDT',
        type: OrderType.MARKET,
        side: OrderSide.BUY,
        quantity: 5.0, // More than the recommendedSize from risk check
        source: ExecutionSource.MANUAL,
        userId: 'user-123',
        exchangeId: 'delta'
      };
      
      const result = await service.executeOrder(request);
      
      expect(result).toBeDefined();
      expect(result.status).toBe(OrderExecutionStatus.REJECTED);
      expect(result.error).toBeDefined();
      expect(result.error!.code).toBe('RISK_CHECK_FAILED');
      expect(mockExchangeConnector.createOrder).not.toHaveBeenCalled();
    });
  });
  
  describe('cancelOrder', () => {
    it('should cancel an order successfully', async () => {
      const result = await service.cancelOrder('order-id', 'delta', 'user-123');
      
      expect(result).toBe(true);
      expect(mockExchangeConnector.cancelOrder).toHaveBeenCalledTimes(1);
    });
  });
  
  describe('getOrder', () => {
    it('should fetch an order by ID', async () => {
      const result = await service.getOrder('order-id', 'delta', 'user-123');
      
      expect(result).toBeDefined();
      expect(result.status).toBe(OrderExecutionStatus.FILLED);
      expect(result.exchangeOrderId).toBe('exchange-order-id');
      expect(mockExchangeConnector.fetchOrder).toHaveBeenCalledTimes(1);
    });
  });
}); 