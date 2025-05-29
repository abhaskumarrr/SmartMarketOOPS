import { PrismaClient } from '@prisma/client';
import { CircuitBreakerService } from '../../../src/services/trading/circuitBreakerService';
import { 
  CircuitBreakerStatus,
  CircuitBreakerType
} from '../../../src/types/risk';

// Mock Prisma
jest.mock('@prisma/client', () => {
  const mockPrismaClient = {
    riskSettings: {
      findUnique: jest.fn(),
    },
    riskAlert: {
      create: jest.fn(),
    },
    position: {
      findMany: jest.fn(),
      count: jest.fn(),
      aggregate: jest.fn(),
    },
    circuitBreaker: {
      findFirst: jest.fn(),
      create: jest.fn(),
      update: jest.fn(),
      findMany: jest.fn(),
    },
    $transaction: jest.fn((callback) => callback(mockPrismaClient)),
  };
  
  return {
    PrismaClient: jest.fn(() => mockPrismaClient),
  };
});

describe('CircuitBreakerService', () => {
  let circuitBreakerService: CircuitBreakerService;
  let prisma: any;
  
  beforeEach(() => {
    jest.clearAllMocks();
    
    // Initialize service with mocked Prisma
    prisma = new PrismaClient();
    circuitBreakerService = new CircuitBreakerService(prisma);
  });
  
  describe('checkAndActivateCircuitBreakers', () => {
    it('should activate drawdown circuit breaker when threshold is exceeded', async () => {
      // Given
      const userId = 'user123';
      const accountBalance = 100000;
      const drawdownPercentage = 11; // 11% drawdown
      
      // Mock risk settings with 10% max drawdown
      prisma.riskSettings.findUnique.mockResolvedValue({
        id: 'risk1',
        userId,
        maxDrawdown: 10,
        enabledCircuitBreakers: true
      });
      
      // Mock no existing circuit breakers
      prisma.circuitBreaker.findFirst.mockResolvedValue(null);
      
      // Mock circuit breaker creation
      prisma.circuitBreaker.create.mockResolvedValue({
        id: 'cb1',
        userId,
        type: CircuitBreakerType.MAX_DRAWDOWN_BREAKER,
        status: CircuitBreakerStatus.ACTIVE,
        activationReason: 'Drawdown exceeded maximum threshold of 10%',
        activatedAt: expect.any(Date),
        resetAt: null,
        metadata: {
          currentDrawdown: 11,
          maxAllowed: 10
        }
      });
      
      // Mock positions aggregate for unrealized P&L
      prisma.position.aggregate.mockResolvedValue({
        _sum: {
          unrealizedPnl: -11000, // -11% of account balance
        }
      });
      
      // When
      const result = await circuitBreakerService.checkAndActivateCircuitBreakers(userId, accountBalance, drawdownPercentage);
      
      // Then
      expect(result.activated).toBe(true);
      expect(result.circuitBreakers).toHaveLength(1);
      expect(result.circuitBreakers[0].type).toBe(CircuitBreakerType.MAX_DRAWDOWN_BREAKER);
      expect(result.circuitBreakers[0].status).toBe(CircuitBreakerStatus.ACTIVE);
      expect(prisma.circuitBreaker.create).toHaveBeenCalled();
    });
    
    it('should not activate circuit breakers when within thresholds', async () => {
      // Given
      const userId = 'user123';
      const accountBalance = 100000;
      const drawdownPercentage = 5; // 5% drawdown
      
      // Mock risk settings with 10% max drawdown
      prisma.riskSettings.findUnique.mockResolvedValue({
        id: 'risk1',
        userId,
        maxDrawdown: 10,
        enabledCircuitBreakers: true
      });
      
      // Mock no existing circuit breakers
      prisma.circuitBreaker.findFirst.mockResolvedValue(null);
      
      // Mock positions aggregate for unrealized P&L
      prisma.position.aggregate.mockResolvedValue({
        _sum: {
          unrealizedPnl: -5000, // -5% of account balance
        }
      });
      
      // When
      const result = await circuitBreakerService.checkAndActivateCircuitBreakers(userId, accountBalance, drawdownPercentage);
      
      // Then
      expect(result.activated).toBe(false);
      expect(result.circuitBreakers).toHaveLength(0);
      expect(prisma.circuitBreaker.create).not.toHaveBeenCalled();
    });
    
    it('should not activate circuit breakers when they are disabled', async () => {
      // Given
      const userId = 'user123';
      const accountBalance = 100000;
      const drawdownPercentage = 15; // 15% drawdown
      
      // Mock risk settings with circuit breakers disabled
      prisma.riskSettings.findUnique.mockResolvedValue({
        id: 'risk1',
        userId,
        maxDrawdown: 10,
        enabledCircuitBreakers: false // Disabled
      });
      
      // When
      const result = await circuitBreakerService.checkAndActivateCircuitBreakers(userId, accountBalance, drawdownPercentage);
      
      // Then
      expect(result.activated).toBe(false);
      expect(result.circuitBreakers).toHaveLength(0);
      expect(prisma.circuitBreaker.create).not.toHaveBeenCalled();
    });
  });
  
  describe('resetCircuitBreaker', () => {
    it('should reset an active circuit breaker', async () => {
      // Given
      const circuitBreakerId = 'cb1';
      
      // Mock finding the circuit breaker
      prisma.circuitBreaker.findFirst.mockResolvedValue({
        id: circuitBreakerId,
        userId: 'user123',
        type: CircuitBreakerType.MAX_DRAWDOWN_BREAKER,
        status: CircuitBreakerStatus.ACTIVE,
        activationReason: 'Drawdown exceeded maximum threshold',
        activatedAt: new Date(Date.now() - 3600000), // 1 hour ago
        resetAt: null
      });
      
      // Mock update call
      prisma.circuitBreaker.update.mockResolvedValue({
        id: circuitBreakerId,
        userId: 'user123',
        type: CircuitBreakerType.MAX_DRAWDOWN_BREAKER,
        status: CircuitBreakerStatus.RESET,
        activationReason: 'Drawdown exceeded maximum threshold',
        activatedAt: new Date(Date.now() - 3600000),
        resetAt: expect.any(Date),
        resetReason: 'Manually reset by user'
      });
      
      // When
      const result = await circuitBreakerService.resetCircuitBreaker(circuitBreakerId, 'Manually reset by user');
      
      // Then
      expect(result.status).toBe(CircuitBreakerStatus.RESET);
      expect(result.resetAt).toBeDefined();
      expect(result.resetReason).toBe('Manually reset by user');
      expect(prisma.circuitBreaker.update).toHaveBeenCalledWith({
        where: { id: circuitBreakerId },
        data: {
          status: CircuitBreakerStatus.RESET,
          resetAt: expect.any(Date),
          resetReason: 'Manually reset by user'
        }
      });
    });
    
    it('should throw error when resetting non-existent circuit breaker', async () => {
      // Given
      const circuitBreakerId = 'non-existent-id';
      
      // Mock finding the circuit breaker - returns null
      prisma.circuitBreaker.findFirst.mockResolvedValue(null);
      
      // When & Then
      await expect(
        circuitBreakerService.resetCircuitBreaker(circuitBreakerId, 'Manually reset by user')
      ).rejects.toThrow('Circuit breaker not found');
      
      expect(prisma.circuitBreaker.update).not.toHaveBeenCalled();
    });
  });
  
  describe('getActiveCircuitBreakers', () => {
    it('should return all active circuit breakers for a user', async () => {
      // Given
      const userId = 'user123';
      const mockBreakers = [
        {
          id: 'cb1',
          userId,
          type: CircuitBreakerType.MAX_DRAWDOWN_BREAKER,
          status: CircuitBreakerStatus.ACTIVE,
          activationReason: 'Drawdown exceeded maximum threshold',
          activatedAt: new Date(),
          resetAt: null
        },
        {
          id: 'cb2',
          userId,
          type: CircuitBreakerType.DAILY_LOSS_BREAKER,
          status: CircuitBreakerStatus.ACTIVE,
          activationReason: 'Daily loss exceeded maximum threshold',
          activatedAt: new Date(),
          resetAt: null
        }
      ];
      
      // Mock findMany call
      prisma.circuitBreaker.findMany.mockResolvedValue(mockBreakers);
      
      // When
      const result = await circuitBreakerService.getActiveCircuitBreakers(userId);
      
      // Then
      expect(result).toHaveLength(2);
      expect(result[0].type).toBe(CircuitBreakerType.MAX_DRAWDOWN_BREAKER);
      expect(result[1].type).toBe(CircuitBreakerType.DAILY_LOSS_BREAKER);
      expect(prisma.circuitBreaker.findMany).toHaveBeenCalledWith({
        where: {
          userId,
          status: CircuitBreakerStatus.ACTIVE
        }
      });
    });
    
    it('should return empty array when no active circuit breakers exist', async () => {
      // Given
      const userId = 'user123';
      
      // Mock findMany call - returns empty array
      prisma.circuitBreaker.findMany.mockResolvedValue([]);
      
      // When
      const result = await circuitBreakerService.getActiveCircuitBreakers(userId);
      
      // Then
      expect(result).toHaveLength(0);
      expect(prisma.circuitBreaker.findMany).toHaveBeenCalledWith({
        where: {
          userId,
          status: CircuitBreakerStatus.ACTIVE
        }
      });
    });
  });
}); 