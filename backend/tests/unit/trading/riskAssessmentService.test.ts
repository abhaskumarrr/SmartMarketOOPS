import { PrismaClient } from '@prisma/client';
import { RiskAssessmentService } from '../../../src/services/trading/riskAssessmentService';
import { 
  RiskLevel,
  RiskAlertType,
  RiskAssessmentResult
} from '../../../src/types/risk';

// Mock Prisma
jest.mock('@prisma/client', () => {
  const mockPrismaClient = {
    riskSettings: {
      findUnique: jest.fn(),
    },
    riskAlert: {
      create: jest.fn(),
      findMany: jest.fn(),
    },
    position: {
      findMany: jest.fn(),
      count: jest.fn(),
      aggregate: jest.fn(),
    },
    tradingSignal: {
      findMany: jest.fn(),
    },
    $transaction: jest.fn((callback) => callback(mockPrismaClient)),
  };
  
  return {
    PrismaClient: jest.fn(() => mockPrismaClient),
  };
});

describe('RiskAssessmentService', () => {
  let riskAssessmentService: RiskAssessmentService;
  let prisma: any;
  
  beforeEach(() => {
    jest.clearAllMocks();
    
    // Initialize service with mocked Prisma
    prisma = new PrismaClient();
    riskAssessmentService = new RiskAssessmentService(prisma);
  });
  
  describe('assessTradeRisk', () => {
    it('should assess trade risk as LOW for small position with good setup', async () => {
      // Given
      const userId = 'user123';
      const tradeDetails = {
        symbol: 'BTC/USD',
        direction: 'long',
        entryPrice: 40000,
        positionSize: 0.01, // Small position
        stopLossPrice: 39000,
        takeProfitPrice: 43000,
        accountBalance: 100000
      };
      
      // Mock positions to return empty array (no existing positions)
      prisma.position.findMany.mockResolvedValue([]);
      
      // Mock risk settings
      prisma.riskSettings.findUnique.mockResolvedValue({
        id: 'risk1',
        userId,
        maxPositionSize: 0.1,
        maxDrawdown: 10,
        maxDailyLoss: 5,
        maxOpenPositions: 5,
        maxPositionsPerSymbol: 2,
        riskLevel: RiskLevel.MEDIUM
      });
      
      // When
      const result = await riskAssessmentService.assessTradeRisk(userId, tradeDetails);
      
      // Then
      expect(result.riskLevel).toBe(RiskLevel.LOW);
      expect(result.alerts).toHaveLength(0);
      expect(result.factors.positionSizeRisk).toBe(RiskLevel.LOW);
      expect(result.factors.portfolioConcentrationRisk).toBe(RiskLevel.LOW);
      expect(result.isWithinRiskTolerance).toBe(true);
    });
    
    it('should assess trade risk as HIGH for large position size', async () => {
      // Given
      const userId = 'user123';
      const tradeDetails = {
        symbol: 'BTC/USD',
        direction: 'long',
        entryPrice: 40000,
        positionSize: 0.2, // Large position (20% of account)
        stopLossPrice: 39000,
        takeProfitPrice: 43000,
        accountBalance: 100000
      };
      
      // Mock positions to return empty array (no existing positions)
      prisma.position.findMany.mockResolvedValue([]);
      
      // Mock risk settings
      prisma.riskSettings.findUnique.mockResolvedValue({
        id: 'risk1',
        userId,
        maxPositionSize: 0.1, // Max is 10% of account
        maxDrawdown: 10,
        maxDailyLoss: 5,
        maxOpenPositions: 5,
        maxPositionsPerSymbol: 2,
        riskLevel: RiskLevel.MEDIUM
      });
      
      // When
      const result = await riskAssessmentService.assessTradeRisk(userId, tradeDetails);
      
      // Then
      expect(result.riskLevel).toBe(RiskLevel.HIGH);
      expect(result.alerts.length).toBeGreaterThan(0);
      expect(result.alerts[0].type).toBe(RiskAlertType.POSITION_SIZE_EXCEEDED);
      expect(result.factors.positionSizeRisk).toBe(RiskLevel.HIGH);
      expect(result.isWithinRiskTolerance).toBe(false);
    });
    
    it('should assess portfolio concentration risk as HIGH when adding to existing positions', async () => {
      // Given
      const userId = 'user123';
      const tradeDetails = {
        symbol: 'BTC/USD',
        direction: 'long',
        entryPrice: 40000,
        positionSize: 0.05, // 5% of account
        stopLossPrice: 39000,
        takeProfitPrice: 43000,
        accountBalance: 100000
      };
      
      // Mock existing positions for this symbol
      prisma.position.findMany.mockResolvedValue([
        {
          id: 'pos1',
          userId,
          symbol: 'BTC/USD',
          direction: 'long',
          entryPrice: 38000,
          currentPrice: 40000,
          positionSize: 0.1, // 10% of account
          status: 'OPEN'
        }
      ]);
      
      // Mock risk settings
      prisma.riskSettings.findUnique.mockResolvedValue({
        id: 'risk1',
        userId,
        maxPositionSize: 0.1,
        maxDrawdown: 10,
        maxDailyLoss: 5,
        maxOpenPositions: 5,
        maxPositionsPerSymbol: 2,
        riskLevel: RiskLevel.MEDIUM
      });
      
      // When
      const result = await riskAssessmentService.assessTradeRisk(userId, tradeDetails);
      
      // Then
      expect(result.riskLevel).toBe(RiskLevel.HIGH);
      expect(result.alerts.some(a => a.type === RiskAlertType.SYMBOL_CONCENTRATION_LIMIT)).toBe(true);
      expect(result.factors.portfolioConcentrationRisk).toBe(RiskLevel.HIGH);
      expect(result.isWithinRiskTolerance).toBe(false);
    });
  });
  
  describe('assessPortfolioRisk', () => {
    it('should assess portfolio risk as LOW for a balanced portfolio', async () => {
      // Given
      const userId = 'user123';
      
      // Mock positions to return a balanced portfolio
      prisma.position.findMany.mockResolvedValue([
        {
          id: 'pos1',
          userId,
          symbol: 'BTC/USD',
          direction: 'long',
          entryPrice: 40000,
          currentPrice: 41000,
          positionSize: 0.02,
          status: 'OPEN'
        },
        {
          id: 'pos2',
          userId,
          symbol: 'ETH/USD',
          direction: 'long',
          entryPrice: 2000,
          currentPrice: 2100,
          positionSize: 0.3,
          status: 'OPEN'
        },
        {
          id: 'pos3',
          userId,
          symbol: 'SOL/USD',
          direction: 'short',
          entryPrice: 100,
          currentPrice: 95,
          positionSize: 5,
          status: 'OPEN'
        }
      ]);
      
      // Mock risk settings
      prisma.riskSettings.findUnique.mockResolvedValue({
        id: 'risk1',
        userId,
        maxPositionSize: 0.1,
        maxDrawdown: 10,
        maxDailyLoss: 5,
        maxOpenPositions: 5,
        maxPositionsPerSymbol: 2,
        riskLevel: RiskLevel.MEDIUM
      });
      
      // Mock aggregate queries
      prisma.position.aggregate.mockResolvedValue({
        _sum: {
          unrealizedPnl: 1500,
          initialMargin: 30000
        }
      });
      
      // When
      const result = await riskAssessmentService.assessPortfolioRisk(userId, 100000);
      
      // Then
      expect(result.overallRiskLevel).toBe(RiskLevel.LOW);
      expect(result.alerts).toHaveLength(0);
      expect(result.metrics.totalPositions).toBe(3);
      expect(result.metrics.totalExposure).toBeDefined();
      expect(result.metrics.directionExposure).toHaveProperty('long');
      expect(result.metrics.directionExposure).toHaveProperty('short');
    });
    
    it('should assess portfolio risk as HIGH when drawdown approaches limit', async () => {
      // Given
      const userId = 'user123';
      const accountBalance = 100000;
      
      // Mock positions
      prisma.position.findMany.mockResolvedValue([
        {
          id: 'pos1',
          userId,
          symbol: 'BTC/USD',
          direction: 'long',
          entryPrice: 40000,
          currentPrice: 36000,
          positionSize: 0.1,
          unrealizedPnl: -4000,
          status: 'OPEN'
        },
        {
          id: 'pos2',
          userId,
          symbol: 'ETH/USD',
          direction: 'long',
          entryPrice: 2000,
          currentPrice: 1800,
          positionSize: 2,
          unrealizedPnl: -400,
          status: 'OPEN'
        }
      ]);
      
      // Mock risk settings with 5% max drawdown
      prisma.riskSettings.findUnique.mockResolvedValue({
        id: 'risk1',
        userId,
        maxPositionSize: 0.1,
        maxDrawdown: 5, // 5% max drawdown
        maxDailyLoss: 5,
        maxOpenPositions: 5,
        maxPositionsPerSymbol: 2,
        riskLevel: RiskLevel.MEDIUM
      });
      
      // Mock aggregate queries with large unrealized loss
      prisma.position.aggregate.mockResolvedValue({
        _sum: {
          unrealizedPnl: -4400, // -4.4% drawdown
          initialMargin: 50000
        }
      });
      
      // When
      const result = await riskAssessmentService.assessPortfolioRisk(userId, accountBalance);
      
      // Then
      expect(result.overallRiskLevel).toBe(RiskLevel.HIGH);
      expect(result.alerts.some(a => a.type === RiskAlertType.APPROACHING_MAX_DRAWDOWN)).toBe(true);
      expect(result.metrics.currentDrawdown).toBeCloseTo(4.4);
      expect(result.isWithinRiskTolerance).toBe(true); // Still within but approaching limit
    });
  });
  
  describe('createRiskAlert', () => {
    it('should create a risk alert in the database', async () => {
      // Given
      const userId = 'user123';
      const alert = {
        type: RiskAlertType.POSITION_SIZE_EXCEEDED,
        message: 'Position size exceeds maximum allowed',
        severity: RiskLevel.HIGH,
        metadata: {
          symbol: 'BTC/USD',
          currentSize: 0.2,
          maxSize: 0.1
        }
      };
      
      // Mock create call
      prisma.riskAlert.create.mockResolvedValue({
        id: 'alert1',
        userId,
        ...alert,
        createdAt: new Date(),
        resolvedAt: null
      });
      
      // When
      const result = await riskAssessmentService.createRiskAlert(userId, alert);
      
      // Then
      expect(result).toHaveProperty('id', 'alert1');
      expect(result).toHaveProperty('userId', userId);
      expect(result).toHaveProperty('type', RiskAlertType.POSITION_SIZE_EXCEEDED);
      expect(prisma.riskAlert.create).toHaveBeenCalledWith({
        data: {
          userId,
          ...alert
        }
      });
    });
  });
}); 