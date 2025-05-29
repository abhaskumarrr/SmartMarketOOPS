import { PrismaClient } from '@prisma/client';
import { RiskManagementService } from '../../../src/services/trading/riskManagementService';
import { 
  RiskLevel, 
  PositionSizingMethod, 
  StopLossType, 
  TakeProfitType,
  RiskSettings
} from '../../../src/types/risk';

// Mock Prisma
jest.mock('@prisma/client', () => {
  const mockPrismaClient = {
    riskSettings: {
      findUnique: jest.fn(),
      create: jest.fn(),
      update: jest.fn(),
      upsert: jest.fn(),
    },
    user: {
      findUnique: jest.fn(),
    },
    tradingBot: {
      findUnique: jest.fn(),
    },
    position: {
      findMany: jest.fn(),
    },
    $transaction: jest.fn((callback) => callback(mockPrismaClient)),
  };
  
  return {
    PrismaClient: jest.fn(() => mockPrismaClient),
  };
});

describe('RiskManagementService', () => {
  let riskManagementService: RiskManagementService;
  let prisma: any;
  
  beforeEach(() => {
    jest.clearAllMocks();
    
    // Initialize service with mocked Prisma
    prisma = new PrismaClient();
    riskManagementService = new RiskManagementService(prisma);
  });
  
  describe('calculatePositionSize', () => {
    it('should calculate position size using FIXED_AMOUNT method', async () => {
      // Given
      const accountBalance = 10000;
      const symbol = 'BTC/USD';
      const riskAmount = 100;
      const entryPrice = 40000;
      const stopLossPrice = 39000;
      const method = PositionSizingMethod.FIXED_AMOUNT;
      
      // When
      const result = await riskManagementService.calculatePositionSize(
        accountBalance,
        symbol,
        riskAmount,
        entryPrice,
        stopLossPrice,
        method
      );
      
      // Then
      expect(result.positionSize).toBe(0.0025); // 100 / (40000 - 39000)
      expect(result.riskAmount).toBe(100);
      expect(result.riskPercentage).toBe(1); // (100 / 10000) * 100
    });
    
    it('should calculate position size using FIXED_PERCENTAGE method', async () => {
      // Given
      const accountBalance = 10000;
      const symbol = 'BTC/USD';
      const riskPercentage = 1; // 1% risk
      const entryPrice = 40000;
      const stopLossPrice = 39000;
      const method = PositionSizingMethod.FIXED_PERCENTAGE;
      
      // When
      const result = await riskManagementService.calculatePositionSize(
        accountBalance,
        symbol,
        riskPercentage,
        entryPrice,
        stopLossPrice,
        method
      );
      
      // Then
      expect(result.positionSize).toBe(0.0025); // (10000 * 0.01) / (40000 - 39000)
      expect(result.riskAmount).toBe(100); // 10000 * 0.01
      expect(result.riskPercentage).toBe(1);
    });
    
    it('should adjust position size based on confidence level', async () => {
      // Given
      const accountBalance = 10000;
      const symbol = 'BTC/USD';
      const riskPercentage = 1; // 1% risk
      const entryPrice = 40000;
      const stopLossPrice = 39000;
      const method = PositionSizingMethod.FIXED_PERCENTAGE;
      const confidenceLevel = 0.5; // 50% confidence
      
      // When
      const result = await riskManagementService.calculatePositionSize(
        accountBalance,
        symbol,
        riskPercentage,
        entryPrice,
        stopLossPrice,
        method,
        confidenceLevel
      );
      
      // Then - position size should be reduced by confidence factor
      expect(result.positionSize).toBe(0.00125); // 0.0025 * 0.5
      expect(result.riskAmount).toBe(50); // 100 * 0.5
      expect(result.riskPercentage).toBe(0.5); // 1 * 0.5
    });
  });
  
  describe('calculateStopLoss', () => {
    it('should calculate fixed price stop loss', async () => {
      // Given
      const entryPrice = 40000;
      const stopLossType = StopLossType.FIXED_PRICE;
      const stopLossValue = 39000;
      
      // When
      const result = riskManagementService.calculateStopLoss(
        entryPrice,
        stopLossType,
        stopLossValue
      );
      
      // Then
      expect(result).toBe(39000);
    });
    
    it('should calculate percentage-based stop loss', async () => {
      // Given
      const entryPrice = 40000;
      const stopLossType = StopLossType.PERCENTAGE;
      const stopLossValue = 2.5; // 2.5% below entry
      
      // When
      const result = riskManagementService.calculateStopLoss(
        entryPrice,
        stopLossType,
        stopLossValue
      );
      
      // Then
      expect(result).toBe(39000); // 40000 * (1 - 0.025)
    });
    
    it('should calculate ATR-based stop loss', async () => {
      // Given
      const entryPrice = 40000;
      const stopLossType = StopLossType.ATR_MULTIPLE;
      const stopLossValue = 2; // 2 ATR units
      const atrValue = 500; // Example ATR value
      
      // When
      const result = riskManagementService.calculateStopLoss(
        entryPrice,
        stopLossType,
        stopLossValue,
        atrValue
      );
      
      // Then
      expect(result).toBe(39000); // 40000 - (2 * 500)
    });
  });
  
  describe('getRiskSettings', () => {
    it('should get user risk settings when they exist', async () => {
      // Given
      const userId = 'user123';
      const mockRiskSettings = {
        id: 'risk1',
        userId,
        botId: null,
        maxPositionSize: 0.1,
        maxDrawdown: 10,
        defaultStopLossType: StopLossType.PERCENTAGE,
        defaultStopLossValue: 2,
        defaultTakeProfitType: TakeProfitType.PERCENTAGE,
        defaultTakeProfitValue: 5,
        maxDailyLoss: 5,
        riskLevel: RiskLevel.MEDIUM,
        positionSizingMethod: PositionSizingMethod.FIXED_PERCENTAGE,
        defaultRiskPerTrade: 1,
        maxOpenPositions: 5,
        maxPositionsPerSymbol: 2,
        enabledCircuitBreakers: true,
        createdAt: new Date(),
        updatedAt: new Date(),
      };
      
      // Mock the findUnique call
      prisma.riskSettings.findUnique.mockResolvedValue(mockRiskSettings);
      
      // When
      const result = await riskManagementService.getRiskSettings(userId);
      
      // Then
      expect(result).toEqual(mockRiskSettings);
      expect(prisma.riskSettings.findUnique).toHaveBeenCalledWith({
        where: { userId }
      });
    });
    
    it('should return default risk settings when user settings do not exist', async () => {
      // Given
      const userId = 'user123';
      
      // Mock the findUnique call to return null
      prisma.riskSettings.findUnique.mockResolvedValue(null);
      
      // When
      const result = await riskManagementService.getRiskSettings(userId);
      
      // Then
      expect(result).toMatchObject({
        riskLevel: RiskLevel.MEDIUM,
        maxPositionSize: 0.1,
        maxDrawdown: 10,
        positionSizingMethod: PositionSizingMethod.FIXED_PERCENTAGE,
        defaultRiskPerTrade: 1,
      });
    });
  });
  
  describe('updateRiskSettings', () => {
    it('should update existing risk settings', async () => {
      // Given
      const userId = 'user123';
      const updateData = {
        riskLevel: RiskLevel.LOW,
        maxPositionSize: 0.05,
        defaultRiskPerTrade: 0.5,
      };
      
      // Mock the update call
      prisma.riskSettings.update.mockResolvedValue({
        id: 'risk1',
        userId,
        ...updateData,
      });
      
      // When
      const result = await riskManagementService.updateRiskSettings(userId, updateData);
      
      // Then
      expect(result).toMatchObject(updateData);
      expect(prisma.riskSettings.update).toHaveBeenCalledWith({
        where: { userId },
        data: updateData,
      });
    });
    
    it('should create risk settings if they do not exist', async () => {
      // Given
      const userId = 'user123';
      const updateData = {
        riskLevel: RiskLevel.LOW,
        maxPositionSize: 0.05,
        defaultRiskPerTrade: 0.5,
      };
      
      // Mock the update call to throw an error
      prisma.riskSettings.update.mockRejectedValue(new Error('Not found'));
      
      // Mock the create call
      prisma.riskSettings.create.mockResolvedValue({
        id: 'risk1',
        userId,
        ...updateData,
      });
      
      // When
      const result = await riskManagementService.updateRiskSettings(userId, updateData);
      
      // Then
      expect(result).toMatchObject(updateData);
      expect(prisma.riskSettings.create).toHaveBeenCalledWith({
        data: {
          userId,
          ...updateData,
        },
      });
    });
  });
}); 