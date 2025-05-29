import { Request, Response } from 'express';
import * as riskController from '../../../src/controllers/riskController';
import { AuthenticatedRequest } from '../../../src/middleware/authMiddleware';

// Mock the risk services
jest.mock('../../../src/services/trading/riskManagementService', () => {
  return {
    default: {
      getRiskSettings: jest.fn(),
      updateRiskSettings: jest.fn(),
      calculatePositionSize: jest.fn(),
      calculateStopLoss: jest.fn(),
      calculateTakeProfit: jest.fn()
    }
  };
});

jest.mock('../../../src/services/trading/riskAssessmentService', () => {
  return {
    default: {
      assessTradeRisk: jest.fn(),
      assessPortfolioRisk: jest.fn(),
      createRiskAlert: jest.fn(),
      getRiskAlerts: jest.fn()
    }
  };
});

jest.mock('../../../src/services/trading/circuitBreakerService', () => {
  return {
    default: {
      checkAndActivateCircuitBreakers: jest.fn(),
      resetCircuitBreaker: jest.fn(),
      getActiveCircuitBreakers: jest.fn()
    }
  };
});

describe('RiskController', () => {
  let req: Partial<AuthenticatedRequest>;
  let res: Partial<Response>;
  
  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();
    
    // Mock request object
    req = {
      user: { id: 'user123' },
      params: {},
      body: {},
      query: {}
    };
    
    // Mock response object
    res = {
      status: jest.fn().mockReturnThis(),
      json: jest.fn()
    };
  });
  
  describe('getRiskSettings', () => {
    it('should return risk settings for a user', async () => {
      // Mock risk settings
      const mockSettings = {
        id: 'risk1',
        userId: 'user123',
        maxPositionSize: 0.1,
        maxDrawdown: 10
      };
      
      // Setup service mock
      const riskManagementService = require('../../../src/services/trading/riskManagementService').default;
      riskManagementService.getRiskSettings.mockResolvedValue(mockSettings);
      
      // Call the controller method
      await riskController.getRiskSettings(req as AuthenticatedRequest, res as Response);
      
      // Assertions
      expect(riskManagementService.getRiskSettings).toHaveBeenCalledWith('user123', undefined);
      expect(res.status).toHaveBeenCalledWith(200);
      expect(res.json).toHaveBeenCalledWith(mockSettings);
    });
    
    it('should handle errors appropriately', async () => {
      // Setup service mock to throw an error
      const error = new Error('Database error');
      const riskManagementService = require('../../../src/services/trading/riskManagementService').default;
      riskManagementService.getRiskSettings.mockRejectedValue(error);
      
      // Call the controller method
      await riskController.getRiskSettings(req as AuthenticatedRequest, res as Response);
      
      // Assertions
      expect(res.status).toHaveBeenCalledWith(500);
      expect(res.json).toHaveBeenCalledWith({
        error: 'Failed to get risk settings',
        message: error.message
      });
    });
  });
  
  describe('saveRiskSettings', () => {
    it('should update risk settings for a user', async () => {
      // Setup request body
      req.body = {
        maxPositionSize: 0.05,
        maxDrawdown: 5
      };
      
      // Mock updated settings
      const mockUpdatedSettings = {
        id: 'risk1',
        userId: 'user123',
        maxPositionSize: 0.05,
        maxDrawdown: 5
      };
      
      // Setup service mock
      const riskManagementService = require('../../../src/services/trading/riskManagementService').default;
      riskManagementService.saveRiskSettings.mockResolvedValue(mockUpdatedSettings);
      
      // Call the controller method
      await riskController.saveRiskSettings(req as AuthenticatedRequest, res as Response);
      
      // Assertions
      expect(riskManagementService.saveRiskSettings).toHaveBeenCalledWith({
        userId: 'user123',
        maxPositionSize: 0.05,
        maxDrawdown: 5
      });
      expect(res.status).toHaveBeenCalledWith(200);
      expect(res.json).toHaveBeenCalledWith(mockUpdatedSettings);
    });
  });
  
  describe('calculatePositionSize', () => {
    it('should calculate position size based on parameters', async () => {
      // Setup request body
      req.body = {
        symbol: 'BTC/USD',
        direction: 'long',
        entryPrice: 40000,
        stopLossPrice: 39000
      };
      
      // Mock calculation result
      const mockResult = {
        positionSize: 0.0025,
        riskAmount: 100,
        riskPercentage: 1
      };
      
      // Setup service mock
      const riskManagementService = require('../../../src/services/trading/riskManagementService').default;
      riskManagementService.calculatePositionSize.mockResolvedValue(mockResult);
      
      // Call the controller method
      await riskController.calculatePositionSize(req as AuthenticatedRequest, res as Response);
      
      // Assertions
      expect(riskManagementService.calculatePositionSize).toHaveBeenCalledWith({
        userId: 'user123',
        botId: undefined,
        symbol: 'BTC/USD',
        direction: 'long',
        entryPrice: 40000,
        stopLossPrice: 39000,
        stopLossPercentage: undefined,
        confidence: undefined
      });
      expect(res.status).toHaveBeenCalledWith(200);
      expect(res.json).toHaveBeenCalledWith(mockResult);
    });
  });
  
  describe('analyzeTradeRisk', () => {
    it('should assess risk for a potential trade', async () => {
      // Setup request body
      req.body = {
        symbol: 'BTC/USD',
        direction: 'long',
        entryPrice: 40000,
        positionSize: 0.1,
        stopLossPrice: 39000
      };
      
      // Mock assessment result
      const mockResult = {
        riskLevel: 'MEDIUM',
        isWithinRiskTolerance: true,
        alerts: [],
        factors: {
          positionSizeRisk: 'LOW',
          stopLossRisk: 'MEDIUM',
          portfolioConcentrationRisk: 'LOW'
        }
      };
      
      // Setup service mock
      const riskAssessmentService = require('../../../src/services/trading/riskAssessmentService').default;
      riskAssessmentService.analyzeTradeRisk.mockResolvedValue(mockResult);
      
      // Call the controller method
      await riskController.analyzeTradeRisk(req as AuthenticatedRequest, res as Response);
      
      // Assertions
      expect(riskAssessmentService.analyzeTradeRisk).toHaveBeenCalledWith(
        'user123',
        req.body
      );
      expect(res.status).toHaveBeenCalledWith(200);
      expect(res.json).toHaveBeenCalledWith(mockResult);
    });
  });
  
  describe('generateRiskReport', () => {
    it('should generate a risk report for a user', async () => {
      // Mock assessment result
      const mockResult = {
        overallRiskLevel: 'LOW',
        isWithinRiskTolerance: true,
        alerts: [],
        metrics: {
          totalPositions: 3,
          totalExposure: 25000,
          currentDrawdown: 2.5,
          directionExposure: {
            long: 20000,
            short: 5000
          }
        }
      };
      
      // Setup service mock
      const riskAssessmentService = require('../../../src/services/trading/riskAssessmentService').default;
      riskAssessmentService.generateRiskReport.mockResolvedValue(mockResult);
      
      // Call the controller method
      await riskController.generateRiskReport(req as AuthenticatedRequest, res as Response);
      
      // Assertions
      expect(riskAssessmentService.generateRiskReport).toHaveBeenCalledWith('user123');
      expect(res.status).toHaveBeenCalledWith(200);
      expect(res.json).toHaveBeenCalledWith(mockResult);
    });
  });
  
  describe('getRiskAlerts', () => {
    it('should get risk alerts for a user', async () => {
      // Setup request query
      req.query = {
        status: 'active'
      };
      
      // Mock alerts
      const mockAlerts = [
        {
          id: 'alert1',
          userId: 'user123',
          type: 'POSITION_SIZE_EXCEEDED',
          message: 'Position size exceeds maximum allowed',
          severity: 'HIGH',
          createdAt: new Date(),
          resolvedAt: null
        }
      ];
      
      // Setup service mock
      const riskAssessmentService = require('../../../src/services/trading/riskAssessmentService').default;
      riskAssessmentService.getRiskAlerts.mockResolvedValue(mockAlerts);
      
      // Call the controller method
      await riskController.getRiskAlerts(req as AuthenticatedRequest, res as Response);
      
      // Assertions
      expect(riskAssessmentService.getRiskAlerts).toHaveBeenCalledWith(
        'user123',
        'active'
      );
      expect(res.status).toHaveBeenCalledWith(200);
      expect(res.json).toHaveBeenCalledWith(mockAlerts);
    });
  });
  
  describe('getCircuitBreakerStatus', () => {
    it('should get active circuit breakers for a user', async () => {
      // Mock circuit breakers
      const mockBreakers = [
        {
          id: 'cb1',
          userId: 'user123',
          type: 'MAX_DRAWDOWN_BREAKER',
          status: 'ACTIVE',
          activationReason: 'Drawdown exceeded maximum threshold',
          activatedAt: new Date(),
          resetAt: null
        }
      ];
      
      // Setup service mock
      const circuitBreakerService = require('../../../src/services/trading/circuitBreakerService').default;
      circuitBreakerService.getActiveCircuitBreakers.mockResolvedValue(mockBreakers);
      
      // Call the controller method
      await riskController.getCircuitBreakerStatus(req as AuthenticatedRequest, res as Response);
      
      // Assertions
      expect(circuitBreakerService.getActiveCircuitBreakers).toHaveBeenCalledWith('user123');
      expect(res.status).toHaveBeenCalledWith(200);
      expect(res.json).toHaveBeenCalledWith(mockBreakers);
    });
  });
  
  describe('resetCircuitBreaker', () => {
    it('should reset a circuit breaker', async () => {
      // Setup request params and body
      req.params = { id: 'cb1' };
      req.body = { reason: 'Manual reset by admin' };
      
      // Mock reset result
      const mockResult = {
        id: 'cb1',
        userId: 'user123',
        status: 'RESET',
        resetAt: new Date(),
        resetReason: 'Manual reset by admin'
      };
      
      // Setup service mock
      const circuitBreakerService = require('../../../src/services/trading/circuitBreakerService').default;
      circuitBreakerService.resetCircuitBreaker.mockResolvedValue(mockResult);
      
      // Call the controller method
      await riskController.resetCircuitBreaker(req as AuthenticatedRequest, res as Response);
      
      // Assertions
      expect(circuitBreakerService.resetCircuitBreaker).toHaveBeenCalledWith(
        'cb1',
        'Manual reset by admin'
      );
      expect(res.status).toHaveBeenCalledWith(200);
      expect(res.json).toHaveBeenCalledWith(mockResult);
    });
  });
}); 