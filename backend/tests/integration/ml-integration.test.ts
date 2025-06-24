/**
 * ML Model Integration Tests
 * Tests the integration between ML models and the trading system
 */

import { EnhancedMLModel } from '../../../ml/src/enhanced_ml_model';
import { FibonacciMLModel } from '../../../ml/src/fibonacci_ml_model';
import { generateTestMarketData, generateTestPerformance } from '../helpers/testHelpers';
import { logger } from '../../src/utils/logger';
import * as fs from 'fs';
import * as path from 'path';

describe('ML Model Integration Tests', () => {
  let enhancedModel: EnhancedMLModel;
  let fibonacciModel: FibonacciMLModel;
  
  beforeAll(async () => {
    // Initialize ML models
    enhancedModel = new EnhancedMLModel();
    fibonacciModel = new FibonacciMLModel();
    
    logger.info('ML model integration test setup completed');
  });

  describe('Enhanced ML Model Integration', () => {
    it('should initialize model correctly', () => {
      expect(enhancedModel).toBeDefined();
      expect(enhancedModel.get_model_info()).toHaveProperty('name');
      expect(enhancedModel.get_model_info().name).toBe('EnhancedMLModel');
      expect(enhancedModel.get_model_info()).toHaveProperty('type');
      expect(enhancedModel.get_model_info().type).toBe('enhanced_ml');
    });

    it('should make predictions with proper format', () => {
      const marketData = generateTestMarketData('BTCUSD', 50);
      const prediction = enhancedModel.predict(marketData);
      
      // Verify prediction structure
      expect(prediction).toHaveProperty('action');
      expect(['buy', 'sell', 'hold']).toContain(prediction.action);
      
      expect(prediction).toHaveProperty('confidence');
      expect(prediction.confidence).toBeGreaterThanOrEqual(0);
      expect(prediction.confidence).toBeLessThanOrEqual(1);
      
      expect(prediction).toHaveProperty('position_size');
      expect(prediction.position_size).toBeGreaterThanOrEqual(0);
      expect(prediction.position_size).toBeLessThanOrEqual(1);
      
      expect(prediction).toHaveProperty('timestamp');
      expect(new Date(prediction.timestamp)).toBeInstanceOf(Date);
      
      logger.info('Enhanced ML prediction test passed:', prediction);
    });

    it('should handle different market conditions', () => {
      const testScenarios = [
        { symbol: 'BTCUSD', periods: 20 },
        { symbol: 'ETHUSD', periods: 50 },
        { symbol: 'ADAUSD', periods: 100 },
        { symbol: 'SOLUSD', periods: 30 }
      ];
      
      for (const scenario of testScenarios) {
        const marketData = generateTestMarketData(scenario.symbol, scenario.periods);
        const prediction = enhancedModel.predict(marketData);
        
        expect(prediction).toHaveProperty('action');
        expect(prediction).toHaveProperty('confidence');
        expect(prediction.confidence).toBeGreaterThanOrEqual(0);
        expect(prediction.confidence).toBeLessThanOrEqual(1);
        
        logger.debug(`Enhanced ML prediction for ${scenario.symbol}:`, {
          action: prediction.action,
          confidence: prediction.confidence,
          periods: scenario.periods
        });
      }
    });

    it('should provide consistent predictions for same input', () => {
      const marketData = generateTestMarketData('BTCUSD', 50);
      
      // Make multiple predictions with same data
      const prediction1 = enhancedModel.predict(marketData);
      const prediction2 = enhancedModel.predict(marketData);
      const prediction3 = enhancedModel.predict(marketData);
      
      // Predictions should be consistent (allowing for small numerical differences)
      expect(prediction1.action).toBe(prediction2.action);
      expect(prediction2.action).toBe(prediction3.action);
      
      expect(Math.abs(prediction1.confidence - prediction2.confidence)).toBeLessThan(0.01);
      expect(Math.abs(prediction2.confidence - prediction3.confidence)).toBeLessThan(0.01);
    });

    it('should handle edge cases gracefully', () => {
      // Test with minimal data
      const minimalData = generateTestMarketData('BTCUSD', 1);
      const prediction1 = enhancedModel.predict(minimalData);
      expect(prediction1).toHaveProperty('action');
      
      // Test with empty-like data
      const emptyData = [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]];
      const prediction2 = enhancedModel.predict(emptyData);
      expect(prediction2).toHaveProperty('action');
      
      // Test with extreme values
      const extremeData = [[[100000, 100000, 100000, 100000, 100000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]];
      const prediction3 = enhancedModel.predict(extremeData);
      expect(prediction3).toHaveProperty('action');
    });

    it('should evaluate model performance', () => {
      const marketData = generateTestMarketData('BTCUSD', 100);
      const labels = Array(100).fill(0).map(() => [
        Math.random() * 0.1 - 0.05, // Price change
        Math.random() > 0.5 ? 1 : -1  // Direction
      ]);
      
      const evaluation = enhancedModel.evaluate(marketData, labels);
      
      expect(evaluation).toHaveProperty('mse');
      expect(evaluation).toHaveProperty('mae');
      expect(evaluation).toHaveProperty('r2');
      expect(evaluation).toHaveProperty('direction_accuracy');
      expect(evaluation).toHaveProperty('confidence_error');
      
      expect(evaluation.direction_accuracy).toBeGreaterThanOrEqual(0);
      expect(evaluation.direction_accuracy).toBeLessThanOrEqual(1);
      
      logger.info('Enhanced ML evaluation metrics:', evaluation);
    });
  });

  describe('Fibonacci ML Model Integration', () => {
    it('should initialize model correctly', () => {
      expect(fibonacciModel).toBeDefined();
      expect(fibonacciModel.get_model_info()).toHaveProperty('name');
      expect(fibonacciModel.get_model_info().name).toBe('FibonacciMLModel');
      expect(fibonacciModel.get_model_info()).toHaveProperty('type');
      expect(fibonacciModel.get_model_info().type).toBe('fibonacci_ml');
    });

    it('should make Fibonacci-specific predictions', () => {
      const marketData = generateTestMarketData('BTCUSD', 50);
      const prediction = fibonacciModel.predict(marketData);
      
      // Verify Fibonacci-specific prediction structure
      expect(prediction).toHaveProperty('action');
      expect(['buy', 'sell', 'hold']).toContain(prediction.action);
      
      expect(prediction).toHaveProperty('fibonacci_level');
      expect(prediction.fibonacci_level).toBeGreaterThanOrEqual(0);
      expect(prediction.fibonacci_level).toBeLessThanOrEqual(1);
      
      expect(prediction).toHaveProperty('trend_direction');
      expect(['uptrend', 'downtrend', 'sideways']).toContain(prediction.trend_direction);
      
      expect(prediction).toHaveProperty('fibonacci_levels');
      expect(prediction.fibonacci_levels).toHaveProperty('fib_0.236');
      expect(prediction.fibonacci_levels).toHaveProperty('fib_0.382');
      expect(prediction.fibonacci_levels).toHaveProperty('fib_0.5');
      expect(prediction.fibonacci_levels).toHaveProperty('fib_0.618');
      expect(prediction.fibonacci_levels).toHaveProperty('fib_0.786');
      
      expect(prediction).toHaveProperty('support_strength');
      expect(prediction).toHaveProperty('resistance_strength');
      
      logger.info('Fibonacci ML prediction test passed:', prediction);
    });

    it('should calculate Fibonacci levels correctly', () => {
      const marketData = generateTestMarketData('BTCUSD', 50);
      const prediction = fibonacciModel.predict(marketData);
      
      const fibLevels = prediction.fibonacci_levels;
      
      // Fibonacci levels should be in correct order
      expect(fibLevels['fib_0.236']).toBeGreaterThan(fibLevels['fib_0.382']);
      expect(fibLevels['fib_0.382']).toBeGreaterThan(fibLevels['fib_0.5']);
      expect(fibLevels['fib_0.5']).toBeGreaterThan(fibLevels['fib_0.618']);
      expect(fibLevels['fib_0.618']).toBeGreaterThan(fibLevels['fib_0.786']);
      
      // All levels should be positive numbers
      Object.values(fibLevels).forEach(level => {
        expect(typeof level).toBe('number');
        expect(level).toBeGreaterThan(0);
      });
    });

    it('should provide trend analysis', () => {
      const testScenarios = [
        { symbol: 'BTCUSD', periods: 30 },
        { symbol: 'ETHUSD', periods: 60 },
        { symbol: 'ADAUSD', periods: 90 }
      ];
      
      for (const scenario of testScenarios) {
        const marketData = generateTestMarketData(scenario.symbol, scenario.periods);
        const prediction = fibonacciModel.predict(marketData);
        
        expect(prediction).toHaveProperty('trend_direction');
        expect(prediction).toHaveProperty('trend_probabilities');
        
        const trendProbs = prediction.trend_probabilities;
        expect(trendProbs).toHaveProperty('uptrend');
        expect(trendProbs).toHaveProperty('downtrend');
        expect(trendProbs).toHaveProperty('sideways');
        
        // Probabilities should sum to approximately 1
        const totalProb = trendProbs.uptrend + trendProbs.downtrend + trendProbs.sideways;
        expect(Math.abs(totalProb - 1.0)).toBeLessThan(0.1);
        
        logger.debug(`Fibonacci trend analysis for ${scenario.symbol}:`, {
          trend: prediction.trend_direction,
          probabilities: trendProbs
        });
      }
    });

    it('should evaluate Fibonacci model performance', () => {
      const marketData = generateTestMarketData('BTCUSD', 100);
      const labels = Array(100).fill(0).map(() => [
        Math.random() * 0.1 - 0.05, // Price change
        Math.random() > 0.5 ? 1 : -1, // Direction
        Math.floor(Math.random() * 3), // Trend (0=down, 1=side, 2=up)
        Math.random(), // Support strength
        Math.random(), // Resistance strength
        ...Array(5).fill(0).map(() => Math.random()) // Fibonacci level probabilities
      ]);
      
      const evaluation = fibonacciModel.evaluate(marketData, labels);
      
      expect(evaluation).toHaveProperty('mse');
      expect(evaluation).toHaveProperty('mae');
      expect(evaluation).toHaveProperty('r2');
      expect(evaluation).toHaveProperty('fibonacci_accuracy');
      expect(evaluation).toHaveProperty('trend_accuracy');
      
      logger.info('Fibonacci ML evaluation metrics:', evaluation);
    });
  });

  describe('Model Comparison and Ensemble', () => {
    it('should compare predictions from both models', () => {
      const marketData = generateTestMarketData('BTCUSD', 50);
      
      const enhancedPrediction = enhancedModel.predict(marketData);
      const fibonacciPrediction = fibonacciModel.predict(marketData);
      
      // Both should provide valid predictions
      expect(enhancedPrediction).toHaveProperty('action');
      expect(fibonacciPrediction).toHaveProperty('action');
      
      // Compare confidence levels
      const enhancedConfidence = enhancedPrediction.confidence;
      const fibonacciConfidence = fibonacciPrediction.confidence;
      
      expect(enhancedConfidence).toBeGreaterThanOrEqual(0);
      expect(fibonacciConfidence).toBeGreaterThanOrEqual(0);
      
      logger.info('Model comparison:', {
        enhanced: {
          action: enhancedPrediction.action,
          confidence: enhancedConfidence
        },
        fibonacci: {
          action: fibonacciPrediction.action,
          confidence: fibonacciConfidence,
          trend: fibonacciPrediction.trend_direction
        }
      });
    });

    it('should create ensemble predictions', () => {
      const marketData = generateTestMarketData('BTCUSD', 50);
      
      const enhancedPrediction = enhancedModel.predict(marketData);
      const fibonacciPrediction = fibonacciModel.predict(marketData);
      
      // Create simple ensemble prediction
      const ensemblePrediction = createEnsemblePrediction(enhancedPrediction, fibonacciPrediction);
      
      expect(ensemblePrediction).toHaveProperty('action');
      expect(ensemblePrediction).toHaveProperty('confidence');
      expect(ensemblePrediction).toHaveProperty('models_agreement');
      expect(ensemblePrediction).toHaveProperty('weighted_confidence');
      
      expect(ensemblePrediction.confidence).toBeGreaterThanOrEqual(0);
      expect(ensemblePrediction.confidence).toBeLessThanOrEqual(1);
      
      logger.info('Ensemble prediction:', ensemblePrediction);
    });
  });

  describe('Model Persistence', () => {
    it('should save and load Enhanced ML model', async () => {
      const modelPath = path.join(__dirname, '../temp/enhanced_model_test.pt');
      const modelDir = path.dirname(modelPath);
      
      // Ensure directory exists
      if (!fs.existsSync(modelDir)) {
        fs.mkdirSync(modelDir, { recursive: true });
      }
      
      try {
        // Save model
        enhancedModel.save(modelPath);
        expect(fs.existsSync(modelPath)).toBe(true);
        
        // Load model
        const loadedModel = EnhancedMLModel.load(modelPath);
        expect(loadedModel).toBeDefined();
        
        const originalInfo = enhancedModel.get_model_info();
        const loadedInfo = loadedModel.get_model_info();
        
        expect(loadedInfo.name).toBe(originalInfo.name);
        expect(loadedInfo.type).toBe(originalInfo.type);
        
        // Test that loaded model can make predictions
        const marketData = generateTestMarketData('BTCUSD', 30);
        const prediction = loadedModel.predict(marketData);
        expect(prediction).toHaveProperty('action');
        
        logger.info('Enhanced ML model save/load test passed');
      } finally {
        // Cleanup
        if (fs.existsSync(modelPath)) {
          fs.unlinkSync(modelPath);
        }
        const scalerPath = modelPath.replace('.pt', '_scaler.pkl');
        if (fs.existsSync(scalerPath)) {
          fs.unlinkSync(scalerPath);
        }
      }
    });

    it('should save and load Fibonacci ML model', async () => {
      const modelPath = path.join(__dirname, '../temp/fibonacci_model_test.pt');
      const modelDir = path.dirname(modelPath);
      
      // Ensure directory exists
      if (!fs.existsSync(modelDir)) {
        fs.mkdirSync(modelDir, { recursive: true });
      }
      
      try {
        // Save model
        fibonacciModel.save(modelPath);
        expect(fs.existsSync(modelPath)).toBe(true);
        
        // Load model
        const loadedModel = FibonacciMLModel.load(modelPath);
        expect(loadedModel).toBeDefined();
        
        const originalInfo = fibonacciModel.get_model_info();
        const loadedInfo = loadedModel.get_model_info();
        
        expect(loadedInfo.name).toBe(originalInfo.name);
        expect(loadedInfo.type).toBe(originalInfo.type);
        expect(loadedInfo.fibonacci_levels).toEqual(originalInfo.fibonacci_levels);
        
        // Test that loaded model can make predictions
        const marketData = generateTestMarketData('BTCUSD', 30);
        const prediction = loadedModel.predict(marketData);
        expect(prediction).toHaveProperty('action');
        expect(prediction).toHaveProperty('fibonacci_level');
        
        logger.info('Fibonacci ML model save/load test passed');
      } finally {
        // Cleanup
        if (fs.existsSync(modelPath)) {
          fs.unlinkSync(modelPath);
        }
        const scalerPath = modelPath.replace('.pt', '_scaler.pkl');
        if (fs.existsSync(scalerPath)) {
          fs.unlinkSync(scalerPath);
        }
      }
    });
  });
});

/**
 * Create ensemble prediction from multiple model predictions
 */
function createEnsemblePrediction(enhancedPred: any, fibonacciPred: any): any {
  // Check if models agree on action
  const modelsAgree = enhancedPred.action === fibonacciPred.action;
  
  // Weight the predictions based on confidence
  const enhancedWeight = enhancedPred.confidence;
  const fibonacciWeight = fibonacciPred.confidence;
  const totalWeight = enhancedWeight + fibonacciWeight;
  
  // Calculate weighted confidence
  const weightedConfidence = totalWeight > 0 
    ? (enhancedWeight * enhancedPred.confidence + fibonacciWeight * fibonacciPred.confidence) / totalWeight
    : 0;
  
  // Determine ensemble action
  let ensembleAction = 'hold';
  if (modelsAgree) {
    ensembleAction = enhancedPred.action;
  } else {
    // If models disagree, choose based on higher confidence
    ensembleAction = enhancedPred.confidence > fibonacciPred.confidence 
      ? enhancedPred.action 
      : fibonacciPred.action;
  }
  
  // Adjust confidence based on agreement
  const agreementBonus = modelsAgree ? 0.1 : -0.1;
  const finalConfidence = Math.max(0, Math.min(1, weightedConfidence + agreementBonus));
  
  return {
    action: ensembleAction,
    confidence: finalConfidence,
    models_agreement: modelsAgree,
    weighted_confidence: weightedConfidence,
    enhanced_prediction: enhancedPred,
    fibonacci_prediction: fibonacciPred,
    timestamp: new Date().toISOString()
  };
}