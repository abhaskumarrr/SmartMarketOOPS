/**
 * Performance Controller
 * Handles HTTP requests for performance testing
 */

import { Request, Response } from 'express';
import { AuthenticatedRequest } from '../../middleware/authMiddleware';
import { createLogger } from '../../utils/logger';
import perfTestService from '../../services/performance/perfTestService';
import loadTestService from '../../services/performance/loadTestService';
import prisma from '../../utils/prismaClient';
import {
  PerformanceTestConfig,
  PerformanceTestType,
  SystemLoadTest
} from '../../types/performance';

// Create logger
const logger = createLogger('PerformanceController');

/**
 * Create a new performance test
 * @route POST /api/performance/tests
 */
export const createTest = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const config: PerformanceTestConfig = req.body;
    
    // Validate request
    if (!config.name || !config.testType || !config.duration || !config.concurrency) {
      res.status(400).json({
        status: 'error',
        message: 'Missing required test configuration parameters'
      });
      return;
    }
    
    // Create test
    const test = await perfTestService.createTest(config);
    
    res.status(201).json({
      status: 'success',
      data: test
    });
  } catch (error) {
    logger.error('Error creating performance test', {
      error: error instanceof Error ? error.message : String(error),
      userId: req.user?.id
    });
    
    res.status(500).json({
      status: 'error',
      message: 'Failed to create performance test',
      error: error instanceof Error ? error.message : String(error)
    });
  }
};

/**
 * Get a performance test by ID
 * @route GET /api/performance/tests/:id
 */
export const getTest = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const { id } = req.params;
    
    // Get test
    const test = await perfTestService.getTest(id);
    
    res.status(200).json({
      status: 'success',
      data: test
    });
  } catch (error) {
    logger.error('Error getting performance test', {
      error: error instanceof Error ? error.message : String(error),
      userId: req.user?.id,
      testId: req.params.id
    });
    
    res.status(error.message?.includes('not found') ? 404 : 500).json({
      status: 'error',
      message: 'Failed to get performance test',
      error: error instanceof Error ? error.message : String(error)
    });
  }
};

/**
 * Get all performance tests
 * @route GET /api/performance/tests
 */
export const getAllTests = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const { type } = req.query;
    
    // Get tests
    const tests = await perfTestService.getAllTests(
      type ? type as PerformanceTestType : undefined
    );
    
    res.status(200).json({
      status: 'success',
      data: tests
    });
  } catch (error) {
    logger.error('Error getting all performance tests', {
      error: error instanceof Error ? error.message : String(error),
      userId: req.user?.id
    });
    
    res.status(500).json({
      status: 'error',
      message: 'Failed to get performance tests',
      error: error instanceof Error ? error.message : String(error)
    });
  }
};

/**
 * Start a performance test
 * @route POST /api/performance/tests/:id/start
 */
export const startTest = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const { id } = req.params;
    
    // Start test
    const result = await perfTestService.startTest(id);
    
    res.status(200).json({
      status: 'success',
      data: result
    });
  } catch (error) {
    logger.error('Error starting performance test', {
      error: error instanceof Error ? error.message : String(error),
      userId: req.user?.id,
      testId: req.params.id
    });
    
    res.status(error.message?.includes('not found') ? 404 : 500).json({
      status: 'error',
      message: 'Failed to start performance test',
      error: error instanceof Error ? error.message : String(error)
    });
  }
};

/**
 * Get a test result by ID
 * @route GET /api/performance/results/:id
 */
export const getTestResult = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const { id } = req.params;
    
    // Get result
    const result = await perfTestService.getTestResult(id);
    
    res.status(200).json({
      status: 'success',
      data: result
    });
  } catch (error) {
    logger.error('Error getting test result', {
      error: error instanceof Error ? error.message : String(error),
      userId: req.user?.id,
      resultId: req.params.id
    });
    
    res.status(error.message?.includes('not found') ? 404 : 500).json({
      status: 'error',
      message: 'Failed to get test result',
      error: error instanceof Error ? error.message : String(error)
    });
  }
};

/**
 * Get all test results for a test
 * @route GET /api/performance/tests/:id/results
 */
export const getTestResults = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const { id } = req.params;
    
    // Get results
    const results = await perfTestService.getTestResults(id);
    
    res.status(200).json({
      status: 'success',
      data: results
    });
  } catch (error) {
    logger.error('Error getting test results', {
      error: error instanceof Error ? error.message : String(error),
      userId: req.user?.id,
      testId: req.params.id
    });
    
    res.status(500).json({
      status: 'error',
      message: 'Failed to get test results',
      error: error instanceof Error ? error.message : String(error)
    });
  }
};

/**
 * Run a load test
 * @route POST /api/performance/load-test
 */
export const runLoadTest = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const config: SystemLoadTest = req.body;
    
    // Validate request
    if (!config.name || !config.stages || config.stages.length === 0 || !config.targetEndpoints || config.targetEndpoints.length === 0) {
      res.status(400).json({
        status: 'error',
        message: 'Missing required load test configuration parameters'
      });
      return;
    }
    
    // Calculate total duration
    const duration = config.stages.reduce((total, stage) => total + stage.duration, 0);
    
    // Save test configuration
    const dbTest = await prisma.performanceTest.create({
      data: {
        name: config.name,
        description: config.description,
        testType: PerformanceTestType.LOAD_TEST,
        duration,
        concurrency: Math.max(...config.stages.map(stage => stage.target)),
        options: {
          stages: config.stages,
          targetEndpoints: config.targetEndpoints
        } as any
      }
    });
    
    // Run load test
    const resultId = await loadTestService.runLoadTest(
      {
        ...config,
        id: dbTest.id,
        duration
      },
      async (result) => {
        // Save or update result in database
        try {
          const existingResult = await prisma.performanceTestResult.findUnique({
            where: { id: result.id }
          });
          
          if (existingResult) {
            // Update existing result
            await prisma.performanceTestResult.update({
              where: { id: result.id },
              data: {
                status: result.status,
                endTime: result.endTime ? new Date(result.endTime) : undefined,
                duration: result.duration,
                metrics: result.metrics as any,
                errors: result.errors as any
              }
            });
          } else {
            // Create new result
            await prisma.performanceTestResult.create({
              data: {
                id: result.id,
                testId: result.testId,
                status: result.status,
                startTime: new Date(result.startTime),
                endTime: result.endTime ? new Date(result.endTime) : undefined,
                duration: result.duration,
                metrics: result.metrics as any,
                errors: result.errors as any
              }
            });
          }
        } catch (error) {
          logger.error('Error saving load test result', {
            error: error instanceof Error ? error.message : String(error),
            userId: req.user?.id,
            resultId: result.id
          });
        }
      }
    );
    
    res.status(202).json({
      status: 'success',
      message: 'Load test started',
      data: {
        testId: dbTest.id,
        resultId
      }
    });
  } catch (error) {
    logger.error('Error running load test', {
      error: error instanceof Error ? error.message : String(error),
      userId: req.user?.id
    });
    
    res.status(500).json({
      status: 'error',
      message: 'Failed to run load test',
      error: error instanceof Error ? error.message : String(error)
    });
  }
}; 