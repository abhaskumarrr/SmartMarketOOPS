/**
 * Performance Controller
 * Handles HTTP requests for performance testing
 */
import { Response } from 'express';
import { AuthenticatedRequest } from '../../middleware/authMiddleware';
/**
 * Create a new performance test
 * @route POST /api/performance/tests
 */
export declare const createTest: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Get a performance test by ID
 * @route GET /api/performance/tests/:id
 */
export declare const getTest: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Get all performance tests
 * @route GET /api/performance/tests
 */
export declare const getAllTests: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Start a performance test
 * @route POST /api/performance/tests/:id/start
 */
export declare const startTest: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Get a test result by ID
 * @route GET /api/performance/results/:id
 */
export declare const getTestResult: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Get all test results for a test
 * @route GET /api/performance/tests/:id/results
 */
export declare const getTestResults: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Run a load test
 * @route POST /api/performance/load-test
 */
export declare const runLoadTest: (req: AuthenticatedRequest, res: Response) => Promise<void>;
