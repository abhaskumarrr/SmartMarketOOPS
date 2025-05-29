/**
 * Performance Routes
 * Routes for performance testing
 */

import express from 'express';
import { authenticateJWT } from '../../middleware/authMiddleware';
import {
  createTest,
  getTest,
  getAllTests,
  startTest,
  getTestResult,
  getTestResults,
  runLoadTest
} from '../../controllers/performance/performanceController';

const router = express.Router();

// All routes require authentication
router.use(authenticateJWT);

// Performance test routes
router.post('/tests', createTest);
router.get('/tests', getAllTests);
router.get('/tests/:id', getTest);
router.post('/tests/:id/start', startTest);
router.get('/tests/:id/results', getTestResults);

// Test result routes
router.get('/results/:id', getTestResult);

// Load test routes
router.post('/load-test', runLoadTest);

export default router; 