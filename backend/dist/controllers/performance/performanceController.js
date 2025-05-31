"use strict";
/**
 * Performance Controller
 * Handles HTTP requests for performance testing
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.runLoadTest = exports.getTestResults = exports.getTestResult = exports.startTest = exports.getAllTests = exports.getTest = exports.createTest = void 0;
const logger_1 = require("../../utils/logger");
const perfTestService_1 = __importDefault(require("../../services/performance/perfTestService"));
const loadTestService_1 = __importDefault(require("../../services/performance/loadTestService"));
const prismaClient_1 = __importDefault(require("../../utils/prismaClient"));
const performance_1 = require("../../types/performance");
// Create logger
const logger = (0, logger_1.createLogger)('PerformanceController');
/**
 * Create a new performance test
 * @route POST /api/performance/tests
 */
const createTest = async (req, res) => {
    try {
        const config = req.body;
        // Validate request
        if (!config.name || !config.testType || !config.duration || !config.concurrency) {
            res.status(400).json({
                status: 'error',
                message: 'Missing required test configuration parameters'
            });
            return;
        }
        // Create test
        const test = await perfTestService_1.default.createTest(config);
        res.status(201).json({
            status: 'success',
            data: test
        });
    }
    catch (error) {
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
exports.createTest = createTest;
/**
 * Get a performance test by ID
 * @route GET /api/performance/tests/:id
 */
const getTest = async (req, res) => {
    try {
        const { id } = req.params;
        // Get test
        const test = await perfTestService_1.default.getTest(id);
        res.status(200).json({
            status: 'success',
            data: test
        });
    }
    catch (error) {
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
exports.getTest = getTest;
/**
 * Get all performance tests
 * @route GET /api/performance/tests
 */
const getAllTests = async (req, res) => {
    try {
        const { type } = req.query;
        // Get tests
        const tests = await perfTestService_1.default.getAllTests(type ? type : undefined);
        res.status(200).json({
            status: 'success',
            data: tests
        });
    }
    catch (error) {
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
exports.getAllTests = getAllTests;
/**
 * Start a performance test
 * @route POST /api/performance/tests/:id/start
 */
const startTest = async (req, res) => {
    try {
        const { id } = req.params;
        // Start test
        const result = await perfTestService_1.default.startTest(id);
        res.status(200).json({
            status: 'success',
            data: result
        });
    }
    catch (error) {
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
exports.startTest = startTest;
/**
 * Get a test result by ID
 * @route GET /api/performance/results/:id
 */
const getTestResult = async (req, res) => {
    try {
        const { id } = req.params;
        // Get result
        const result = await perfTestService_1.default.getTestResult(id);
        res.status(200).json({
            status: 'success',
            data: result
        });
    }
    catch (error) {
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
exports.getTestResult = getTestResult;
/**
 * Get all test results for a test
 * @route GET /api/performance/tests/:id/results
 */
const getTestResults = async (req, res) => {
    try {
        const { id } = req.params;
        // Get results
        const results = await perfTestService_1.default.getTestResults(id);
        res.status(200).json({
            status: 'success',
            data: results
        });
    }
    catch (error) {
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
exports.getTestResults = getTestResults;
/**
 * Run a load test
 * @route POST /api/performance/load-test
 */
const runLoadTest = async (req, res) => {
    try {
        const config = req.body;
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
        const dbTest = await prismaClient_1.default.performanceTest.create({
            data: {
                name: config.name,
                description: config.description,
                testType: performance_1.PerformanceTestType.LOAD_TEST,
                duration,
                concurrency: Math.max(...config.stages.map(stage => stage.target)),
                options: {
                    stages: config.stages,
                    targetEndpoints: config.targetEndpoints
                }
            }
        });
        // Run load test
        const resultId = await loadTestService_1.default.runLoadTest({
            ...config,
            id: dbTest.id,
            duration
        }, async (result) => {
            // Save or update result in database
            try {
                const existingResult = await prismaClient_1.default.performanceTestResult.findUnique({
                    where: { id: result.id }
                });
                if (existingResult) {
                    // Update existing result
                    await prismaClient_1.default.performanceTestResult.update({
                        where: { id: result.id },
                        data: {
                            status: result.status,
                            endTime: result.endTime ? new Date(result.endTime) : undefined,
                            duration: result.duration,
                            metrics: result.metrics,
                            errors: result.errors
                        }
                    });
                }
                else {
                    // Create new result
                    await prismaClient_1.default.performanceTestResult.create({
                        data: {
                            id: result.id,
                            testId: result.testId,
                            status: result.status,
                            startTime: new Date(result.startTime),
                            endTime: result.endTime ? new Date(result.endTime) : undefined,
                            duration: result.duration,
                            metrics: result.metrics,
                            errors: result.errors
                        }
                    });
                }
            }
            catch (error) {
                logger.error('Error saving load test result', {
                    error: error instanceof Error ? error.message : String(error),
                    userId: req.user?.id,
                    resultId: result.id
                });
            }
        });
        res.status(202).json({
            status: 'success',
            message: 'Load test started',
            data: {
                testId: dbTest.id,
                resultId
            }
        });
    }
    catch (error) {
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
exports.runLoadTest = runLoadTest;
//# sourceMappingURL=performanceController.js.map