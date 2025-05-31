"use strict";
/**
 * Performance Test Service
 * Handles performance testing operations
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const prismaClient_1 = __importDefault(require("../../utils/prismaClient"));
const logger_1 = require("../../utils/logger");
const performance_1 = require("../../types/performance");
// Create logger
const logger = (0, logger_1.createLogger)('PerformanceTestService');
/**
 * Performance Test Service
 * Manages performance testing
 */
class PerformanceTestService {
    /**
     * Create a new performance test
     * @param config - Performance test configuration
     * @returns Created test configuration
     */
    async createTest(config) {
        try {
            logger.info(`Creating performance test: ${config.name}`);
            const test = await prismaClient_1.default.performanceTest.create({
                data: {
                    name: config.name,
                    description: config.description,
                    testType: config.testType,
                    duration: config.duration,
                    concurrency: config.concurrency,
                    rampUp: config.rampUp,
                    targetEndpoint: config.targetEndpoint,
                    modelId: config.modelId,
                    strategyId: config.strategyId,
                    symbol: config.symbol,
                    timeframe: config.timeframe,
                    options: config.options
                }
            });
            return {
                id: test.id,
                name: test.name,
                description: test.description || undefined,
                testType: test.testType,
                duration: test.duration,
                concurrency: test.concurrency,
                rampUp: test.rampUp || undefined,
                targetEndpoint: test.targetEndpoint || undefined,
                modelId: test.modelId || undefined,
                strategyId: test.strategyId || undefined,
                symbol: test.symbol || undefined,
                timeframe: test.timeframe || undefined,
                options: test.options || undefined,
                createdAt: test.createdAt.toISOString(),
                updatedAt: test.updatedAt.toISOString()
            };
        }
        catch (error) {
            const logData = {
                config,
                error: error instanceof Error ? error.message : String(error)
            };
            logger.error(`Error creating performance test: ${config.name}`, logData);
            throw error;
        }
    }
    /**
     * Get performance test by ID
     * @param id - Test ID
     * @returns Test configuration
     */
    async getTest(id) {
        try {
            logger.info(`Getting performance test: ${id}`);
            const test = await prismaClient_1.default.performanceTest.findUnique({
                where: { id }
            });
            if (!test) {
                throw new Error(`Performance test not found: ${id}`);
            }
            return {
                id: test.id,
                name: test.name,
                description: test.description || undefined,
                testType: test.testType,
                duration: test.duration,
                concurrency: test.concurrency,
                rampUp: test.rampUp || undefined,
                targetEndpoint: test.targetEndpoint || undefined,
                modelId: test.modelId || undefined,
                strategyId: test.strategyId || undefined,
                symbol: test.symbol || undefined,
                timeframe: test.timeframe || undefined,
                options: test.options || undefined,
                createdAt: test.createdAt.toISOString(),
                updatedAt: test.updatedAt.toISOString()
            };
        }
        catch (error) {
            const logData = {
                id,
                error: error instanceof Error ? error.message : String(error)
            };
            logger.error(`Error getting performance test: ${id}`, logData);
            throw error;
        }
    }
    /**
     * Get all performance tests
     * @param testType - Optional filter by test type
     * @returns Array of test configurations
     */
    async getAllTests(testType) {
        try {
            logger.info('Getting all performance tests');
            const where = testType ? { testType } : undefined;
            const tests = await prismaClient_1.default.performanceTest.findMany({
                where,
                orderBy: { createdAt: 'desc' }
            });
            return tests.map(test => ({
                id: test.id,
                name: test.name,
                description: test.description || undefined,
                testType: test.testType,
                duration: test.duration,
                concurrency: test.concurrency,
                rampUp: test.rampUp || undefined,
                targetEndpoint: test.targetEndpoint || undefined,
                modelId: test.modelId || undefined,
                strategyId: test.strategyId || undefined,
                symbol: test.symbol || undefined,
                timeframe: test.timeframe || undefined,
                options: test.options || undefined,
                createdAt: test.createdAt.toISOString(),
                updatedAt: test.updatedAt.toISOString()
            }));
        }
        catch (error) {
            const logData = {
                testType,
                error: error instanceof Error ? error.message : String(error)
            };
            logger.error('Error getting all performance tests', logData);
            throw error;
        }
    }
    /**
     * Start a performance test
     * @param id - Test ID
     * @returns Test result
     */
    async startTest(id) {
        try {
            logger.info(`Starting performance test: ${id}`);
            // Get test configuration
            const test = await this.getTest(id);
            // Create test result
            const testResult = await prismaClient_1.default.performanceTestResult.create({
                data: {
                    testId: id,
                    status: performance_1.TestStatus.RUNNING,
                    startTime: new Date(),
                    metrics: {
                        throughput: 0,
                        latencyP50: 0,
                        latencyP90: 0,
                        latencyP95: 0,
                        latencyP99: 0,
                        latencyAvg: 0,
                        latencyMin: 0,
                        latencyMax: 0,
                        errorRate: 0,
                        successRate: 0
                    }
                }
            });
            // Start test asynchronously
            this.executeTest(test, testResult.id).catch(error => {
                logger.error(`Error executing test: ${id}`, {
                    error: error instanceof Error ? error.message : String(error)
                });
            });
            return {
                id: testResult.id,
                testId: testResult.testId,
                testConfig: test,
                status: testResult.status,
                startTime: testResult.startTime.toISOString(),
                metrics: testResult.metrics,
                createdAt: testResult.createdAt.toISOString(),
                updatedAt: testResult.updatedAt.toISOString()
            };
        }
        catch (error) {
            const logData = {
                id,
                error: error instanceof Error ? error.message : String(error)
            };
            logger.error(`Error starting performance test: ${id}`, logData);
            throw error;
        }
    }
    /**
     * Execute a performance test
     * @param test - Test configuration
     * @param resultId - Test result ID
     * @private
     */
    async executeTest(test, resultId) {
        try {
            logger.info(`Executing performance test: ${test.id}, result: ${resultId}`);
            // Implementation will depend on test type
            switch (test.testType) {
                case performance_1.PerformanceTestType.API_LATENCY:
                    await this.executeApiLatencyTest(test, resultId);
                    break;
                case performance_1.PerformanceTestType.ML_PREDICTION_THROUGHPUT:
                    await this.executeMlPredictionTest(test, resultId);
                    break;
                case performance_1.PerformanceTestType.SIGNAL_GENERATION:
                    await this.executeSignalGenerationTest(test, resultId);
                    break;
                case performance_1.PerformanceTestType.STRATEGY_EXECUTION:
                    await this.executeStrategyExecutionTest(test, resultId);
                    break;
                case performance_1.PerformanceTestType.END_TO_END:
                    await this.executeEndToEndTest(test, resultId);
                    break;
                case performance_1.PerformanceTestType.LOAD_TEST:
                    await this.executeLoadTest(test, resultId);
                    break;
                case performance_1.PerformanceTestType.STRESS_TEST:
                    await this.executeStressTest(test, resultId);
                    break;
                default:
                    throw new Error(`Unsupported test type: ${test.testType}`);
            }
        }
        catch (error) {
            const logData = {
                testId: test.id,
                resultId,
                error: error instanceof Error ? error.message : String(error)
            };
            logger.error(`Error executing performance test: ${test.id}`, logData);
            // Update test result with failure
            await this.updateTestResult(resultId, {
                status: performance_1.TestStatus.FAILED,
                endTime: new Date().toISOString(),
                errors: [{
                        timestamp: new Date().toISOString(),
                        message: error instanceof Error ? error.message : String(error),
                        count: 1
                    }]
            });
        }
    }
    /**
     * Execute API latency test
     * @param test - Test configuration
     * @param resultId - Test result ID
     * @private
     */
    async executeApiLatencyTest(test, resultId) {
        // This is a placeholder implementation
        // In a real system, this would use a tool like k6, Artillery, or a custom test runner
        // Mark as completed with sample metrics
        await this.updateTestResult(resultId, {
            status: performance_1.TestStatus.COMPLETED,
            endTime: new Date().toISOString(),
            duration: test.duration * 1000,
            metrics: {
                throughput: 100,
                latencyP50: 50,
                latencyP90: 80,
                latencyP95: 90,
                latencyP99: 120,
                latencyAvg: 60,
                latencyMin: 20,
                latencyMax: 150,
                errorRate: 0.5,
                successRate: 99.5,
                customMetrics: {
                    requestsPerSecond: 100,
                    connections: test.concurrency
                }
            }
        });
        // Generate optimization recommendations
        await this.generateRecommendations(resultId);
    }
    /**
     * Execute ML prediction test
     * @param test - Test configuration
     * @param resultId - Test result ID
     * @private
     */
    async executeMlPredictionTest(test, resultId) {
        // This is a placeholder implementation
        // Mark as completed with sample metrics
        await this.updateTestResult(resultId, {
            status: performance_1.TestStatus.COMPLETED,
            endTime: new Date().toISOString(),
            duration: test.duration * 1000,
            metrics: {
                throughput: 50,
                latencyP50: 100,
                latencyP90: 150,
                latencyP95: 180,
                latencyP99: 250,
                latencyAvg: 120,
                latencyMin: 80,
                latencyMax: 300,
                errorRate: 0,
                successRate: 100,
                predictionLatencyAvg: 120,
                predictionThroughput: 50,
                modelLoadTime: 200,
                customMetrics: {
                    batchSize: test.options?.batchSize || 1,
                    inferenceTime: 80
                }
            }
        });
        // Generate optimization recommendations
        await this.generateRecommendations(resultId);
    }
    /**
     * Execute signal generation test
     * @param test - Test configuration
     * @param resultId - Test result ID
     * @private
     */
    async executeSignalGenerationTest(test, resultId) {
        // Placeholder implementation
        await this.updateTestResult(resultId, {
            status: performance_1.TestStatus.COMPLETED,
            endTime: new Date().toISOString(),
            duration: test.duration * 1000,
            metrics: {
                throughput: 30,
                latencyP50: 150,
                latencyP90: 220,
                latencyP95: 250,
                latencyP99: 320,
                latencyAvg: 180,
                latencyMin: 100,
                latencyMax: 400,
                errorRate: 1.2,
                successRate: 98.8,
                signalGenerationLatencyAvg: 180,
                signalsThroughput: 30
            }
        });
        await this.generateRecommendations(resultId);
    }
    /**
     * Execute strategy execution test
     * @param test - Test configuration
     * @param resultId - Test result ID
     * @private
     */
    async executeStrategyExecutionTest(test, resultId) {
        // Placeholder implementation
        await this.updateTestResult(resultId, {
            status: performance_1.TestStatus.COMPLETED,
            endTime: new Date().toISOString(),
            duration: test.duration * 1000,
            metrics: {
                throughput: 20,
                latencyP50: 200,
                latencyP90: 300,
                latencyP95: 350,
                latencyP99: 450,
                latencyAvg: 230,
                latencyMin: 150,
                latencyMax: 500,
                errorRate: 0.8,
                successRate: 99.2,
                strategyExecutionLatencyAvg: 230,
                strategyThroughput: 20
            }
        });
        await this.generateRecommendations(resultId);
    }
    /**
     * Execute end-to-end test
     * @param test - Test configuration
     * @param resultId - Test result ID
     * @private
     */
    async executeEndToEndTest(test, resultId) {
        // Placeholder implementation
        await this.updateTestResult(resultId, {
            status: performance_1.TestStatus.COMPLETED,
            endTime: new Date().toISOString(),
            duration: test.duration * 1000,
            metrics: {
                throughput: 10,
                latencyP50: 500,
                latencyP90: 700,
                latencyP95: 800,
                latencyP99: 1000,
                latencyAvg: 550,
                latencyMin: 300,
                latencyMax: 1200,
                errorRate: 2.5,
                successRate: 97.5,
                endToEndLatencyAvg: 550,
                endToEndThroughput: 10
            }
        });
        await this.generateRecommendations(resultId);
    }
    /**
     * Execute load test
     * @param test - Test configuration
     * @param resultId - Test result ID
     * @private
     */
    async executeLoadTest(test, resultId) {
        // Placeholder implementation
        await this.updateTestResult(resultId, {
            status: performance_1.TestStatus.COMPLETED,
            endTime: new Date().toISOString(),
            duration: test.duration * 1000,
            metrics: {
                throughput: 200,
                latencyP50: 80,
                latencyP90: 120,
                latencyP95: 150,
                latencyP99: 200,
                latencyAvg: 90,
                latencyMin: 30,
                latencyMax: 250,
                errorRate: 1.5,
                successRate: 98.5,
                cpuUsageAvg: 45,
                cpuUsageMax: 75,
                memoryUsageAvg: 512,
                memoryUsageMax: 768
            }
        });
        await this.generateRecommendations(resultId);
    }
    /**
     * Execute stress test
     * @param test - Test configuration
     * @param resultId - Test result ID
     * @private
     */
    async executeStressTest(test, resultId) {
        // Placeholder implementation
        await this.updateTestResult(resultId, {
            status: performance_1.TestStatus.COMPLETED,
            endTime: new Date().toISOString(),
            duration: test.duration * 1000,
            metrics: {
                throughput: 350,
                latencyP50: 120,
                latencyP90: 200,
                latencyP95: 250,
                latencyP99: 350,
                latencyAvg: 150,
                latencyMin: 50,
                latencyMax: 500,
                errorRate: 5.5,
                successRate: 94.5,
                cpuUsageAvg: 85,
                cpuUsageMax: 95,
                memoryUsageAvg: 1024,
                memoryUsageMax: 1536
            }
        });
        await this.generateRecommendations(resultId);
    }
    /**
     * Update test result
     * @param resultId - Test result ID
     * @param update - Update data
     * @private
     */
    async updateTestResult(resultId, update) {
        try {
            logger.debug(`Updating test result: ${resultId}`);
            // Get current result
            const currentResult = await prismaClient_1.default.performanceTestResult.findUnique({
                where: { id: resultId }
            });
            if (!currentResult) {
                throw new Error(`Test result not found: ${resultId}`);
            }
            // Merge metrics
            const currentMetrics = currentResult.metrics;
            const metrics = update.metrics
                ? { ...currentMetrics, ...update.metrics }
                : currentMetrics;
            // Update result
            await prismaClient_1.default.performanceTestResult.update({
                where: { id: resultId },
                data: {
                    status: update.status || currentResult.status,
                    endTime: update.endTime ? new Date(update.endTime) : currentResult.endTime,
                    duration: update.duration !== undefined ? update.duration : currentResult.duration,
                    metrics: metrics,
                    errors: update.errors ? update.errors : currentResult.errors
                }
            });
        }
        catch (error) {
            const logData = {
                resultId,
                update,
                error: error instanceof Error ? error.message : String(error)
            };
            logger.error(`Error updating test result: ${resultId}`, logData);
            throw error;
        }
    }
    /**
     * Generate optimization recommendations
     * @param resultId - Test result ID
     * @private
     */
    async generateRecommendations(resultId) {
        try {
            logger.info(`Generating recommendations for test result: ${resultId}`);
            // Get test result
            const result = await prismaClient_1.default.performanceTestResult.findUnique({
                where: { id: resultId },
                include: { test: true }
            });
            if (!result) {
                throw new Error(`Test result not found: ${resultId}`);
            }
            const metrics = result.metrics;
            // Generate recommendations based on metrics
            const recommendations = [];
            // Check latency
            if (metrics.latencyP95 > 200) {
                recommendations.push({
                    category: performance_1.OptimizationCategory.API_ENDPOINT,
                    impact: performance_1.OptimizationImpact.HIGH,
                    description: 'High 95th percentile latency detected. Optimize API endpoint response time.',
                    implementation: 'Consider adding caching, optimizing database queries, or implementing query optimization.',
                    estimatedImprovement: '40-60% reduction in p95 latency'
                });
            }
            // Check error rate
            if (metrics.errorRate > 1) {
                recommendations.push({
                    category: performance_1.OptimizationCategory.CODE_OPTIMIZATION,
                    impact: performance_1.OptimizationImpact.CRITICAL,
                    description: `High error rate (${metrics.errorRate}%) detected. Investigate error sources.`,
                    implementation: 'Add error handling, retry logic, or fix bugs in the code.',
                    estimatedImprovement: '90-100% reduction in error rate'
                });
            }
            // Check CPU usage
            if (metrics.cpuUsageMax && metrics.cpuUsageMax > 80) {
                recommendations.push({
                    category: performance_1.OptimizationCategory.CONCURRENCY,
                    impact: performance_1.OptimizationImpact.MEDIUM,
                    description: 'High CPU usage detected. Optimize concurrency settings.',
                    implementation: 'Adjust concurrency settings, implement worker pools, or optimize CPU-intensive operations.',
                    estimatedImprovement: '30-40% reduction in CPU usage'
                });
            }
            // Check memory usage
            if (metrics.memoryUsageMax && metrics.memoryUsageMax > 1024) {
                recommendations.push({
                    category: performance_1.OptimizationCategory.MEMORY_USAGE,
                    impact: performance_1.OptimizationImpact.MEDIUM,
                    description: 'High memory usage detected. Optimize memory consumption.',
                    implementation: 'Implement memory pooling, reduce object allocations, or optimize data structures.',
                    estimatedImprovement: '20-30% reduction in memory usage'
                });
            }
            // ML-specific recommendations
            if (result.test.testType === 'ML_PREDICTION_THROUGHPUT' && metrics.predictionLatencyAvg && metrics.predictionLatencyAvg > 100) {
                recommendations.push({
                    category: performance_1.OptimizationCategory.ML_MODEL,
                    impact: performance_1.OptimizationImpact.HIGH,
                    description: 'High ML prediction latency detected. Optimize ML model performance.',
                    implementation: 'Consider model quantization, pruning, or using a smaller model architecture.',
                    estimatedImprovement: '50-70% reduction in prediction latency'
                });
            }
            // Add recommendations to database
            for (const rec of recommendations) {
                await prismaClient_1.default.optimizationRecommendation.create({
                    data: {
                        testResultId: resultId,
                        category: rec.category,
                        impact: rec.impact,
                        description: rec.description,
                        implementation: rec.implementation,
                        estimatedImprovement: rec.estimatedImprovement
                    }
                });
            }
            logger.info(`Generated ${recommendations.length} recommendations for test result: ${resultId}`);
        }
        catch (error) {
            const logData = {
                resultId,
                error: error instanceof Error ? error.message : String(error)
            };
            logger.error(`Error generating recommendations for test result: ${resultId}`, logData);
            throw error;
        }
    }
    /**
     * Get test result by ID
     * @param id - Result ID
     * @returns Test result
     */
    async getTestResult(id) {
        try {
            logger.info(`Getting test result: ${id}`);
            const result = await prismaClient_1.default.performanceTestResult.findUnique({
                where: { id },
                include: {
                    test: true,
                    recommendations: true
                }
            });
            if (!result) {
                throw new Error(`Test result not found: ${id}`);
            }
            // Convert test to PerformanceTestConfig
            const testConfig = {
                id: result.test.id,
                name: result.test.name,
                description: result.test.description || undefined,
                testType: result.test.testType,
                duration: result.test.duration,
                concurrency: result.test.concurrency,
                rampUp: result.test.rampUp || undefined,
                targetEndpoint: result.test.targetEndpoint || undefined,
                modelId: result.test.modelId || undefined,
                strategyId: result.test.strategyId || undefined,
                symbol: result.test.symbol || undefined,
                timeframe: result.test.timeframe || undefined,
                options: result.test.options || undefined,
                createdAt: result.test.createdAt.toISOString(),
                updatedAt: result.test.updatedAt.toISOString()
            };
            return {
                id: result.id,
                testId: result.testId,
                testConfig,
                status: result.status,
                startTime: result.startTime.toISOString(),
                endTime: result.endTime ? result.endTime.toISOString() : undefined,
                duration: result.duration || undefined,
                metrics: result.metrics,
                errors: result.errors || undefined,
                createdAt: result.createdAt.toISOString(),
                updatedAt: result.updatedAt.toISOString()
            };
        }
        catch (error) {
            const logData = {
                id,
                error: error instanceof Error ? error.message : String(error)
            };
            logger.error(`Error getting test result: ${id}`, logData);
            throw error;
        }
    }
    /**
     * Get all test results for a test
     * @param testId - Test ID
     * @returns Array of test results
     */
    async getTestResults(testId) {
        try {
            logger.info(`Getting test results for test: ${testId}`);
            const results = await prismaClient_1.default.performanceTestResult.findMany({
                where: { testId },
                include: {
                    test: true,
                    recommendations: true
                },
                orderBy: { createdAt: 'desc' }
            });
            return results.map(result => {
                // Convert test to PerformanceTestConfig
                const testConfig = {
                    id: result.test.id,
                    name: result.test.name,
                    description: result.test.description || undefined,
                    testType: result.test.testType,
                    duration: result.test.duration,
                    concurrency: result.test.concurrency,
                    rampUp: result.test.rampUp || undefined,
                    targetEndpoint: result.test.targetEndpoint || undefined,
                    modelId: result.test.modelId || undefined,
                    strategyId: result.test.strategyId || undefined,
                    symbol: result.test.symbol || undefined,
                    timeframe: result.test.timeframe || undefined,
                    options: result.test.options || undefined,
                    createdAt: result.test.createdAt.toISOString(),
                    updatedAt: result.test.updatedAt.toISOString()
                };
                return {
                    id: result.id,
                    testId: result.testId,
                    testConfig,
                    status: result.status,
                    startTime: result.startTime.toISOString(),
                    endTime: result.endTime ? result.endTime.toISOString() : undefined,
                    duration: result.duration || undefined,
                    metrics: result.metrics,
                    errors: result.errors || undefined,
                    createdAt: result.createdAt.toISOString(),
                    updatedAt: result.updatedAt.toISOString()
                };
            });
        }
        catch (error) {
            const logData = {
                testId,
                error: error instanceof Error ? error.message : String(error)
            };
            logger.error(`Error getting test results for test: ${testId}`, logData);
            throw error;
        }
    }
}
// Create singleton instance
const perfTestService = new PerformanceTestService();
exports.default = perfTestService;
//# sourceMappingURL=perfTestService.js.map