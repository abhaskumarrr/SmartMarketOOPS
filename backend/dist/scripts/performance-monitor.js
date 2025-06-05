"use strict";
/**
 * Performance Monitoring Script
 * Monitors system performance and generates reports
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.PerformanceMonitor = void 0;
const perf_hooks_1 = require("perf_hooks");
const cacheService_1 = require("../services/cacheService");
const databaseOptimizationService_1 = require("../services/databaseOptimizationService");
const axios_1 = __importDefault(require("axios"));
class PerformanceMonitor {
    constructor(baseUrl = 'http://localhost:3333', monitoringInterval = 30000) {
        this.isRunning = false;
        this.baseUrl = baseUrl;
        this.monitoringInterval = monitoringInterval;
    }
    async start() {
        console.log('Starting performance monitoring...');
        this.isRunning = true;
        while (this.isRunning) {
            try {
                const report = await this.generateReport();
                await this.analyzeReport(report);
                await this.sleep(this.monitoringInterval);
            }
            catch (error) {
                console.error('Error during monitoring:', error);
                await this.sleep(5000); // Wait 5 seconds before retrying
            }
        }
    }
    stop() {
        console.log('Stopping performance monitoring...');
        this.isRunning = false;
    }
    async generateReport() {
        const start = perf_hooks_1.performance.now();
        // Test API response time
        const apiStart = perf_hooks_1.performance.now();
        try {
            await axios_1.default.get(`${this.baseUrl}/health`);
        }
        catch (error) {
            console.warn('Health check failed:', error);
        }
        const apiResponseTime = perf_hooks_1.performance.now() - apiStart;
        // Get system metrics
        const memoryUsage = process.memoryUsage();
        const uptime = process.uptime();
        // Get database metrics
        const dbService = (0, databaseOptimizationService_1.getDatabaseOptimizationService)();
        const dbStats = dbService ? dbService.getQueryStats() : null;
        // Get cache metrics
        const cacheService = (0, cacheService_1.getCacheService)();
        const cacheStats = cacheService ? cacheService.getStats() : null;
        // Get detailed metrics from API
        let apiMetrics = null;
        try {
            const response = await axios_1.default.get(`${this.baseUrl}/metrics`);
            apiMetrics = response.data;
        }
        catch (error) {
            console.warn('Failed to get API metrics:', error);
        }
        const report = {
            timestamp: new Date().toISOString(),
            system: {
                uptime,
                memory: memoryUsage,
                cpu: this.getCpuUsage(),
            },
            api: {
                responseTime: apiResponseTime,
                throughput: apiMetrics?.requests?.recent?.requests || 0,
                errorRate: apiMetrics?.requests?.recent?.errorRate || 0,
            },
            database: {
                avgQueryTime: dbStats?.avgQueryTime || 0,
                slowQueries: dbStats?.slowQueries || 0,
                connectionPool: apiMetrics?.database || null,
            },
            cache: {
                hitRate: cacheStats?.hitRate || 0,
                memoryUsage: 0, // Would need Redis info for this
                operations: cacheStats || null,
            },
        };
        return report;
    }
    getCpuUsage() {
        const cpuUsage = process.cpuUsage();
        return {
            user: cpuUsage.user / 1000000, // Convert to seconds
            system: cpuUsage.system / 1000000,
        };
    }
    async analyzeReport(report) {
        console.log('\n=== Performance Report ===');
        console.log(`Timestamp: ${report.timestamp}`);
        // System analysis
        console.log('\n--- System Metrics ---');
        console.log(`Uptime: ${Math.floor(report.system.uptime / 3600)}h ${Math.floor((report.system.uptime % 3600) / 60)}m`);
        console.log(`Memory Usage: ${Math.round(report.system.memory.heapUsed / 1024 / 1024)}MB / ${Math.round(report.system.memory.heapTotal / 1024 / 1024)}MB`);
        console.log(`RSS: ${Math.round(report.system.memory.rss / 1024 / 1024)}MB`);
        // Memory warnings
        if (report.system.memory.heapUsed / report.system.memory.heapTotal > 0.8) {
            console.warn('⚠️  HIGH MEMORY USAGE: Heap usage > 80%');
        }
        if (report.system.memory.rss > 512 * 1024 * 1024) { // 512MB
            console.warn('⚠️  HIGH RSS MEMORY: > 512MB');
        }
        // API analysis
        console.log('\n--- API Metrics ---');
        console.log(`Response Time: ${report.api.responseTime.toFixed(2)}ms`);
        console.log(`Throughput: ${report.api.throughput} req/hour`);
        console.log(`Error Rate: ${report.api.errorRate.toFixed(2)}%`);
        // API warnings
        if (report.api.responseTime > 1000) {
            console.warn('⚠️  SLOW API RESPONSE: > 1000ms');
        }
        if (report.api.errorRate > 5) {
            console.warn('⚠️  HIGH ERROR RATE: > 5%');
        }
        // Database analysis
        console.log('\n--- Database Metrics ---');
        console.log(`Avg Query Time: ${report.database.avgQueryTime.toFixed(2)}ms`);
        console.log(`Slow Queries: ${report.database.slowQueries}`);
        // Database warnings
        if (report.database.avgQueryTime > 100) {
            console.warn('⚠️  SLOW DATABASE QUERIES: Avg > 100ms');
        }
        if (report.database.slowQueries > 10) {
            console.warn('⚠️  TOO MANY SLOW QUERIES: > 10');
        }
        // Cache analysis
        console.log('\n--- Cache Metrics ---');
        console.log(`Hit Rate: ${report.cache.hitRate.toFixed(2)}%`);
        // Cache warnings
        if (report.cache.hitRate < 70) {
            console.warn('⚠️  LOW CACHE HIT RATE: < 70%');
        }
        // Overall health score
        const healthScore = this.calculateHealthScore(report);
        console.log(`\n--- Overall Health Score: ${healthScore}/100 ---`);
        if (healthScore < 70) {
            console.warn('🚨 SYSTEM HEALTH CRITICAL: Score < 70');
        }
        else if (healthScore < 85) {
            console.warn('⚠️  SYSTEM HEALTH DEGRADED: Score < 85');
        }
        else {
            console.log('✅ SYSTEM HEALTH GOOD');
        }
        console.log('========================\n');
    }
    calculateHealthScore(report) {
        let score = 100;
        // Memory score (20 points)
        const memoryUsage = report.system.memory.heapUsed / report.system.memory.heapTotal;
        if (memoryUsage > 0.9)
            score -= 20;
        else if (memoryUsage > 0.8)
            score -= 15;
        else if (memoryUsage > 0.7)
            score -= 10;
        else if (memoryUsage > 0.6)
            score -= 5;
        // API response time score (25 points)
        if (report.api.responseTime > 2000)
            score -= 25;
        else if (report.api.responseTime > 1000)
            score -= 15;
        else if (report.api.responseTime > 500)
            score -= 10;
        else if (report.api.responseTime > 200)
            score -= 5;
        // Error rate score (20 points)
        if (report.api.errorRate > 10)
            score -= 20;
        else if (report.api.errorRate > 5)
            score -= 15;
        else if (report.api.errorRate > 2)
            score -= 10;
        else if (report.api.errorRate > 1)
            score -= 5;
        // Database score (20 points)
        if (report.database.avgQueryTime > 500)
            score -= 20;
        else if (report.database.avgQueryTime > 200)
            score -= 15;
        else if (report.database.avgQueryTime > 100)
            score -= 10;
        else if (report.database.avgQueryTime > 50)
            score -= 5;
        // Cache score (15 points)
        if (report.cache.hitRate < 50)
            score -= 15;
        else if (report.cache.hitRate < 60)
            score -= 10;
        else if (report.cache.hitRate < 70)
            score -= 8;
        else if (report.cache.hitRate < 80)
            score -= 5;
        return Math.max(0, score);
    }
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    async runOnce() {
        console.log('Running single performance check...');
        const report = await this.generateReport();
        await this.analyzeReport(report);
    }
    async loadTest(duration = 60000, concurrency = 10) {
        console.log(`Starting load test: ${concurrency} concurrent requests for ${duration}ms`);
        const startTime = Date.now();
        const results = [];
        const errors = [];
        const makeRequest = async () => {
            while (Date.now() - startTime < duration) {
                const requestStart = perf_hooks_1.performance.now();
                try {
                    await axios_1.default.get(`${this.baseUrl}/health`);
                    results.push(perf_hooks_1.performance.now() - requestStart);
                }
                catch (error) {
                    errors.push(Date.now());
                }
                await this.sleep(100); // Small delay between requests
            }
        };
        // Start concurrent requests
        const promises = Array(concurrency).fill(null).map(() => makeRequest());
        await Promise.all(promises);
        // Analyze results
        console.log('\n=== Load Test Results ===');
        console.log(`Total Requests: ${results.length}`);
        console.log(`Total Errors: ${errors.length}`);
        console.log(`Error Rate: ${((errors.length / (results.length + errors.length)) * 100).toFixed(2)}%`);
        if (results.length > 0) {
            const avgResponseTime = results.reduce((a, b) => a + b, 0) / results.length;
            const maxResponseTime = Math.max(...results);
            const minResponseTime = Math.min(...results);
            console.log(`Avg Response Time: ${avgResponseTime.toFixed(2)}ms`);
            console.log(`Min Response Time: ${minResponseTime.toFixed(2)}ms`);
            console.log(`Max Response Time: ${maxResponseTime.toFixed(2)}ms`);
            // Calculate percentiles
            const sorted = results.sort((a, b) => a - b);
            const p95 = sorted[Math.floor(sorted.length * 0.95)];
            const p99 = sorted[Math.floor(sorted.length * 0.99)];
            console.log(`95th Percentile: ${p95.toFixed(2)}ms`);
            console.log(`99th Percentile: ${p99.toFixed(2)}ms`);
        }
        console.log('========================\n');
    }
}
exports.PerformanceMonitor = PerformanceMonitor;
// CLI interface
const args = process.argv.slice(2);
const command = args[0] || 'once';
const monitor = new PerformanceMonitor();
switch (command) {
    case 'start':
        monitor.start().catch(console.error);
        break;
    case 'load-test':
        const duration = parseInt(args[1]) || 60000;
        const concurrency = parseInt(args[2]) || 10;
        monitor.loadTest(duration, concurrency).catch(console.error);
        break;
    case 'once':
    default:
        monitor.runOnce().catch(console.error);
        break;
}
// Handle graceful shutdown
process.on('SIGINT', () => {
    monitor.stop();
    process.exit(0);
});
//# sourceMappingURL=performance-monitor.js.map