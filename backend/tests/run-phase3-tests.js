"use strict";
/**
 * Phase 3 Test Runner
 * Comprehensive test suite runner for integration and testing phase
 */
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.Phase3TestRunner = void 0;
const child_process_1 = require("child_process");
const logger_1 = require("../src/utils/logger");
const fs = __importStar(require("fs"));
const path = __importStar(require("path"));
class Phase3TestRunner {
    constructor() {
        this.results = [];
        this.startTime = 0;
    }
    async runAllTests() {
        this.startTime = Date.now();
        logger_1.logger.info('🚀 Starting Phase 3: Integration & Testing');
        // Create test results directory
        const resultsDir = path.join(__dirname, 'results');
        if (!fs.existsSync(resultsDir)) {
            fs.mkdirSync(resultsDir, { recursive: true });
        }
        try {
            // Run test suites in order
            await this.runTestSuite('End-to-End Tests', 'npm run test:e2e');
            await this.runTestSuite('Performance Tests', 'npm run test:performance');
            await this.runTestSuite('Security Audit', 'npm run test:security');
            await this.runTestSuite('ML Integration Tests', 'npm run test:ml-integration');
            await this.runTestSuite('Unit Tests', 'npm run test:unit');
            await this.runTestSuite('Integration Tests', 'npm run test:integration');
            // Generate coverage report
            await this.generateCoverageReport();
            // Create summary
            const summary = this.createSummary();
            // Save results
            await this.saveResults(summary);
            // Print summary
            this.printSummary(summary);
            return summary;
        }
        catch (error) {
            logger_1.logger.error('Error running Phase 3 tests:', error);
            throw error;
        }
    }
    async runTestSuite(suiteName, command) {
        logger_1.logger.info(`\n📋 Running ${suiteName}...`);
        const startTime = Date.now();
        try {
            const output = (0, child_process_1.execSync)(command, {
                cwd: path.join(__dirname, '..'),
                encoding: 'utf8',
                timeout: 300000, // 5 minutes timeout
                stdio: 'pipe'
            });
            const duration = Date.now() - startTime;
            const result = this.parseTestOutput(suiteName, output, duration);
            this.results.push(result);
            logger_1.logger.info(`✅ ${suiteName} completed: ${result.passed} passed, ${result.failed} failed`);
        }
        catch (error) {
            const duration = Date.now() - startTime;
            const result = {
                suite: suiteName,
                passed: 0,
                failed: 1,
                skipped: 0,
                duration,
                errors: [error.message || 'Unknown error']
            };
            this.results.push(result);
            logger_1.logger.error(`❌ ${suiteName} failed:`, error.message);
        }
    }
    parseTestOutput(suiteName, output, duration) {
        // Parse Jest output
        const lines = output.split('\n');
        let passed = 0;
        let failed = 0;
        let skipped = 0;
        const errors = [];
        for (const line of lines) {
            // Jest test results parsing
            if (line.includes('✓') || line.includes('PASS')) {
                passed++;
            }
            else if (line.includes('✗') || line.includes('FAIL')) {
                failed++;
                errors.push(line.trim());
            }
            else if (line.includes('○') || line.includes('SKIP')) {
                skipped++;
            }
            // Extract specific test counts from summary
            const testMatch = line.match(/Tests:\s+(\d+)\s+failed,\s+(\d+)\s+passed,\s+(\d+)\s+total/);
            if (testMatch) {
                failed = parseInt(testMatch[1]);
                passed = parseInt(testMatch[2]);
            }
            const passOnlyMatch = line.match(/Tests:\s+(\d+)\s+passed,\s+(\d+)\s+total/);
            if (passOnlyMatch) {
                passed = parseInt(passOnlyMatch[1]);
                failed = 0;
            }
        }
        return {
            suite: suiteName,
            passed,
            failed,
            skipped,
            duration,
            errors
        };
    }
    async generateCoverageReport() {
        logger_1.logger.info('\n📊 Generating coverage report...');
        try {
            const output = (0, child_process_1.execSync)('npm run test:coverage', {
                cwd: path.join(__dirname, '..'),
                encoding: 'utf8',
                timeout: 120000 // 2 minutes timeout
            });
            // Parse coverage from output
            const coverageMatch = output.match(/All files\s+\|\s+([\d.]+)/);
            const overallCoverage = coverageMatch ? parseFloat(coverageMatch[1]) : 0;
            // Update results with coverage info
            this.results.forEach(result => {
                result.coverage = overallCoverage;
            });
            logger_1.logger.info(`📈 Overall test coverage: ${overallCoverage}%`);
        }
        catch (error) {
            logger_1.logger.warn('Could not generate coverage report:', error);
        }
    }
    createSummary() {
        const totalTests = this.results.reduce((sum, r) => sum + r.passed + r.failed + r.skipped, 0);
        const totalPassed = this.results.reduce((sum, r) => sum + r.passed, 0);
        const totalFailed = this.results.reduce((sum, r) => sum + r.failed, 0);
        const totalSkipped = this.results.reduce((sum, r) => sum + r.skipped, 0);
        const totalDuration = Date.now() - this.startTime;
        const overallCoverage = this.results.find(r => r.coverage)?.coverage || 0;
        return {
            totalSuites: this.results.length,
            totalTests,
            totalPassed,
            totalFailed,
            totalSkipped,
            totalDuration,
            overallCoverage,
            results: this.results,
            timestamp: new Date().toISOString()
        };
    }
    async saveResults(summary) {
        const resultsDir = path.join(__dirname, 'results');
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        // Save detailed results
        const detailedPath = path.join(resultsDir, `phase3-detailed-${timestamp}.json`);
        fs.writeFileSync(detailedPath, JSON.stringify(summary, null, 2));
        // Save summary report
        const summaryPath = path.join(resultsDir, `phase3-summary-${timestamp}.md`);
        const summaryReport = this.generateMarkdownReport(summary);
        fs.writeFileSync(summaryPath, summaryReport);
        // Save latest results (overwrite)
        const latestPath = path.join(resultsDir, 'phase3-latest.json');
        fs.writeFileSync(latestPath, JSON.stringify(summary, null, 2));
        logger_1.logger.info(`📄 Test results saved to ${resultsDir}`);
    }
    generateMarkdownReport(summary) {
        const successRate = (summary.totalPassed / summary.totalTests * 100).toFixed(1);
        const durationMinutes = (summary.totalDuration / 60000).toFixed(1);
        let report = `# Phase 3: Integration & Testing Results\n\n`;
        report += `**Generated:** ${summary.timestamp}\n\n`;
        report += `## Summary\n\n`;
        report += `- **Total Test Suites:** ${summary.totalSuites}\n`;
        report += `- **Total Tests:** ${summary.totalTests}\n`;
        report += `- **Passed:** ${summary.totalPassed} (${successRate}%)\n`;
        report += `- **Failed:** ${summary.totalFailed}\n`;
        report += `- **Skipped:** ${summary.totalSkipped}\n`;
        report += `- **Duration:** ${durationMinutes} minutes\n`;
        report += `- **Coverage:** ${summary.overallCoverage}%\n\n`;
        // Overall status
        const overallStatus = summary.totalFailed === 0 ? '✅ PASSED' : '❌ FAILED';
        report += `## Overall Status: ${overallStatus}\n\n`;
        // Detailed results
        report += `## Detailed Results\n\n`;
        for (const result of summary.results) {
            const status = result.failed === 0 ? '✅' : '❌';
            const duration = (result.duration / 1000).toFixed(1);
            report += `### ${status} ${result.suite}\n\n`;
            report += `- **Passed:** ${result.passed}\n`;
            report += `- **Failed:** ${result.failed}\n`;
            report += `- **Skipped:** ${result.skipped}\n`;
            report += `- **Duration:** ${duration}s\n`;
            if (result.coverage) {
                report += `- **Coverage:** ${result.coverage}%\n`;
            }
            if (result.errors.length > 0) {
                report += `\n**Errors:**\n`;
                for (const error of result.errors) {
                    report += `- ${error}\n`;
                }
            }
            report += `\n`;
        }
        // Recommendations
        report += `## Recommendations\n\n`;
        if (summary.totalFailed > 0) {
            report += `- ❗ **Fix failing tests** before proceeding to Phase 4\n`;
            report += `- 🔍 **Review error logs** for detailed failure information\n`;
        }
        if (summary.overallCoverage < 80) {
            report += `- 📈 **Improve test coverage** (current: ${summary.overallCoverage}%, target: 80%+)\n`;
        }
        if (summary.totalDuration > 600000) { // 10 minutes
            report += `- ⚡ **Optimize test performance** (current duration: ${durationMinutes} minutes)\n`;
        }
        if (summary.totalFailed === 0 && summary.overallCoverage >= 80) {
            report += `- 🎉 **All tests passed!** Ready to proceed to Phase 4: Deployment\n`;
            report += `- ✅ **Test coverage meets requirements**\n`;
            report += `- 🚀 **System is ready for production deployment**\n`;
        }
        return report;
    }
    printSummary(summary) {
        const successRate = (summary.totalPassed / summary.totalTests * 100).toFixed(1);
        const durationMinutes = (summary.totalDuration / 60000).toFixed(1);
        console.log('\n' + '='.repeat(60));
        console.log('🎯 PHASE 3: INTEGRATION & TESTING SUMMARY');
        console.log('='.repeat(60));
        console.log(`📊 Total Tests: ${summary.totalTests}`);
        console.log(`✅ Passed: ${summary.totalPassed} (${successRate}%)`);
        console.log(`❌ Failed: ${summary.totalFailed}`);
        console.log(`⏭️  Skipped: ${summary.totalSkipped}`);
        console.log(`⏱️  Duration: ${durationMinutes} minutes`);
        console.log(`📈 Coverage: ${summary.overallCoverage}%`);
        console.log('='.repeat(60));
        if (summary.totalFailed === 0) {
            console.log('🎉 ALL TESTS PASSED! Ready for Phase 4: Deployment');
        }
        else {
            console.log('❌ Some tests failed. Please review and fix before proceeding.');
        }
        console.log('='.repeat(60) + '\n');
    }
}
exports.Phase3TestRunner = Phase3TestRunner;
// Run tests if this file is executed directly
if (require.main === module) {
    const runner = new Phase3TestRunner();
    runner.runAllTests()
        .then(summary => {
        process.exit(summary.totalFailed === 0 ? 0 : 1);
    })
        .catch(error => {
        logger_1.logger.error('Phase 3 test runner failed:', error);
        process.exit(1);
    });
}
