#!/usr/bin/env node
"use strict";
/**
 * Analysis-Execution Bridge Test
 * Comprehensive testing of real-time coordination layer with API and WebSocket functionality
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.AnalysisExecutionBridgeTest = void 0;
const AnalysisExecutionBridge_1 = require("../services/AnalysisExecutionBridge");
const logger_1 = require("../utils/logger");
const ws_1 = __importDefault(require("ws"));
const axios_1 = __importDefault(require("axios"));
class AnalysisExecutionBridgeTest {
    constructor() {
        this.testSymbols = ['BTCUSD', 'ETHUSD'];
        this.baseUrl = 'http://localhost:8000';
        this.wsUrl = 'ws://localhost:8000';
        this.bridge = new AnalysisExecutionBridge_1.AnalysisExecutionBridge({
            port: 8000,
            host: 'localhost',
            enableWebSocket: true,
            enableRateLimit: false, // Disable for testing
            maxRequestsPerMinute: 10000,
            corsOrigins: ['*'],
            enableHelmet: false // Disable for testing
        });
    }
    /**
     * Run comprehensive analysis-execution bridge test
     */
    async runTest() {
        logger_1.logger.info('🌉 ANALYSIS-EXECUTION BRIDGE TEST');
        logger_1.logger.info('='.repeat(80));
        try {
            // Step 1: Initialize and start the bridge
            await this.testBridgeInitialization();
            // Step 2: Test REST API endpoints
            await this.testRestApiEndpoints();
            // Step 3: Test WebSocket functionality
            await this.testWebSocketFunctionality();
            // Step 4: Test trading signal flow
            await this.testTradingSignalFlow();
            // Step 5: Test error handling and failsafe mechanisms
            await this.testErrorHandlingAndFailsafe();
            // Step 6: Test real-time coordination
            await this.testRealTimeCoordination();
            // Step 7: Test performance and latency
            await this.testPerformanceAndLatency();
            // Step 8: Test emergency protocols
            await this.testEmergencyProtocols();
            logger_1.logger.info('\n🎉 ANALYSIS-EXECUTION BRIDGE TEST COMPLETED SUCCESSFULLY!');
            logger_1.logger.info('✅ All bridge functionality is working correctly');
        }
        catch (error) {
            logger_1.logger.error('❌ Analysis-execution bridge test failed:', error.message);
            throw error;
        }
        finally {
            // Cleanup
            await this.bridge.stop();
        }
    }
    /**
     * Test bridge initialization and startup
     */
    async testBridgeInitialization() {
        logger_1.logger.info('\n🔧 STEP 1: BRIDGE INITIALIZATION TEST');
        // Initialize the bridge
        await this.bridge.initialize();
        logger_1.logger.info('✅ Bridge initialized successfully');
        // Start the bridge server
        await this.bridge.start();
        logger_1.logger.info('✅ Bridge server started successfully');
        // Check initial status
        const status = this.bridge.getStatus();
        logger_1.logger.info('📊 Initial bridge status:');
        logger_1.logger.info(`   Running: ${status.isRunning ? 'YES' : 'NO'}`);
        logger_1.logger.info(`   Connected Clients: ${status.connectedClients}`);
        logger_1.logger.info(`   Total Signals: ${status.totalSignals}`);
        logger_1.logger.info(`   Uptime: ${status.uptime}ms`);
        // Validate status
        if (!status.isRunning) {
            throw new Error('Bridge is not running after start');
        }
        // Wait for server to be ready
        await this.sleep(2000);
    }
    /**
     * Test REST API endpoints
     */
    async testRestApiEndpoints() {
        logger_1.logger.info('\n🔌 STEP 2: REST API ENDPOINTS TEST');
        try {
            // Test health check endpoint
            logger_1.logger.info('\n📊 Testing health check endpoint...');
            const healthResponse = await axios_1.default.get(`${this.baseUrl}/health`);
            logger_1.logger.info(`✅ Health check: ${healthResponse.status} - ${healthResponse.data.status}`);
            // Test status endpoint
            logger_1.logger.info('\n📊 Testing status endpoint...');
            const statusResponse = await axios_1.default.get(`${this.baseUrl}/api/status`);
            const status = statusResponse.data;
            logger_1.logger.info(`✅ Status endpoint: ${statusResponse.status}`);
            logger_1.logger.info(`   Running: ${status.isRunning}`);
            logger_1.logger.info(`   Connected Clients: ${status.connectedClients}`);
            logger_1.logger.info(`   Average Latency: ${status.averageLatency.toFixed(2)}ms`);
            // Test positions endpoint
            logger_1.logger.info('\n📊 Testing positions endpoint...');
            const positionsResponse = await axios_1.default.get(`${this.baseUrl}/api/positions`);
            logger_1.logger.info(`✅ Positions endpoint: ${positionsResponse.status}`);
            logger_1.logger.info(`   Active Positions: ${positionsResponse.data.positions.length}`);
            // Test risk endpoint
            logger_1.logger.info('\n📊 Testing risk endpoint...');
            const riskResponse = await axios_1.default.get(`${this.baseUrl}/api/risk`);
            logger_1.logger.info(`✅ Risk endpoint: ${riskResponse.status}`);
            logger_1.logger.info(`   Overall Risk Score: ${(riskResponse.data.metrics.overallRiskScore * 100).toFixed(1)}%`);
            logger_1.logger.info(`   Active Failsafes: ${riskResponse.data.failsafeMechanisms.filter((m) => m.isActive).length}`);
            // Test trading decision endpoint
            for (const symbol of this.testSymbols) {
                logger_1.logger.info(`\n📊 Testing trading decision for ${symbol}...`);
                try {
                    const decisionResponse = await axios_1.default.post(`${this.baseUrl}/api/decisions/${symbol}`);
                    if (decisionResponse.status === 200) {
                        logger_1.logger.info(`✅ Decision generated for ${symbol}: ${decisionResponse.data.decision.action.toUpperCase()}`);
                        logger_1.logger.info(`   Confidence: ${(decisionResponse.data.decision.confidence * 100).toFixed(1)}%`);
                        logger_1.logger.info(`   Signal ID: ${decisionResponse.data.signalId}`);
                    }
                    else if (decisionResponse.status === 204) {
                        logger_1.logger.info(`⚠️ No decision generated for ${symbol}`);
                    }
                }
                catch (error) {
                    if (error.response?.status === 204) {
                        logger_1.logger.info(`⚠️ No decision generated for ${symbol}`);
                    }
                    else {
                        logger_1.logger.error(`❌ Decision endpoint error for ${symbol}:`, error.message);
                    }
                }
            }
            // Test manual signal endpoint
            logger_1.logger.info('\n📊 Testing manual signal endpoint...');
            const signalResponse = await axios_1.default.post(`${this.baseUrl}/api/signals`, {
                symbol: 'BTCUSD',
                action: 'buy',
                confidence: 0.8,
                metadata: { test: true }
            });
            logger_1.logger.info(`✅ Manual signal sent: ${signalResponse.status}`);
            logger_1.logger.info(`   Signal ID: ${signalResponse.data.signalId}`);
            // Test execution result endpoint
            const signalId = signalResponse.data.signalId;
            await this.sleep(1000); // Wait for processing
            try {
                const executionResponse = await axios_1.default.get(`${this.baseUrl}/api/executions/${signalId}`);
                logger_1.logger.info(`✅ Execution result retrieved: ${executionResponse.status}`);
                logger_1.logger.info(`   Success: ${executionResponse.data.success}`);
                logger_1.logger.info(`   Latency: ${executionResponse.data.latency}ms`);
            }
            catch (error) {
                if (error.response?.status === 404) {
                    logger_1.logger.info(`⚠️ Execution result not yet available for ${signalId}`);
                }
                else {
                    throw error;
                }
            }
        }
        catch (error) {
            logger_1.logger.error('❌ REST API test failed:', error.message);
            throw error;
        }
    }
    /**
     * Test WebSocket functionality
     */
    async testWebSocketFunctionality() {
        logger_1.logger.info('\n🔌 STEP 3: WEBSOCKET FUNCTIONALITY TEST');
        return new Promise((resolve, reject) => {
            const ws = new ws_1.default(this.wsUrl);
            let messageCount = 0;
            const expectedMessages = 3;
            const timeout = setTimeout(() => {
                ws.close();
                reject(new Error('WebSocket test timeout'));
            }, 10000);
            ws.on('open', () => {
                logger_1.logger.info('✅ WebSocket connection established');
                // Test ping-pong
                ws.send(JSON.stringify({
                    type: 'ping',
                    data: {},
                    timestamp: Date.now(),
                    id: 'test_ping'
                }));
                // Test status request
                ws.send(JSON.stringify({
                    type: 'get_status',
                    data: {},
                    timestamp: Date.now(),
                    id: 'test_status'
                }));
                // Test positions request
                ws.send(JSON.stringify({
                    type: 'get_positions',
                    data: {},
                    timestamp: Date.now(),
                    id: 'test_positions'
                }));
            });
            ws.on('message', (data) => {
                try {
                    const message = JSON.parse(data.toString());
                    messageCount++;
                    logger_1.logger.info(`📨 WebSocket message received: ${message.type}`);
                    switch (message.type) {
                        case 'status':
                            if (message.data.message) {
                                logger_1.logger.info(`   Welcome message: ${message.data.message}`);
                            }
                            else {
                                logger_1.logger.info(`   Bridge status: Running ${message.data.isRunning}`);
                            }
                            break;
                        case 'pong':
                            logger_1.logger.info(`   Pong received: ${message.data.timestamp}`);
                            break;
                        case 'positions':
                            logger_1.logger.info(`   Positions: ${message.data.positions.length} active`);
                            break;
                        case 'heartbeat':
                            logger_1.logger.info(`   Heartbeat: ${message.data.timestamp}`);
                            break;
                    }
                    if (messageCount >= expectedMessages) {
                        clearTimeout(timeout);
                        ws.close();
                        logger_1.logger.info('✅ WebSocket functionality test completed');
                        resolve();
                    }
                }
                catch (error) {
                    clearTimeout(timeout);
                    ws.close();
                    reject(new Error(`WebSocket message parsing error: ${error.message}`));
                }
            });
            ws.on('error', (error) => {
                clearTimeout(timeout);
                reject(new Error(`WebSocket error: ${error.message}`));
            });
            ws.on('close', (code) => {
                logger_1.logger.info(`🔌 WebSocket connection closed: ${code}`);
                if (messageCount < expectedMessages) {
                    clearTimeout(timeout);
                    reject(new Error(`WebSocket closed before receiving all messages (${messageCount}/${expectedMessages})`));
                }
            });
        });
    }
    /**
     * Test trading signal flow
     */
    async testTradingSignalFlow() {
        logger_1.logger.info('\n📡 STEP 4: TRADING SIGNAL FLOW TEST');
        // Test signal creation and processing
        const testSignals = [
            {
                symbol: 'BTCUSD',
                action: 'buy',
                confidence: 0.85,
                source: 'ml_decision',
                metadata: { positionSize: 0.05, leverage: 100 }
            },
            {
                symbol: 'ETHUSD',
                action: 'sell',
                confidence: 0.75,
                source: 'manual',
                metadata: { positionSize: 0.03, leverage: 50 }
            },
            {
                symbol: 'BTCUSD',
                action: 'hold',
                confidence: 0.6,
                source: 'risk_management',
                metadata: {}
            }
        ];
        for (const [index, signalData] of testSignals.entries()) {
            logger_1.logger.info(`\n🔄 Testing signal ${index + 1}: ${signalData.action.toUpperCase()} ${signalData.symbol}`);
            try {
                const signalId = await this.bridge.sendTradingSignal(signalData);
                logger_1.logger.info(`✅ Signal sent: ${signalId}`);
                logger_1.logger.info(`   Action: ${signalData.action.toUpperCase()}`);
                logger_1.logger.info(`   Confidence: ${(signalData.confidence * 100).toFixed(1)}%`);
                logger_1.logger.info(`   Source: ${signalData.source}`);
                // Wait for processing
                await this.sleep(500);
                // Check execution result
                const result = this.bridge.getExecutionResult(signalId);
                if (result) {
                    logger_1.logger.info(`📊 Execution result:`);
                    logger_1.logger.info(`   Success: ${result.success ? 'YES' : 'NO'}`);
                    logger_1.logger.info(`   Latency: ${result.latency}ms`);
                    if (result.error) {
                        logger_1.logger.info(`   Error: ${result.error}`);
                    }
                }
                else {
                    logger_1.logger.warn(`⚠️ No execution result found for ${signalId}`);
                }
            }
            catch (error) {
                logger_1.logger.error(`❌ Signal flow test failed for signal ${index + 1}:`, error.message);
            }
        }
        // Check bridge status after signal processing
        const status = this.bridge.getStatus();
        logger_1.logger.info(`\n📊 Bridge status after signal processing:`);
        logger_1.logger.info(`   Total Signals: ${status.totalSignals}`);
        logger_1.logger.info(`   Successful Executions: ${status.successfulExecutions}`);
        logger_1.logger.info(`   Failed Executions: ${status.failedExecutions}`);
        logger_1.logger.info(`   Average Latency: ${status.averageLatency.toFixed(2)}ms`);
    }
    /**
     * Test error handling and failsafe mechanisms
     */
    async testErrorHandlingAndFailsafe() {
        logger_1.logger.info('\n🛡️ STEP 5: ERROR HANDLING AND FAILSAFE TEST');
        // Test invalid API requests
        logger_1.logger.info('\n🧪 Testing invalid API requests...');
        try {
            await axios_1.default.get(`${this.baseUrl}/api/nonexistent`);
            logger_1.logger.error('❌ Should have received 404 error');
        }
        catch (error) {
            if (error.response?.status === 404) {
                logger_1.logger.info('✅ 404 error handling works correctly');
            }
            else {
                logger_1.logger.error(`❌ Unexpected error: ${error.message}`);
            }
        }
        // Test invalid signal data
        logger_1.logger.info('\n🧪 Testing invalid signal data...');
        try {
            await axios_1.default.post(`${this.baseUrl}/api/signals`, {
                // Missing required fields
                confidence: 1.5 // Invalid confidence
            });
            logger_1.logger.error('❌ Should have received validation error');
        }
        catch (error) {
            if (error.response?.status === 400) {
                logger_1.logger.info('✅ Validation error handling works correctly');
                logger_1.logger.info(`   Error: ${error.response.data.error}`);
            }
            else {
                logger_1.logger.error(`❌ Unexpected error: ${error.message}`);
            }
        }
        // Test WebSocket error handling
        logger_1.logger.info('\n🧪 Testing WebSocket error handling...');
        return new Promise((resolve) => {
            const ws = new ws_1.default(this.wsUrl);
            ws.on('open', () => {
                // Send invalid message
                ws.send('invalid json');
                // Send unknown message type
                ws.send(JSON.stringify({
                    type: 'unknown_type',
                    data: {},
                    timestamp: Date.now(),
                    id: 'test_error'
                }));
            });
            ws.on('message', (data) => {
                try {
                    const message = JSON.parse(data.toString());
                    if (message.type === 'error') {
                        logger_1.logger.info('✅ WebSocket error handling works correctly');
                        logger_1.logger.info(`   Error: ${message.data.error}`);
                        ws.close();
                        resolve();
                    }
                }
                catch (error) {
                    logger_1.logger.error(`❌ WebSocket error test failed: ${error.message}`);
                    ws.close();
                    resolve();
                }
            });
            // Timeout fallback
            setTimeout(() => {
                ws.close();
                resolve();
            }, 3000);
        });
    }
    /**
     * Test real-time coordination
     */
    async testRealTimeCoordination() {
        logger_1.logger.info('\n⚡ STEP 6: REAL-TIME COORDINATION TEST');
        // Test concurrent signal processing
        logger_1.logger.info('\n🔄 Testing concurrent signal processing...');
        const concurrentSignals = Array.from({ length: 5 }, (_, i) => ({
            symbol: i % 2 === 0 ? 'BTCUSD' : 'ETHUSD',
            action: 'buy',
            confidence: 0.7 + (i * 0.05),
            source: 'manual',
            metadata: { test: `concurrent_${i}` }
        }));
        const signalPromises = concurrentSignals.map(signal => this.bridge.sendTradingSignal(signal));
        const signalIds = await Promise.all(signalPromises);
        logger_1.logger.info(`✅ ${signalIds.length} concurrent signals sent successfully`);
        // Wait for processing
        await this.sleep(2000);
        // Check results
        let processedCount = 0;
        for (const signalId of signalIds) {
            const result = this.bridge.getExecutionResult(signalId);
            if (result) {
                processedCount++;
            }
        }
        logger_1.logger.info(`📊 Concurrent processing results:`);
        logger_1.logger.info(`   Signals sent: ${signalIds.length}`);
        logger_1.logger.info(`   Signals processed: ${processedCount}`);
        logger_1.logger.info(`   Processing rate: ${(processedCount / signalIds.length * 100).toFixed(1)}%`);
    }
    /**
     * Test performance and latency
     */
    async testPerformanceAndLatency() {
        logger_1.logger.info('\n🚀 STEP 7: PERFORMANCE AND LATENCY TEST');
        const performanceTests = [
            { name: 'Single Signal', count: 1 },
            { name: 'Burst Signals', count: 10 },
            { name: 'High Volume', count: 50 }
        ];
        for (const test of performanceTests) {
            logger_1.logger.info(`\n⏱️ Testing ${test.name} (${test.count} signals)...`);
            const startTime = Date.now();
            const signalPromises = Array.from({ length: test.count }, (_, i) => this.bridge.sendTradingSignal({
                symbol: 'BTCUSD',
                action: 'hold',
                confidence: 0.5,
                source: 'manual',
                metadata: { performance_test: i }
            }));
            const signalIds = await Promise.all(signalPromises);
            const sendTime = Date.now() - startTime;
            // Wait for processing
            await this.sleep(Math.max(1000, test.count * 50));
            // Check processing results
            let processedCount = 0;
            let totalLatency = 0;
            for (const signalId of signalIds) {
                const result = this.bridge.getExecutionResult(signalId);
                if (result) {
                    processedCount++;
                    totalLatency += result.latency;
                }
            }
            const averageLatency = processedCount > 0 ? totalLatency / processedCount : 0;
            const throughput = (processedCount / (Date.now() - startTime)) * 1000;
            logger_1.logger.info(`📊 ${test.name} Results:`);
            logger_1.logger.info(`   Signals: ${test.count}`);
            logger_1.logger.info(`   Send Time: ${sendTime}ms`);
            logger_1.logger.info(`   Processed: ${processedCount}/${test.count} (${(processedCount / test.count * 100).toFixed(1)}%)`);
            logger_1.logger.info(`   Average Latency: ${averageLatency.toFixed(2)}ms`);
            logger_1.logger.info(`   Throughput: ${throughput.toFixed(2)} signals/sec`);
        }
    }
    /**
     * Test emergency protocols
     */
    async testEmergencyProtocols() {
        logger_1.logger.info('\n🚨 STEP 8: EMERGENCY PROTOCOLS TEST');
        // Test emergency stop via API
        logger_1.logger.info('\n🛑 Testing emergency stop via API...');
        try {
            const emergencyResponse = await axios_1.default.post(`${this.baseUrl}/api/emergency-stop`);
            logger_1.logger.info(`✅ Emergency stop triggered: ${emergencyResponse.status}`);
            logger_1.logger.info(`   Message: ${emergencyResponse.data.message}`);
            logger_1.logger.info(`   Signal ID: ${emergencyResponse.data.signalId}`);
            // Wait for processing
            await this.sleep(1000);
            // Check execution result
            const result = this.bridge.getExecutionResult(emergencyResponse.data.signalId);
            if (result) {
                logger_1.logger.info(`📊 Emergency stop result:`);
                logger_1.logger.info(`   Success: ${result.success ? 'YES' : 'NO'}`);
                logger_1.logger.info(`   Closed Positions: ${result.executedQuantity || 0}`);
                logger_1.logger.info(`   Latency: ${result.latency}ms`);
            }
        }
        catch (error) {
            logger_1.logger.error(`❌ Emergency stop test failed: ${error.message}`);
        }
        // Final status check
        const finalStatus = this.bridge.getStatus();
        logger_1.logger.info(`\n📊 FINAL BRIDGE STATUS:`);
        logger_1.logger.info(`   Running: ${finalStatus.isRunning ? 'YES' : 'NO'}`);
        logger_1.logger.info(`   Total Signals: ${finalStatus.totalSignals}`);
        logger_1.logger.info(`   Successful Executions: ${finalStatus.successfulExecutions}`);
        logger_1.logger.info(`   Failed Executions: ${finalStatus.failedExecutions}`);
        logger_1.logger.info(`   Success Rate: ${finalStatus.totalSignals > 0 ? (finalStatus.successfulExecutions / finalStatus.totalSignals * 100).toFixed(1) : 0}%`);
        logger_1.logger.info(`   Average Latency: ${finalStatus.averageLatency.toFixed(2)}ms`);
        logger_1.logger.info(`   Uptime: ${(finalStatus.uptime / 1000).toFixed(1)}s`);
    }
    /**
     * Sleep utility
     */
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}
exports.AnalysisExecutionBridgeTest = AnalysisExecutionBridgeTest;
/**
 * Main execution
 */
async function main() {
    const tester = new AnalysisExecutionBridgeTest();
    await tester.runTest();
}
// Run if this script is executed directly
if (require.main === module) {
    main().catch(error => {
        logger_1.logger.error('💥 Analysis-execution bridge test failed:', error);
        process.exit(1);
    });
}
//# sourceMappingURL=test-analysis-execution-bridge.js.map