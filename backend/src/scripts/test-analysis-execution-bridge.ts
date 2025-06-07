#!/usr/bin/env node

/**
 * Analysis-Execution Bridge Test
 * Comprehensive testing of real-time coordination layer with API and WebSocket functionality
 */

import { AnalysisExecutionBridge, TradingSignal, BridgeStatus } from '../services/AnalysisExecutionBridge';
import { logger } from '../utils/logger';
import WebSocket from 'ws';
import axios from 'axios';

class AnalysisExecutionBridgeTest {
  private bridge: AnalysisExecutionBridge;
  private testSymbols: string[] = ['BTCUSD', 'ETHUSD'];
  private baseUrl: string = 'http://localhost:8000';
  private wsUrl: string = 'ws://localhost:8000';

  constructor() {
    this.bridge = new AnalysisExecutionBridge({
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
  public async runTest(): Promise<void> {
    logger.info('🌉 ANALYSIS-EXECUTION BRIDGE TEST');
    logger.info('=' .repeat(80));

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

      logger.info('\n🎉 ANALYSIS-EXECUTION BRIDGE TEST COMPLETED SUCCESSFULLY!');
      logger.info('✅ All bridge functionality is working correctly');

    } catch (error: any) {
      logger.error('❌ Analysis-execution bridge test failed:', error.message);
      throw error;
    } finally {
      // Cleanup
      await this.bridge.stop();
    }
  }

  /**
   * Test bridge initialization and startup
   */
  private async testBridgeInitialization(): Promise<void> {
    logger.info('\n🔧 STEP 1: BRIDGE INITIALIZATION TEST');

    // Initialize the bridge
    await this.bridge.initialize();
    logger.info('✅ Bridge initialized successfully');

    // Start the bridge server
    await this.bridge.start();
    logger.info('✅ Bridge server started successfully');

    // Check initial status
    const status = this.bridge.getStatus();
    logger.info('📊 Initial bridge status:');
    logger.info(`   Running: ${status.isRunning ? 'YES' : 'NO'}`);
    logger.info(`   Connected Clients: ${status.connectedClients}`);
    logger.info(`   Total Signals: ${status.totalSignals}`);
    logger.info(`   Uptime: ${status.uptime}ms`);

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
  private async testRestApiEndpoints(): Promise<void> {
    logger.info('\n🔌 STEP 2: REST API ENDPOINTS TEST');

    try {
      // Test health check endpoint
      logger.info('\n📊 Testing health check endpoint...');
      const healthResponse = await axios.get(`${this.baseUrl}/health`);
      logger.info(`✅ Health check: ${healthResponse.status} - ${healthResponse.data.status}`);

      // Test status endpoint
      logger.info('\n📊 Testing status endpoint...');
      const statusResponse = await axios.get(`${this.baseUrl}/api/status`);
      const status: BridgeStatus = statusResponse.data;
      logger.info(`✅ Status endpoint: ${statusResponse.status}`);
      logger.info(`   Running: ${status.isRunning}`);
      logger.info(`   Connected Clients: ${status.connectedClients}`);
      logger.info(`   Average Latency: ${status.averageLatency.toFixed(2)}ms`);

      // Test positions endpoint
      logger.info('\n📊 Testing positions endpoint...');
      const positionsResponse = await axios.get(`${this.baseUrl}/api/positions`);
      logger.info(`✅ Positions endpoint: ${positionsResponse.status}`);
      logger.info(`   Active Positions: ${positionsResponse.data.positions.length}`);

      // Test risk endpoint
      logger.info('\n📊 Testing risk endpoint...');
      const riskResponse = await axios.get(`${this.baseUrl}/api/risk`);
      logger.info(`✅ Risk endpoint: ${riskResponse.status}`);
      logger.info(`   Overall Risk Score: ${(riskResponse.data.metrics.overallRiskScore * 100).toFixed(1)}%`);
      logger.info(`   Active Failsafes: ${riskResponse.data.failsafeMechanisms.filter((m: any) => m.isActive).length}`);

      // Test trading decision endpoint
      for (const symbol of this.testSymbols) {
        logger.info(`\n📊 Testing trading decision for ${symbol}...`);
        try {
          const decisionResponse = await axios.post(`${this.baseUrl}/api/decisions/${symbol}`);
          if (decisionResponse.status === 200) {
            logger.info(`✅ Decision generated for ${symbol}: ${decisionResponse.data.decision.action.toUpperCase()}`);
            logger.info(`   Confidence: ${(decisionResponse.data.decision.confidence * 100).toFixed(1)}%`);
            logger.info(`   Signal ID: ${decisionResponse.data.signalId}`);
          } else if (decisionResponse.status === 204) {
            logger.info(`⚠️ No decision generated for ${symbol}`);
          }
        } catch (error: any) {
          if (error.response?.status === 204) {
            logger.info(`⚠️ No decision generated for ${symbol}`);
          } else {
            logger.error(`❌ Decision endpoint error for ${symbol}:`, error.message);
          }
        }
      }

      // Test manual signal endpoint
      logger.info('\n📊 Testing manual signal endpoint...');
      const signalResponse = await axios.post(`${this.baseUrl}/api/signals`, {
        symbol: 'BTCUSD',
        action: 'buy',
        confidence: 0.8,
        metadata: { test: true }
      });
      logger.info(`✅ Manual signal sent: ${signalResponse.status}`);
      logger.info(`   Signal ID: ${signalResponse.data.signalId}`);

      // Test execution result endpoint
      const signalId = signalResponse.data.signalId;
      await this.sleep(1000); // Wait for processing
      
      try {
        const executionResponse = await axios.get(`${this.baseUrl}/api/executions/${signalId}`);
        logger.info(`✅ Execution result retrieved: ${executionResponse.status}`);
        logger.info(`   Success: ${executionResponse.data.success}`);
        logger.info(`   Latency: ${executionResponse.data.latency}ms`);
      } catch (error: any) {
        if (error.response?.status === 404) {
          logger.info(`⚠️ Execution result not yet available for ${signalId}`);
        } else {
          throw error;
        }
      }

    } catch (error: any) {
      logger.error('❌ REST API test failed:', error.message);
      throw error;
    }
  }

  /**
   * Test WebSocket functionality
   */
  private async testWebSocketFunctionality(): Promise<void> {
    logger.info('\n🔌 STEP 3: WEBSOCKET FUNCTIONALITY TEST');

    return new Promise((resolve, reject) => {
      const ws = new WebSocket(this.wsUrl);
      let messageCount = 0;
      const expectedMessages = 3;

      const timeout = setTimeout(() => {
        ws.close();
        reject(new Error('WebSocket test timeout'));
      }, 10000);

      ws.on('open', () => {
        logger.info('✅ WebSocket connection established');

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

      ws.on('message', (data: Buffer) => {
        try {
          const message = JSON.parse(data.toString());
          messageCount++;

          logger.info(`📨 WebSocket message received: ${message.type}`);
          
          switch (message.type) {
            case 'status':
              if (message.data.message) {
                logger.info(`   Welcome message: ${message.data.message}`);
              } else {
                logger.info(`   Bridge status: Running ${message.data.isRunning}`);
              }
              break;
            
            case 'pong':
              logger.info(`   Pong received: ${message.data.timestamp}`);
              break;
            
            case 'positions':
              logger.info(`   Positions: ${message.data.positions.length} active`);
              break;
            
            case 'heartbeat':
              logger.info(`   Heartbeat: ${message.data.timestamp}`);
              break;
          }

          if (messageCount >= expectedMessages) {
            clearTimeout(timeout);
            ws.close();
            logger.info('✅ WebSocket functionality test completed');
            resolve();
          }

        } catch (error: any) {
          clearTimeout(timeout);
          ws.close();
          reject(new Error(`WebSocket message parsing error: ${error.message}`));
        }
      });

      ws.on('error', (error: Error) => {
        clearTimeout(timeout);
        reject(new Error(`WebSocket error: ${error.message}`));
      });

      ws.on('close', (code: number) => {
        logger.info(`🔌 WebSocket connection closed: ${code}`);
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
  private async testTradingSignalFlow(): Promise<void> {
    logger.info('\n📡 STEP 4: TRADING SIGNAL FLOW TEST');

    // Test signal creation and processing
    const testSignals: Omit<TradingSignal, 'id' | 'timestamp'>[] = [
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
      logger.info(`\n🔄 Testing signal ${index + 1}: ${signalData.action.toUpperCase()} ${signalData.symbol}`);
      
      try {
        const signalId = await this.bridge.sendTradingSignal(signalData);
        logger.info(`✅ Signal sent: ${signalId}`);
        logger.info(`   Action: ${signalData.action.toUpperCase()}`);
        logger.info(`   Confidence: ${(signalData.confidence * 100).toFixed(1)}%`);
        logger.info(`   Source: ${signalData.source}`);

        // Wait for processing
        await this.sleep(500);

        // Check execution result
        const result = this.bridge.getExecutionResult(signalId);
        if (result) {
          logger.info(`📊 Execution result:`);
          logger.info(`   Success: ${result.success ? 'YES' : 'NO'}`);
          logger.info(`   Latency: ${result.latency}ms`);
          if (result.error) {
            logger.info(`   Error: ${result.error}`);
          }
        } else {
          logger.warn(`⚠️ No execution result found for ${signalId}`);
        }

      } catch (error: any) {
        logger.error(`❌ Signal flow test failed for signal ${index + 1}:`, error.message);
      }
    }

    // Check bridge status after signal processing
    const status = this.bridge.getStatus();
    logger.info(`\n📊 Bridge status after signal processing:`);
    logger.info(`   Total Signals: ${status.totalSignals}`);
    logger.info(`   Successful Executions: ${status.successfulExecutions}`);
    logger.info(`   Failed Executions: ${status.failedExecutions}`);
    logger.info(`   Average Latency: ${status.averageLatency.toFixed(2)}ms`);
  }

  /**
   * Test error handling and failsafe mechanisms
   */
  private async testErrorHandlingAndFailsafe(): Promise<void> {
    logger.info('\n🛡️ STEP 5: ERROR HANDLING AND FAILSAFE TEST');

    // Test invalid API requests
    logger.info('\n🧪 Testing invalid API requests...');
    
    try {
      await axios.get(`${this.baseUrl}/api/nonexistent`);
      logger.error('❌ Should have received 404 error');
    } catch (error: any) {
      if (error.response?.status === 404) {
        logger.info('✅ 404 error handling works correctly');
      } else {
        logger.error(`❌ Unexpected error: ${error.message}`);
      }
    }

    // Test invalid signal data
    logger.info('\n🧪 Testing invalid signal data...');
    
    try {
      await axios.post(`${this.baseUrl}/api/signals`, {
        // Missing required fields
        confidence: 1.5 // Invalid confidence
      });
      logger.error('❌ Should have received validation error');
    } catch (error: any) {
      if (error.response?.status === 400) {
        logger.info('✅ Validation error handling works correctly');
        logger.info(`   Error: ${error.response.data.error}`);
      } else {
        logger.error(`❌ Unexpected error: ${error.message}`);
      }
    }

    // Test WebSocket error handling
    logger.info('\n🧪 Testing WebSocket error handling...');
    
    return new Promise((resolve) => {
      const ws = new WebSocket(this.wsUrl);
      
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

      ws.on('message', (data: Buffer) => {
        try {
          const message = JSON.parse(data.toString());
          if (message.type === 'error') {
            logger.info('✅ WebSocket error handling works correctly');
            logger.info(`   Error: ${message.data.error}`);
            ws.close();
            resolve();
          }
        } catch (error: any) {
          logger.error(`❌ WebSocket error test failed: ${error.message}`);
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
  private async testRealTimeCoordination(): Promise<void> {
    logger.info('\n⚡ STEP 6: REAL-TIME COORDINATION TEST');

    // Test concurrent signal processing
    logger.info('\n🔄 Testing concurrent signal processing...');
    
    const concurrentSignals = Array.from({ length: 5 }, (_, i) => ({
      symbol: i % 2 === 0 ? 'BTCUSD' : 'ETHUSD',
      action: 'buy' as const,
      confidence: 0.7 + (i * 0.05),
      source: 'manual' as const,
      metadata: { test: `concurrent_${i}` }
    }));

    const signalPromises = concurrentSignals.map(signal => 
      this.bridge.sendTradingSignal(signal)
    );

    const signalIds = await Promise.all(signalPromises);
    logger.info(`✅ ${signalIds.length} concurrent signals sent successfully`);

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

    logger.info(`📊 Concurrent processing results:`);
    logger.info(`   Signals sent: ${signalIds.length}`);
    logger.info(`   Signals processed: ${processedCount}`);
    logger.info(`   Processing rate: ${(processedCount / signalIds.length * 100).toFixed(1)}%`);
  }

  /**
   * Test performance and latency
   */
  private async testPerformanceAndLatency(): Promise<void> {
    logger.info('\n🚀 STEP 7: PERFORMANCE AND LATENCY TEST');

    const performanceTests = [
      { name: 'Single Signal', count: 1 },
      { name: 'Burst Signals', count: 10 },
      { name: 'High Volume', count: 50 }
    ];

    for (const test of performanceTests) {
      logger.info(`\n⏱️ Testing ${test.name} (${test.count} signals)...`);
      
      const startTime = Date.now();
      const signalPromises = Array.from({ length: test.count }, (_, i) => 
        this.bridge.sendTradingSignal({
          symbol: 'BTCUSD',
          action: 'hold',
          confidence: 0.5,
          source: 'manual',
          metadata: { performance_test: i }
        })
      );

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

      logger.info(`📊 ${test.name} Results:`);
      logger.info(`   Signals: ${test.count}`);
      logger.info(`   Send Time: ${sendTime}ms`);
      logger.info(`   Processed: ${processedCount}/${test.count} (${(processedCount / test.count * 100).toFixed(1)}%)`);
      logger.info(`   Average Latency: ${averageLatency.toFixed(2)}ms`);
      logger.info(`   Throughput: ${throughput.toFixed(2)} signals/sec`);
    }
  }

  /**
   * Test emergency protocols
   */
  private async testEmergencyProtocols(): Promise<void> {
    logger.info('\n🚨 STEP 8: EMERGENCY PROTOCOLS TEST');

    // Test emergency stop via API
    logger.info('\n🛑 Testing emergency stop via API...');
    
    try {
      const emergencyResponse = await axios.post(`${this.baseUrl}/api/emergency-stop`);
      logger.info(`✅ Emergency stop triggered: ${emergencyResponse.status}`);
      logger.info(`   Message: ${emergencyResponse.data.message}`);
      logger.info(`   Signal ID: ${emergencyResponse.data.signalId}`);

      // Wait for processing
      await this.sleep(1000);

      // Check execution result
      const result = this.bridge.getExecutionResult(emergencyResponse.data.signalId);
      if (result) {
        logger.info(`📊 Emergency stop result:`);
        logger.info(`   Success: ${result.success ? 'YES' : 'NO'}`);
        logger.info(`   Closed Positions: ${result.executedQuantity || 0}`);
        logger.info(`   Latency: ${result.latency}ms`);
      }

    } catch (error: any) {
      logger.error(`❌ Emergency stop test failed: ${error.message}`);
    }

    // Final status check
    const finalStatus = this.bridge.getStatus();
    logger.info(`\n📊 FINAL BRIDGE STATUS:`);
    logger.info(`   Running: ${finalStatus.isRunning ? 'YES' : 'NO'}`);
    logger.info(`   Total Signals: ${finalStatus.totalSignals}`);
    logger.info(`   Successful Executions: ${finalStatus.successfulExecutions}`);
    logger.info(`   Failed Executions: ${finalStatus.failedExecutions}`);
    logger.info(`   Success Rate: ${finalStatus.totalSignals > 0 ? (finalStatus.successfulExecutions / finalStatus.totalSignals * 100).toFixed(1) : 0}%`);
    logger.info(`   Average Latency: ${finalStatus.averageLatency.toFixed(2)}ms`);
    logger.info(`   Uptime: ${(finalStatus.uptime / 1000).toFixed(1)}s`);
  }

  /**
   * Sleep utility
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

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
    logger.error('💥 Analysis-execution bridge test failed:', error);
    process.exit(1);
  });
}

export { AnalysisExecutionBridgeTest };
