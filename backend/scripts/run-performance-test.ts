/**
 * Performance Test Runner Script
 * Run performance tests from the command line
 * 
 * Usage:
 * npx ts-node scripts/run-performance-test.ts --type=API_LATENCY --endpoint=/api/health
 */

import { PrismaClient } from '@prisma/client';
import { v4 as uuidv4 } from 'uuid';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import axios from 'axios';
import * as fs from 'fs';
import * as path from 'path';
import { performance } from 'perf_hooks';
import { PerformanceTestType } from '../src/types/performance';

// Parse command line arguments
const argv = yargs(hideBin(process.argv))
  .option('type', {
    alias: 't',
    description: 'Type of performance test to run',
    type: 'string',
    choices: Object.values(PerformanceTestType),
    demandOption: true
  })
  .option('endpoint', {
    alias: 'e',
    description: 'Target endpoint to test (required for API tests)',
    type: 'string'
  })
  .option('duration', {
    alias: 'd',
    description: 'Test duration in seconds',
    type: 'number',
    default: 10
  })
  .option('concurrency', {
    alias: 'c',
    description: 'Number of concurrent requests',
    type: 'number',
    default: 10
  })
  .option('url', {
    description: 'Base URL for API tests',
    type: 'string',
    default: 'http://localhost:3001'
  })
  .option('outputFile', {
    alias: 'o',
    description: 'Output file for results (JSON)',
    type: 'string'
  })
  .check((argv) => {
    if ((argv.type === 'API_LATENCY' || argv.type === 'LOAD_TEST') && !argv.endpoint) {
      throw new Error('Endpoint is required for API_LATENCY and LOAD_TEST tests');
    }
    return true;
  })
  .help()
  .argv;

// Initialize Prisma client
const prisma = new PrismaClient();

/**
 * Main function
 */
async function main() {
  console.log(`\n📊 Running ${argv.type} performance test\n`);
  console.log(`Duration: ${argv.duration} seconds`);
  console.log(`Concurrency: ${argv.concurrency} requests\n`);
  
  let startTime = performance.now();
  let endTime: number;
  let results: any[] = [];
  
  // Run test based on type
  switch (argv.type) {
    case PerformanceTestType.API_LATENCY:
      results = await runApiLatencyTest(
        `${argv.url}${argv.endpoint}`,
        argv.duration,
        argv.concurrency
      );
      break;
    case PerformanceTestType.LOAD_TEST:
      results = await runLoadTest(
        `${argv.url}${argv.endpoint}`,
        argv.duration,
        argv.concurrency
      );
      break;
    case PerformanceTestType.ML_PREDICTION_THROUGHPUT:
      results = await runMlThroughputTest(
        argv.duration,
        argv.concurrency
      );
      break;
    default:
      console.error(`Test type ${argv.type} not implemented in CLI tool`);
      process.exit(1);
  }
  
  endTime = performance.now();
  
  // Calculate metrics
  const totalDuration = endTime - startTime;
  const successCount = results.filter(r => r.success).length;
  const errorCount = results.length - successCount;
  const successRate = (successCount / results.length) * 100;
  const errorRate = (errorCount / results.length) * 100;
  
  // Calculate latencies
  const latencies = results
    .filter(r => r.success)
    .map(r => r.duration);
  
  latencies.sort((a, b) => a - b);
  
  const latencyMin = latencies.length > 0 ? latencies[0] : 0;
  const latencyMax = latencies.length > 0 ? latencies[latencies.length - 1] : 0;
  const latencyAvg = latencies.length > 0
    ? latencies.reduce((sum, val) => sum + val, 0) / latencies.length
    : 0;
  
  // Calculate percentiles
  const p50Index = Math.floor(latencies.length * 0.5);
  const p90Index = Math.floor(latencies.length * 0.9);
  const p95Index = Math.floor(latencies.length * 0.95);
  const p99Index = Math.floor(latencies.length * 0.99);
  
  const latencyP50 = latencies.length > 0 ? latencies[p50Index] : 0;
  const latencyP90 = latencies.length > 0 ? latencies[p90Index] : 0;
  const latencyP95 = latencies.length > 0 ? latencies[p95Index] : 0;
  const latencyP99 = latencies.length > 0 ? latencies[p99Index] : 0;
  
  const requestsPerSecond = results.length / (totalDuration / 1000);
  
  // Print results
  console.log('\n📈 Test Results:\n');
  console.log(`Total Requests: ${results.length}`);
  console.log(`Successful Requests: ${successCount}`);
  console.log(`Failed Requests: ${errorCount}`);
  console.log(`Success Rate: ${successRate.toFixed(2)}%`);
  console.log(`Error Rate: ${errorRate.toFixed(2)}%`);
  console.log(`Requests/sec: ${requestsPerSecond.toFixed(2)}`);
  console.log('\nLatency (ms):');
  console.log(`  Min: ${latencyMin.toFixed(2)}`);
  console.log(`  Avg: ${latencyAvg.toFixed(2)}`);
  console.log(`  Max: ${latencyMax.toFixed(2)}`);
  console.log(`  P50: ${latencyP50.toFixed(2)}`);
  console.log(`  P90: ${latencyP90.toFixed(2)}`);
  console.log(`  P95: ${latencyP95.toFixed(2)}`);
  console.log(`  P99: ${latencyP99.toFixed(2)}`);
  
  // Create report object
  const report = {
    testType: argv.type,
    timestamp: new Date().toISOString(),
    duration: argv.duration,
    concurrency: argv.concurrency,
    endpoint: argv.endpoint,
    totalRequests: results.length,
    successfulRequests: successCount,
    failedRequests: errorCount,
    successRate,
    errorRate,
    requestsPerSecond,
    latency: {
      min: latencyMin,
      avg: latencyAvg,
      max: latencyMax,
      p50: latencyP50,
      p90: latencyP90,
      p95: latencyP95,
      p99: latencyP99
    }
  };
  
  // Save to file if specified
  if (argv.outputFile) {
    const outputPath = path.resolve(argv.outputFile);
    fs.writeFileSync(outputPath, JSON.stringify(report, null, 2));
    console.log(`\nReport saved to: ${outputPath}`);
  }
  
  // Save to database
  try {
    // Create test record
    const test = await prisma.performanceTest.create({
      data: {
        name: `CLI ${argv.type} Test`,
        testType: argv.type,
        duration: argv.duration,
        concurrency: argv.concurrency,
        targetEndpoint: argv.endpoint,
        options: {
          url: argv.url,
          outputFile: argv.outputFile
        } as any
      }
    });
    
    // Create test result
    await prisma.performanceTestResult.create({
      data: {
        testId: test.id,
        status: 'COMPLETED',
        startTime: new Date(Date.now() - totalDuration),
        endTime: new Date(),
        duration: totalDuration,
        metrics: {
          throughput: requestsPerSecond,
          latencyMin,
          latencyAvg,
          latencyMax,
          latencyP50,
          latencyP90,
          latencyP95,
          latencyP99,
          errorRate,
          successRate
        } as any
      }
    });
    
    console.log(`\nTest results saved to database with ID: ${test.id}`);
  } catch (error) {
    console.error('\nFailed to save results to database:', error);
  }
}

/**
 * Run API latency test
 * @param endpoint API endpoint to test
 * @param duration Test duration in seconds
 * @param concurrency Number of concurrent requests
 * @returns Test results
 */
async function runApiLatencyTest(
  endpoint: string,
  duration: number,
  concurrency: number
): Promise<Array<{ success: boolean; duration: number; error?: string }>> {
  console.log(`Testing API endpoint: ${endpoint}`);
  
  const results: Array<{ success: boolean; duration: number; error?: string }> = [];
  const endTime = Date.now() + (duration * 1000);
  
  // Create axios instance
  const api = axios.create({
    timeout: 10000, // 10 seconds
    validateStatus: () => true // Don't throw on any status code
  });
  
  // Run test until duration is up
  while (Date.now() < endTime) {
    // Create batch of concurrent requests
    const requests = Array(concurrency).fill(0).map(async () => {
      const start = performance.now();
      try {
        const response = await api.get(endpoint);
        const end = performance.now();
        
        results.push({
          success: response.status >= 200 && response.status < 300,
          duration: end - start,
          error: response.status >= 400 ? `HTTP ${response.status}` : undefined
        });
      } catch (error) {
        const end = performance.now();
        
        results.push({
          success: false,
          duration: end - start,
          error: error instanceof Error ? error.message : String(error)
        });
      }
    });
    
    // Wait for all requests to complete
    await Promise.all(requests);
    
    // Small delay to avoid overwhelming the server
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  
  return results;
}

/**
 * Run load test
 * @param endpoint API endpoint to test
 * @param duration Test duration in seconds
 * @param concurrency Number of concurrent requests
 * @returns Test results
 */
async function runLoadTest(
  endpoint: string,
  duration: number,
  concurrency: number
): Promise<Array<{ success: boolean; duration: number; error?: string }>> {
  // For the CLI tool, load test is similar to API latency test
  return runApiLatencyTest(endpoint, duration, concurrency);
}

/**
 * Run ML throughput test
 * @param duration Test duration in seconds
 * @param concurrency Number of concurrent predictions
 * @returns Test results
 */
async function runMlThroughputTest(
  duration: number,
  concurrency: number
): Promise<Array<{ success: boolean; duration: number; error?: string }>> {
  console.log('Testing ML prediction throughput');
  
  const results: Array<{ success: boolean; duration: number; error?: string }> = [];
  const endTime = Date.now() + (duration * 1000);
  
  // Create axios instance
  const api = axios.create({
    timeout: 30000, // 30 seconds
    validateStatus: () => true // Don't throw on any status code
  });
  
  // Run test until duration is up
  while (Date.now() < endTime) {
    // Create batch of concurrent requests
    const requests = Array(concurrency).fill(0).map(async () => {
      const start = performance.now();
      try {
        const response = await api.post('http://localhost:3001/api/bridge/predict', {
          symbol: 'BTCUSD',
          timeframe: '1h',
          limit: 100
        });
        
        const end = performance.now();
        
        results.push({
          success: response.status >= 200 && response.status < 300,
          duration: end - start,
          error: response.status >= 400 ? `HTTP ${response.status}` : undefined
        });
      } catch (error) {
        const end = performance.now();
        
        results.push({
          success: false,
          duration: end - start,
          error: error instanceof Error ? error.message : String(error)
        });
      }
    });
    
    // Wait for all requests to complete
    await Promise.all(requests);
    
    // Small delay to avoid overwhelming the server
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  
  return results;
}

// Run main function
main()
  .catch(e => {
    console.error('Error running performance test:', e);
    process.exit(1);
  })
  .finally(async () => {
    // Disconnect Prisma client
    await prisma.$disconnect();
  }); 