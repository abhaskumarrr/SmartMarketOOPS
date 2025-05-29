/**
 * Test script for Delta Exchange API endpoints
 * 
 * This script tests the newly added Delta Exchange API trading endpoints:
 * - GET /api/delta/orders/history (Order history)
 * - DELETE /api/delta/orders (Cancel all orders)
 * - GET /api/delta/fills (Trade history)
 * 
 * To use:
 * 1. Start the backend server
 * 2. Run this script: node scripts/test-delta-api.js
 * 3. Check the output for success/failure of each endpoint
 */

const axios = require('axios');
require('dotenv').config();

// Constants
const API_URL = 'http://localhost:8000/api/delta';
const TOKEN = process.env.TEST_AUTH_TOKEN; // JWT token for authentication

// Set authorization header for all requests
axios.defaults.headers.common['Authorization'] = `Bearer ${TOKEN}`;

// Test function for order history endpoint
async function testOrderHistory() {
  console.log('\n--- Testing Order History Endpoint ---');
  try {
    const response = await axios.get(`${API_URL}/orders/history`);
    console.log('✅ Order History Success:', response.status);
    console.log('Data sample:', JSON.stringify(response.data).substring(0, 100) + '...');
    return true;
  } catch (error) {
    console.log('❌ Order History Error:', error.response?.status || error.message);
    if (error.response?.data) {
      console.log('Error details:', error.response.data);
    }
    return false;
  }
}

// Test function for cancel all orders endpoint
async function testCancelAllOrders() {
  console.log('\n--- Testing Cancel All Orders Endpoint ---');
  try {
    const response = await axios.delete(`${API_URL}/orders`);
    console.log('✅ Cancel All Orders Success:', response.status);
    console.log('Data sample:', JSON.stringify(response.data).substring(0, 100) + '...');
    return true;
  } catch (error) {
    console.log('❌ Cancel All Orders Error:', error.response?.status || error.message);
    if (error.response?.data) {
      console.log('Error details:', error.response.data);
    }
    return false;
  }
}

// Test function for trade history endpoint
async function testTradeHistory() {
  console.log('\n--- Testing Trade History Endpoint ---');
  try {
    const response = await axios.get(`${API_URL}/fills`);
    console.log('✅ Trade History Success:', response.status);
    console.log('Data sample:', JSON.stringify(response.data).substring(0, 100) + '...');
    return true;
  } catch (error) {
    console.log('❌ Trade History Error:', error.response?.status || error.message);
    if (error.response?.data) {
      console.log('Error details:', error.response.data);
    }
    return false;
  }
}

// Main test function
async function runTests() {
  console.log('=== Delta Exchange API Endpoint Tests ===');
  
  if (!TOKEN) {
    console.error('❌ No test auth token found. Set TEST_AUTH_TOKEN in .env file.');
    return;
  }
  
  // Run all tests
  const results = await Promise.all([
    testOrderHistory(),
    testCancelAllOrders(),
    testTradeHistory()
  ]);
  
  // Summary
  const totalTests = results.length;
  const passedTests = results.filter(result => result).length;
  
  console.log('\n=== Test Summary ===');
  console.log(`Passed: ${passedTests}/${totalTests}`);
  console.log(`Status: ${passedTests === totalTests ? '✅ All tests passed' : '❌ Some tests failed'}`);
}

// Run tests
runTests().catch(error => {
  console.error('Test execution error:', error);
}); 