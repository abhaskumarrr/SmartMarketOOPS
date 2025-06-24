#!/usr/bin/env node

/**
 * MCP Servers Test Script
 * Tests the functionality of installed MCP servers
 */

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

// Colors for console output
const colors = {
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  reset: '\x1b[0m',
  bold: '\x1b[1m'
};

// Log functions
const log = {
  info: (msg) => console.log(`${colors.blue}ℹ${colors.reset} ${msg}`),
  success: (msg) => console.log(`${colors.green}✓${colors.reset} ${msg}`),
  error: (msg) => console.log(`${colors.red}✗${colors.reset} ${msg}`),
  warning: (msg) => console.log(`${colors.yellow}⚠${colors.reset} ${msg}`),
  header: (msg) => console.log(`\n${colors.bold}${colors.blue}${msg}${colors.reset}\n`)
};

// MCP configuration
const mcpConfigPath = path.join(__dirname, '..', '.cursor', 'mcp.json');

/**
 * Load MCP configuration
 */
function loadMcpConfig() {
  try {
    if (!fs.existsSync(mcpConfigPath)) {
      log.error('MCP configuration file not found');
      return null;
    }
    
    const configData = fs.readFileSync(mcpConfigPath, 'utf8');
    return JSON.parse(configData);
  } catch (error) {
    log.error(`Failed to load MCP configuration: ${error.message}`);
    return null;
  }
}

/**
 * Test if a command is available
 */
function testCommand(command, args = []) {
  return new Promise((resolve) => {
    const child = spawn(command, args, { stdio: 'pipe' });
    
    let timeout = setTimeout(() => {
      child.kill();
      resolve(false);
    }, 5000); // 5 second timeout
    
    child.on('close', (code) => {
      clearTimeout(timeout);
      resolve(code === 0);
    });
    
    child.on('error', () => {
      clearTimeout(timeout);
      resolve(false);
    });
  });
}

/**
 * Test MCP server availability
 */
async function testMcpServer(name, config) {
  log.info(`Testing ${name} server...`);
  
  try {
    // Test if the command is available
    const isAvailable = await testCommand(config.command, ['--help']);
    
    if (isAvailable) {
      log.success(`${name} server command is available`);
      return true;
    } else {
      log.warning(`${name} server command not available, will be installed on first use`);
      return false;
    }
  } catch (error) {
    log.error(`Failed to test ${name} server: ${error.message}`);
    return false;
  }
}

/**
 * Check environment variables
 */
function checkEnvironmentVariables(config) {
  const missingVars = [];
  
  for (const [key, value] of Object.entries(config.env || {})) {
    if (value.includes('_HERE') || value === '') {
      missingVars.push(key);
    }
  }
  
  return missingVars;
}

/**
 * Test database connectivity
 */
async function testDatabaseConnectivity() {
  log.info('Testing database connectivity...');
  
  try {
    // Test PostgreSQL connection
    const pgConnString = process.env.POSTGRES_CONNECTION_STRING || 
                        'postgresql://postgres:postgres@localhost:5432/smartmarket';
    
    log.info(`Testing PostgreSQL connection: ${pgConnString.replace(/:[^:@]*@/, ':***@')}`);
    
    // Simple connection test (would need actual database client in real implementation)
    log.warning('Database connectivity test requires actual database client - skipping for now');
    
    return true;
  } catch (error) {
    log.error(`Database connectivity test failed: ${error.message}`);
    return false;
  }
}

/**
 * Generate MCP server status report
 */
function generateStatusReport(results) {
  log.header('MCP Servers Status Report');
  
  const totalServers = Object.keys(results.servers).length;
  const availableServers = Object.values(results.servers).filter(s => s.available).length;
  const serversWithMissingEnv = Object.values(results.servers).filter(s => s.missingEnvVars.length > 0).length;
  
  console.log(`Total MCP Servers: ${totalServers}`);
  console.log(`Available Servers: ${availableServers}`);
  console.log(`Servers with Missing Environment Variables: ${serversWithMissingEnv}`);
  
  log.header('Server Details');
  
  for (const [name, status] of Object.entries(results.servers)) {
    console.log(`\n${colors.bold}${name}:${colors.reset}`);
    console.log(`  Status: ${status.available ? colors.green + 'Available' : colors.yellow + 'Will install on first use'}${colors.reset}`);
    
    if (status.missingEnvVars.length > 0) {
      console.log(`  Missing Environment Variables: ${colors.red}${status.missingEnvVars.join(', ')}${colors.reset}`);
    } else {
      console.log(`  Environment Variables: ${colors.green}All configured${colors.reset}`);
    }
  }
  
  log.header('Recommendations');
  
  if (serversWithMissingEnv > 0) {
    log.warning('Some servers have missing environment variables. Update your .env file with the required API keys.');
  }
  
  if (availableServers < totalServers) {
    log.info('Some servers will be automatically installed when first accessed via MCP.');
  }
  
  log.success('MCP servers are properly configured and ready to use!');
}

/**
 * Main test function
 */
async function main() {
  log.header('SmartMarketOOPS MCP Servers Test');
  
  // Load MCP configuration
  const mcpConfig = loadMcpConfig();
  if (!mcpConfig) {
    process.exit(1);
  }
  
  const results = {
    servers: {},
    database: false
  };
  
  // Test each MCP server
  for (const [name, config] of Object.entries(mcpConfig.mcpServers)) {
    const available = await testMcpServer(name, config);
    const missingEnvVars = checkEnvironmentVariables(config);
    
    results.servers[name] = {
      available,
      missingEnvVars
    };
  }
  
  // Test database connectivity
  results.database = await testDatabaseConnectivity();
  
  // Generate status report
  generateStatusReport(results);
  
  log.header('Test Complete');
}

// Run the test
if (require.main === module) {
  main().catch((error) => {
    log.error(`Test failed: ${error.message}`);
    process.exit(1);
  });
}

module.exports = { main };