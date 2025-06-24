#!/usr/bin/env node

/**
 * Environment variable validation script
 * Checks all required environment variables across the application
 */

require('dotenv').config();

const RED = '\x1b[31m';
const GREEN = '\x1b[32m';
const YELLOW = '\x1b[33m';
const RESET = '\x1b[0m';

// Required variables by component
const requiredVars = {
  core: [
    'NODE_ENV',
    'TRADING_MODE',
    'HOST',
    'PORT',
    'FRONTEND_PORT',
    'ML_PORT'
  ],
  database: [
    'DATABASE_URL',
    'POSTGRES_USER',
    'POSTGRES_PASSWORD',
    'POSTGRES_DB',
    'POSTGRES_PORT'
  ],
  security: [
    'JWT_SECRET',
    'JWT_EXPIRES_IN',
    'JWT_REFRESH_SECRET',
    'COOKIE_SECRET',
    'ENCRYPTION_MASTER_KEY'
  ],
  client: [
    'CLIENT_URL',
    'CORS_ORIGIN',
    'NEXT_PUBLIC_API_URL'
  ],
  deltaExchange: [
    'DELTA_EXCHANGE_API_KEY',
    'DELTA_EXCHANGE_API_SECRET',
    'DELTA_EXCHANGE_TESTNET',
    'DELTA_EXCHANGE_BASE_URL'
  ],
  email: [
    'EMAIL_HOST',
    'EMAIL_PORT',
    'EMAIL_USER',
    'EMAIL_PASSWORD',
    'EMAIL_FROM',
    'EMAIL_FROM_NAME'
  ]
};

// Production-only requirements
const productionOnlyVars = [
  'JWT_SECRET',
  'JWT_REFRESH_SECRET',
  'ENCRYPTION_MASTER_KEY',
  'EMAIL_USER',
  'EMAIL_PASSWORD'
];

// Default values that shouldn't be used in production
const productionDefaultChecks = {
  JWT_SECRET: 'dev-jwt-secret-do-not-use-in-production',
  JWT_REFRESH_SECRET: 'dev-refresh-jwt-secret',
  ENCRYPTION_MASTER_KEY: 'development_key_do_not_use_in_production',
  COOKIE_SECRET: 'SmartMarketOOPS-cookie-secret-key'
};

function checkVariable(varName, required = true) {
  const value = process.env[varName];
  const isProduction = process.env.NODE_ENV === 'production';
  
  if (!value && required) {
    console.log(`${RED}✗ ${varName}${RESET} is not set`);
    return false;
  }
  
  if (value && isProduction) {
    // Check if using default values in production
    if (productionDefaultChecks[varName] && value === productionDefaultChecks[varName]) {
      console.log(`${RED}✗ ${varName}${RESET} is using default value in production`);
      return false;
    }
    
    // Check production-only requirements
    if (productionOnlyVars.includes(varName) && (!value || value.trim() === '')) {
      console.log(`${RED}✗ ${varName}${RESET} must be set in production`);
      return false;
    }
  }
  
  if (value) {
    console.log(`${GREEN}✓ ${varName}${RESET} is configured`);
    return true;
  }
  
  console.log(`${YELLOW}? ${varName}${RESET} is not set (optional)`);
  return true;
}

function checkSection(section, vars) {
  console.log(`\n${YELLOW}Checking ${section} configuration:${RESET}`);
  const results = vars.map(v => checkVariable(v));
  return results.every(r => r);
}

// Main validation
console.log('🔍 Checking environment configuration...\n');

let allValid = true;

// Check each section
for (const [section, vars] of Object.entries(requiredVars)) {
  const sectionValid = checkSection(section, vars);
  allValid = allValid && sectionValid;
}

// Additional production checks
if (process.env.NODE_ENV === 'production') {
  console.log('\n🚨 Production environment detected - performing additional checks:');
  
  // Check CORS in production
  if (process.env.CORS_ORIGIN === '*') {
    console.log(`${RED}✗ CORS_ORIGIN${RESET} should not be * in production`);
    allValid = false;
  }
  
  // Validate DATABASE_URL format
  const dbUrlPattern = /^postgresql:\/\/.+:.+@.+:\d+\/.+(\?.*)?$/;
  if (!dbUrlPattern.test(process.env.DATABASE_URL || '')) {
    console.log(`${RED}✗ DATABASE_URL${RESET} format is invalid`);
    allValid = false;
  }
}

// Final status
console.log('\n' + (allValid 
  ? `${GREEN}✅ All environment variables are properly configured${RESET}`
  : `${RED}❌ Some environment variables need attention${RESET}`
));

// Exit with appropriate code
process.exit(allValid ? 0 : 1); 