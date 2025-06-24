#!/usr/bin/env node

/**
 * Environment variable validation script for SMOOPs trading bot
 * This script checks that all required environment variables are set and valid
 */

require('dotenv').config();
const fs = require('fs');
const path = require('path');
const chalk = require('chalk');

// Define required environment variables by service
const requiredVars = {
  global: [
    'NODE_ENV',
  ],
  database: [
    'POSTGRES_USER',
    'POSTGRES_PASSWORD',
    'POSTGRES_DB',
    'POSTGRES_PORT',
    'DATABASE_URL',
  ],
  backend: [
    'PORT',
    'ENCRYPTION_MASTER_KEY',
  ],
  frontend: [
    'FRONTEND_PORT',
    'NEXT_PUBLIC_API_URL',
  ],
  ml: [
    'ML_PORT',
    'PYTHONUNBUFFERED',
  ],
  exchange: [
    'DELTA_EXCHANGE_TESTNET',
    'DELTA_EXCHANGE_API_KEY',
    'DELTA_EXCHANGE_API_SECRET',
  ],
};

// Variables that need specific validation beyond just being set
const validations = {
  'DATABASE_URL': (value) => {
    const dbUrlPattern = /^postgresql:\/\/.+:.+@.+:\d+\/.+(\?.*)?$/;
    return dbUrlPattern.test(value) ? true : 'Invalid database URL format';
  },
  'DELTA_EXCHANGE_TESTNET': (value) => {
    return ['true', 'false'].includes(value.toLowerCase()) ? true : 'Must be "true" or "false"';
  },
  'NODE_ENV': (value) => {
    return ['development', 'production', 'testing'].includes(value) ? true : 'Must be "development", "production", or "testing"';
  },
  'PORT': (value) => {
    const port = parseInt(value, 10);
    return (port > 0 && port < 65536) ? true : 'Must be a valid port number (1-65535)';
  },
  'ML_PORT': (value) => {
    const port = parseInt(value, 10);
    return (port > 0 && port < 65536) ? true : 'Must be a valid port number (1-65535)';
  },
  'FRONTEND_PORT': (value) => {
    const port = parseInt(value, 10);
    return (port > 0 && port < 65536) ? true : 'Must be a valid port number (1-65535)';
  },
};

// Check for sensitive environment variables in development mode
const developmentWarnings = {
  'ENCRYPTION_MASTER_KEY': 'Using a development encryption key. Generate a secure random key for production.',
};

// Main validation function
function validateEnvironment() {
  console.log(chalk.bold('SMOOPs Environment Variable Validation'));
  console.log('=======================================');
  console.log('');

  const isProduction = process.env.NODE_ENV === 'production';
  const issues = [];
  const warnings = [];

  // Check each category of environment variables
  Object.entries(requiredVars).forEach(([category, vars]) => {
    console.log(chalk.blue.bold(`Checking ${category} variables:`));
    
    vars.forEach(varName => {
      const value = process.env[varName];
      
      // Check if variable exists
      if (!value && value !== '') {
        issues.push(`${varName} is required but not set`);
        console.log(chalk.red(`  ✖ ${varName}: Not set`));
        return;
      }
      
      // If variable has a specific validation
      if (validations[varName]) {
        const validationResult = validations[varName](value);
        if (validationResult !== true) {
          issues.push(`${varName}: ${validationResult}`);
          console.log(chalk.red(`  ✖ ${varName}: ${validationResult}`));
          return;
        }
      }
      
      // Check for development warnings
      if (!isProduction && developmentWarnings[varName]) {
        warnings.push(`${varName}: ${developmentWarnings[varName]}`);
        console.log(chalk.yellow(`  ⚠ ${varName}: ${developmentWarnings[varName]}`));
        return;
      }
      
      // If we get here, the variable is valid
      console.log(chalk.green(`  ✓ ${varName}: OK`));
    });
    
    console.log('');
  });

  // Summarize results
  if (issues.length > 0) {
    console.log(chalk.red.bold(`Found ${issues.length} issue(s) that need to be fixed:`));
    issues.forEach(issue => console.log(chalk.red(`  - ${issue}`)));
    console.log('');
  }
  
  if (warnings.length > 0) {
    console.log(chalk.yellow.bold(`Found ${warnings.length} warning(s):`));
    warnings.forEach(warning => console.log(chalk.yellow(`  - ${warning}`)));
    console.log('');
  }
  
  if (issues.length === 0 && warnings.length === 0) {
    console.log(chalk.green.bold('All environment variables are properly configured!'));
  } else if (issues.length === 0) {
    console.log(chalk.green.bold('Environment is valid but has warnings.'));
  } else {
    console.log(chalk.red.bold('Environment configuration has issues that must be fixed.'));
    process.exit(1);
  }
}

// Execute the validation
validateEnvironment(); 