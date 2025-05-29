/**
 * Environment configuration utility for SMOOPs backend
 * Handles environment variable validation and provides defaults
 */

import dotenv from 'dotenv';
import path from 'path';
import fs from 'fs';

// Load environment variables from the root .env file
dotenv.config({ path: path.join(process.cwd(), '..', '.env') });

// Check for root-level .env file
if (!fs.existsSync(path.join(process.cwd(), '..', '.env'))) {
  console.warn('\x1b[33m%s\x1b[0m', 'WARNING: No .env file found at project root. Using default values.');
}

interface EnvironmentConfig {
  // Node environment (development, production, test)
  NODE_ENV: string;
  
  // Server configuration
  PORT: number;
  HOST: string;
  
  // Database configuration
  DATABASE_URL: string;
  
  // JWT configuration for authentication
  JWT_SECRET: string;
  JWT_EXPIRES_IN: string;
  JWT_REFRESH_SECRET: string;
  
  // CORS configuration
  CORS_ORIGIN: string;
  
  // Client URL for email links
  CLIENT_URL: string;
  
  // Email configuration
  EMAIL_HOST: string;
  EMAIL_PORT: string;
  EMAIL_USER: string;
  EMAIL_PASSWORD: string;
  EMAIL_FROM: string;
  EMAIL_FROM_NAME: string;
  
  // Encryption for API keys
  ENCRYPTION_MASTER_KEY: string;
  
  // Exchange configuration
  DELTA_EXCHANGE_TESTNET: boolean;
  DELTA_EXCHANGE_API_URL: string;
  
  // ML service configuration
  ML_SERVICE_URL: string;
  
  // Logging configuration
  LOG_LEVEL: string;
  
  // Cookie configuration
  COOKIE_DOMAIN?: string;
  COOKIE_SECRET: string;
}

// Environment variables with defaults
const env: EnvironmentConfig = {
  // Node environment (development, production, test)
  NODE_ENV: process.env.NODE_ENV || 'development',
  
  // Server configuration
  PORT: parseInt(process.env.PORT || '3001', 10),
  HOST: process.env.HOST || '0.0.0.0',
  
  // Database configuration
  DATABASE_URL: process.env.DATABASE_URL || 'postgresql://postgres:postgres@localhost:5432/smoops?schema=public',
  
  // JWT configuration for authentication
  JWT_SECRET: process.env.JWT_SECRET || 'dev-jwt-secret-do-not-use-in-production',
  JWT_EXPIRES_IN: process.env.JWT_EXPIRES_IN || '7d',
  JWT_REFRESH_SECRET: process.env.JWT_REFRESH_SECRET || process.env.JWT_SECRET || 'dev-refresh-jwt-secret',
  
  // CORS configuration
  CORS_ORIGIN: process.env.CORS_ORIGIN || '*',
  
  // Client URL for email links
  CLIENT_URL: process.env.CLIENT_URL || 'http://localhost:3000',
  
  // Email configuration
  EMAIL_HOST: process.env.EMAIL_HOST || 'smtp.ethereal.email',
  EMAIL_PORT: process.env.EMAIL_PORT || '587',
  EMAIL_USER: process.env.EMAIL_USER || '',
  EMAIL_PASSWORD: process.env.EMAIL_PASSWORD || '',
  EMAIL_FROM: process.env.EMAIL_FROM || 'noreply@smartmarketoops.com',
  EMAIL_FROM_NAME: process.env.EMAIL_FROM_NAME || 'SmartMarket OOPS',
  
  // Encryption for API keys
  ENCRYPTION_MASTER_KEY: process.env.ENCRYPTION_MASTER_KEY || 'development_key_do_not_use_in_production',
  
  // Exchange configuration
  DELTA_EXCHANGE_TESTNET: process.env.DELTA_EXCHANGE_TESTNET === 'false' ? false : true,
  DELTA_EXCHANGE_API_URL: process.env.DELTA_EXCHANGE_API_URL || 'https://testnet-api.delta.exchange',
  
  // ML service configuration
  ML_SERVICE_URL: process.env.ML_SERVICE_URL || 'http://localhost:3002',
  
  // Logging configuration
  LOG_LEVEL: process.env.LOG_LEVEL || 'info',
  
  // Cookie configuration
  COOKIE_DOMAIN: process.env.COOKIE_DOMAIN,
  COOKIE_SECRET: process.env.COOKIE_SECRET || 'SmartMarketOOPS-cookie-secret-key',
};

// Validate critical environment variables
function validateEnvironment(): void {
  const errors: string[] = [];
  
  // Check for production with default secret
  if (env.NODE_ENV === 'production') {
    if (env.JWT_SECRET === 'dev-jwt-secret-do-not-use-in-production') {
      errors.push('JWT_SECRET is using default value in production mode');
    }
    
    if (env.JWT_REFRESH_SECRET === 'dev-refresh-jwt-secret') {
      errors.push('JWT_REFRESH_SECRET is using default value in production mode');
    }
    
    if (env.ENCRYPTION_MASTER_KEY === 'development_key_do_not_use_in_production') {
      errors.push('ENCRYPTION_MASTER_KEY is using default value in production mode');
    }
    
    if (env.CORS_ORIGIN === '*') {
      errors.push('CORS_ORIGIN should not be * in production mode');
    }
    
    if (!env.EMAIL_USER || !env.EMAIL_PASSWORD) {
      errors.push('EMAIL_USER and EMAIL_PASSWORD must be set in production mode');
    }
  }
  
  // Validate DATABASE_URL format
  const dbUrlPattern = /^postgresql:\/\/.+:.+@.+:\d+\/.+(\?.*)?$/;
  if (!dbUrlPattern.test(env.DATABASE_URL)) {
    errors.push('DATABASE_URL is invalid or missing');
  }
  
  // Log errors and exit if in production
  if (errors.length > 0) {
    console.error('\x1b[31m%s\x1b[0m', 'Environment validation errors:');
    errors.forEach(error => console.error(`- ${error}`));
    
    if (env.NODE_ENV === 'production') {
      console.error('\x1b[31m%s\x1b[0m', 'Exiting due to environment validation errors in production mode.');
      process.exit(1);
    }
  }
}

// In development, log the environment configuration
if (env.NODE_ENV === 'development') {
  console.log('\x1b[36m%s\x1b[0m', 'Environment Configuration:');
  // Strip sensitive values
  const logSafeEnv = { ...env };
  // Mask sensitive values
  logSafeEnv.JWT_SECRET = logSafeEnv.JWT_SECRET ? '********' : 'not set';
  logSafeEnv.JWT_REFRESH_SECRET = logSafeEnv.JWT_REFRESH_SECRET ? '********' : 'not set';
  logSafeEnv.ENCRYPTION_MASTER_KEY = logSafeEnv.ENCRYPTION_MASTER_KEY ? '********' : 'not set';
  logSafeEnv.DATABASE_URL = logSafeEnv.DATABASE_URL.replace(/\/\/(.+):(.+)@/, '//******:******@');
  logSafeEnv.EMAIL_PASSWORD = logSafeEnv.EMAIL_PASSWORD ? '********' : 'not set';
  
  // Log safe environment
  console.log(JSON.stringify(logSafeEnv, null, 2));
}

// Run validation
validateEnvironment();

export default env; 