/**
 * Environment Configuration Utility
 * Provides a centralized way to access environment variables with validation and defaults
 */

const dotenv = require('dotenv');
const path = require('path');
const fs = require('fs');

// Load environment variables from .env file
dotenv.config({
  path: path.resolve(__dirname, '../../../.env')
});

// Define environment variables with defaults and validation
const env = {
  NODE_ENV: process.env.NODE_ENV || 'development',
  PORT: parseInt(process.env.PORT || '3001', 10),
  HOST: process.env.HOST || '0.0.0.0',
  DATABASE_URL: process.env.DATABASE_URL,
  JWT_SECRET: process.env.JWT_SECRET || 'dev-jwt-secret-do-not-use-in-production',
  JWT_EXPIRES_IN: process.env.JWT_EXPIRES_IN || '1h',
  JWT_REFRESH_SECRET: process.env.JWT_REFRESH_SECRET || 'dev-refresh-jwt-secret-do-not-use-in-production',
  SESSION_MAX_AGE: parseInt(process.env.SESSION_MAX_AGE || '3600000', 10), // 1h in ms
  REMEMBER_ME_MAX_AGE: parseInt(process.env.REMEMBER_ME_MAX_AGE || '2592000000', 10), // 30 days in ms
  SESSION_INACTIVITY_TIMEOUT: parseInt(process.env.SESSION_INACTIVITY_TIMEOUT || '1800000', 10), // 30 min in ms
  CORS_ORIGIN: process.env.CLIENT_URL || 'http://localhost:3000',
  CLIENT_URL: process.env.CLIENT_URL || 'http://localhost:3000',
  EMAIL_HOST: process.env.EMAIL_HOST || 'smtp.ethereal.email',
  EMAIL_PORT: parseInt(process.env.EMAIL_PORT || '587', 10),
  EMAIL_USER: process.env.EMAIL_USER || '',
  EMAIL_PASSWORD: process.env.EMAIL_PASSWORD || '',
  EMAIL_FROM: process.env.EMAIL_FROM || 'noreply@smartmarketoops.com',
  EMAIL_FROM_NAME: process.env.EMAIL_FROM_NAME || 'SmartMarket OOPS',
  COOKIE_SECRET: process.env.COOKIE_SECRET || 'SmartMarketOOPS-cookie-secret-key',
  COOKIE_DOMAIN: process.env.COOKIE_DOMAIN || 'localhost',
  ENCRYPTION_MASTER_KEY: process.env.ENCRYPTION_MASTER_KEY || '',
  DELTA_EXCHANGE_TESTNET: process.env.DELTA_EXCHANGE_TESTNET === 'true',
  DELTA_EXCHANGE_API_KEY: process.env.DELTA_EXCHANGE_API_KEY || '',
  DELTA_EXCHANGE_API_SECRET: process.env.DELTA_EXCHANGE_API_SECRET || '',
  DELTA_API_RATE_LIMIT: parseInt(process.env.DELTA_API_RATE_LIMIT || '30', 10),
  DELTA_API_RATE_WINDOW: parseInt(process.env.DELTA_API_RATE_WINDOW || '60000', 10), // 1 minute in ms
  DELTA_EXCHANGE_WS_ENABLED: process.env.DELTA_EXCHANGE_WS_ENABLED === 'true',
  DELTA_EXCHANGE_WS_RECONNECT_INTERVAL: parseInt(process.env.DELTA_EXCHANGE_WS_RECONNECT_INTERVAL || '5000', 10)
};

// Check for required variables and warn if missing
const requiredVars = ['DATABASE_URL', 'JWT_SECRET', 'JWT_REFRESH_SECRET', 'COOKIE_SECRET'];

requiredVars.forEach(variable => {
  if (!env[variable]) {
    console.warn(`WARNING: Environment variable ${variable} is not set and no default value provided.`);
  }
});

module.exports = { env }; 