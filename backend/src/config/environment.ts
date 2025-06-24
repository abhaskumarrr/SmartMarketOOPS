/**
 * Centralized Environment Configuration
 * Provides a single source of truth for all environment variables
 */

// System Configuration
export const NODE_ENV = process.env.NODE_ENV || 'development';
export const PORT = parseInt(process.env.PORT || '3006', 10);
export const TRADING_MODE = process.env.TRADING_MODE || 'test';
export const FORCE_TESTNET = process.env.FORCE_TESTNET === 'true';

// Database Configuration
export const DATABASE_URL = process.env.DATABASE_URL || 'postgresql://postgres:postgres@localhost:5432/smartmarket';
export const POSTGRES_USER = process.env.POSTGRES_USER || 'postgres';
export const POSTGRES_PASSWORD = process.env.POSTGRES_PASSWORD || 'postgres';
export const POSTGRES_DB = process.env.POSTGRES_DB || 'smartmarket';
export const POSTGRES_PORT = parseInt(process.env.POSTGRES_PORT || '5432', 10);

// Redis Configuration
export const REDIS_URL = process.env.REDIS_URL || 'redis://localhost:6379/0';
export const REDIS_HOST = process.env.REDIS_HOST || 'localhost';
export const REDIS_PORT = parseInt(process.env.REDIS_PORT || '6379', 10);
export const REDIS_PASSWORD = process.env.REDIS_PASSWORD || '';

// QuestDB Configuration
export const QUESTDB_HOST = process.env.QUESTDB_HOST || 'localhost';
export const QUESTDB_PORT = parseInt(process.env.QUESTDB_PORT || '9000', 10);
export const QUESTDB_HTTP_PORT = parseInt(process.env.QUESTDB_HTTP_PORT || '9009', 10);

// Security & Authentication
export const JWT_SECRET = process.env.JWT_SECRET || 'development-jwt-secret-key-change-in-production';
export const JWT_EXPIRES_IN = process.env.JWT_EXPIRES_IN || '1h';
export const JWT_REFRESH_SECRET = process.env.JWT_REFRESH_SECRET || 'development-refresh-secret-key-change-in-production';
export const COOKIE_SECRET = process.env.COOKIE_SECRET || 'development-cookie-secret-key-change-in-production';
export const COOKIE_DOMAIN = process.env.COOKIE_DOMAIN || 'localhost';

// Encryption
export const ENCRYPTION_MASTER_KEY = process.env.ENCRYPTION_MASTER_KEY || 'development-encryption-key-32chars';
export const ENCRYPTION_KEY_SECONDARY = process.env.ENCRYPTION_KEY_SECONDARY || 'development-secondary-key-32chars';

// Session Configuration
export const SESSION_MAX_AGE = parseInt(process.env.SESSION_MAX_AGE || '3600000', 10); // 1 hour
export const REMEMBER_ME_MAX_AGE = parseInt(process.env.REMEMBER_ME_MAX_AGE || '2592000000', 10); // 30 days
export const SESSION_INACTIVITY_TIMEOUT = parseInt(process.env.SESSION_INACTIVITY_TIMEOUT || '1800000', 10); // 30 minutes

// CORS & Client Configuration
export const CLIENT_URL = process.env.CLIENT_URL || 'http://localhost:3000';
export const CORS_ORIGIN = process.env.CORS_ORIGIN || 'http://localhost:3000';
export const NEXT_PUBLIC_API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3006';

// Delta Exchange API Configuration
export const DELTA_EXCHANGE_API_KEY = process.env.DELTA_EXCHANGE_API_KEY || '';
export const DELTA_EXCHANGE_API_SECRET = process.env.DELTA_EXCHANGE_API_SECRET || '';
export const DELTA_EXCHANGE_TESTNET = process.env.DELTA_EXCHANGE_TESTNET === 'true';
export const DELTA_EXCHANGE_BASE_URL = process.env.DELTA_EXCHANGE_BASE_URL || 
  (DELTA_EXCHANGE_TESTNET ? 'https://cdn-ind.testnet.deltaex.org' : 'https://api.india.delta.exchange');

// Delta Exchange Configuration
export const DELTA_API_RATE_LIMIT = parseInt(process.env.DELTA_API_RATE_LIMIT || '30', 10);
export const DELTA_API_RATE_WINDOW = parseInt(process.env.DELTA_API_RATE_WINDOW || '60000', 10);
export const DELTA_EXCHANGE_WS_ENABLED = process.env.DELTA_EXCHANGE_WS_ENABLED === 'true';
export const DELTA_EXCHANGE_WS_RECONNECT_INTERVAL = parseInt(process.env.DELTA_EXCHANGE_WS_RECONNECT_INTERVAL || '5000', 10);

// Product IDs
export const DELTA_BTCUSD_PRODUCT_ID = parseInt(process.env.DELTA_BTCUSD_PRODUCT_ID || '84', 10);
export const DELTA_ETHUSD_PRODUCT_ID = parseInt(process.env.DELTA_ETHUSD_PRODUCT_ID || '1699', 10);
export const DELTA_SOLUSD_PRODUCT_ID = parseInt(process.env.DELTA_SOLUSD_PRODUCT_ID || '92572', 10);

// ML Configuration
export const ML_API_URL = process.env.ML_API_URL || 'http://localhost:3002/api';
export const ML_API_KEY = process.env.ML_API_KEY || '';
export const ML_SYSTEM_RECONNECT_INTERVAL = parseInt(process.env.ML_SYSTEM_RECONNECT_INTERVAL || '30000', 10);
export const ML_HEALTH_CHECK_INTERVAL = parseInt(process.env.ML_HEALTH_CHECK_INTERVAL || '300000', 10);
export const ML_BATCH_SIZE = parseInt(process.env.ML_BATCH_SIZE || '20', 10);
export const ML_MAX_CONCURRENT_REQUESTS = parseInt(process.env.ML_MAX_CONCURRENT_REQUESTS || '5', 10);
export const ML_REQUEST_TIMEOUT = parseInt(process.env.ML_REQUEST_TIMEOUT || '60000', 10);
export const ML_MAX_RETRIES = parseInt(process.env.ML_MAX_RETRIES || '3', 10);
export const ML_AUTO_SIGNAL_GENERATION = process.env.ML_AUTO_SIGNAL_GENERATION === 'true';

// Monitoring & Logging
export const LOG_LEVEL = process.env.LOG_LEVEL || 'INFO';
export const LOG_FILE = process.env.LOG_FILE || 'logs/smartmarket.log';
export const ENABLE_METRICS = process.env.ENABLE_METRICS === 'true';
export const METRICS_PORT = parseInt(process.env.METRICS_PORT || '9090', 10);
export const ENABLE_HEALTH_CHECKS = process.env.ENABLE_HEALTH_CHECKS !== 'false';

// Development Tools
export const ENABLE_DEBUG_ROUTES = process.env.ENABLE_DEBUG_ROUTES === 'true';
export const ENABLE_API_DOCS = process.env.ENABLE_API_DOCS !== 'false';
export const ENABLE_CORS_ALL = process.env.ENABLE_CORS_ALL === 'true';

// Export all environment variables as a single object
export default {
  NODE_ENV,
  PORT,
  TRADING_MODE,
  FORCE_TESTNET,
  DATABASE_URL,
  POSTGRES_USER,
  POSTGRES_PASSWORD,
  POSTGRES_DB,
  POSTGRES_PORT,
  REDIS_URL,
  REDIS_HOST,
  REDIS_PORT,
  REDIS_PASSWORD,
  QUESTDB_HOST,
  QUESTDB_PORT,
  QUESTDB_HTTP_PORT,
  JWT_SECRET,
  JWT_EXPIRES_IN,
  JWT_REFRESH_SECRET,
  COOKIE_SECRET,
  COOKIE_DOMAIN,
  ENCRYPTION_MASTER_KEY,
  ENCRYPTION_KEY_SECONDARY,
  SESSION_MAX_AGE,
  REMEMBER_ME_MAX_AGE,
  SESSION_INACTIVITY_TIMEOUT,
  CLIENT_URL,
  CORS_ORIGIN,
  NEXT_PUBLIC_API_URL,
  DELTA_EXCHANGE_API_KEY,
  DELTA_EXCHANGE_API_SECRET,
  DELTA_EXCHANGE_TESTNET,
  DELTA_EXCHANGE_BASE_URL,
  DELTA_API_RATE_LIMIT,
  DELTA_API_RATE_WINDOW,
  DELTA_EXCHANGE_WS_ENABLED,
  DELTA_EXCHANGE_WS_RECONNECT_INTERVAL,
  DELTA_BTCUSD_PRODUCT_ID,
  DELTA_ETHUSD_PRODUCT_ID,
  DELTA_SOLUSD_PRODUCT_ID,
  ML_API_URL,
  ML_API_KEY,
  ML_SYSTEM_RECONNECT_INTERVAL,
  ML_HEALTH_CHECK_INTERVAL,
  ML_BATCH_SIZE,
  ML_MAX_CONCURRENT_REQUESTS,
  ML_REQUEST_TIMEOUT,
  ML_MAX_RETRIES,
  ML_AUTO_SIGNAL_GENERATION,
  LOG_LEVEL,
  LOG_FILE,
  ENABLE_METRICS,
  METRICS_PORT,
  ENABLE_HEALTH_CHECKS,
  ENABLE_DEBUG_ROUTES,
  ENABLE_API_DOCS,
  ENABLE_CORS_ALL
};