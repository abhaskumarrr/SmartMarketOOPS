/**
 * Environment configuration for SMOOPs frontend
 *
 * This module exposes environment variables to the frontend application.
 *
 * IMPORTANT: Only include variables that are safe to expose to the client.
 * Any sensitive information should be kept server-side only.
 */

// API URL for backend services
export const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';

// Environment mode (development, production, test)
export const NODE_ENV = process.env.NODE_ENV || 'development';

// Flag to determine if we're using testnet or real exchange
export const IS_TESTNET = process.env.NEXT_PUBLIC_DELTA_EXCHANGE_TESTNET !== 'false';

// Feature flags
export const FEATURES = {
  // Enable/disable real trading features
  REAL_TRADING: process.env.NEXT_PUBLIC_FEATURE_REAL_TRADING === 'true',

  // Enable/disable backtesting feature
  BACKTESTING: process.env.NEXT_PUBLIC_FEATURE_BACKTESTING !== 'false',

  // Enable/disable advanced analytics
  ADVANCED_ANALYTICS: process.env.NEXT_PUBLIC_FEATURE_ADVANCED_ANALYTICS !== 'false',
};

// UI configuration
export const UI_CONFIG = {
  // Default theme (light, dark, system)
  DEFAULT_THEME: process.env.NEXT_PUBLIC_DEFAULT_THEME || 'system',

  // Default chart timeframe
  DEFAULT_TIMEFRAME: process.env.NEXT_PUBLIC_DEFAULT_TIMEFRAME || '4h',

  // Default trading pair
  DEFAULT_PAIR: process.env.NEXT_PUBLIC_DEFAULT_PAIR || 'BTC-USDT',
};

// Application metadata
export const APP_META = {
  APP_NAME: 'SMOOPs Trading Bot',
  APP_VERSION: process.env.NEXT_PUBLIC_APP_VERSION || '1.0.0',
  APP_DESCRIPTION: 'Smart Money Order Blocks Trading Bot',
};

// Validate environment (client-side)
export const validateEnvironment = () => {
  const warnings = [];

  if (!API_URL) {
    warnings.push('API_URL is not configured. The application may not function correctly.');
  }

  if (FEATURES.REAL_TRADING && IS_TESTNET) {
    warnings.push(
      'Real trading is enabled but testnet mode is also enabled. This may cause confusion.',
    );
  }

  return {
    isValid: warnings.length === 0,
    warnings,
  };
};

export default {
  API_URL,
  NODE_ENV,
  IS_TESTNET,
  FEATURES,
  UI_CONFIG,
  APP_META,
  validateEnvironment,
};
