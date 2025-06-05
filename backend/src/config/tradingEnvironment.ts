/**
 * Trading Environment Configuration
 * Ensures proper data source selection based on environment
 */

import { logger } from '../utils/logger';

export interface TradingEnvironmentConfig {
  mode: 'development' | 'testing' | 'staging' | 'production';
  dataSource: 'delta-exchange' | 'binance' | 'mock' | 'enhanced-mock';
  allowMockData: boolean;
  enforceDataConsistency: boolean;
  deltaExchange: {
    testnet: boolean;
    enforceTestnet: boolean;
  };
  riskManagement: {
    maxLeverage: number;
    maxRiskPerTrade: number;
    maxDailyLoss: number;
  };
  validation: {
    requireDataSourceValidation: boolean;
    requirePriceConsistencyCheck: boolean;
    priceTolerancePercent: number;
  };
}

/**
 * Get trading environment configuration based on NODE_ENV and other environment variables
 */
export function getTradingEnvironmentConfig(): TradingEnvironmentConfig {
  const nodeEnv = process.env.NODE_ENV || 'development';
  const tradingMode = process.env.TRADING_MODE || 'test';
  const forceTestnet = process.env.FORCE_TESTNET === 'true';
  
  // Determine environment mode
  let mode: TradingEnvironmentConfig['mode'] = 'development';
  if (nodeEnv === 'production' && tradingMode === 'live') {
    mode = 'production';
  } else if (nodeEnv === 'production') {
    mode = 'staging';
  } else if (nodeEnv === 'test') {
    mode = 'testing';
  }

  // Base configuration
  const config: TradingEnvironmentConfig = {
    mode,
    dataSource: 'delta-exchange', // Default to live data
    allowMockData: false,
    enforceDataConsistency: true,
    deltaExchange: {
      testnet: true, // Default to testnet for safety
      enforceTestnet: true,
    },
    riskManagement: {
      maxLeverage: 10,
      maxRiskPerTrade: 5,
      maxDailyLoss: 20,
    },
    validation: {
      requireDataSourceValidation: true,
      requirePriceConsistencyCheck: true,
      priceTolerancePercent: 5,
    },
  };

  // Environment-specific overrides
  switch (mode) {
    case 'development':
      config.allowMockData = true;
      config.deltaExchange.testnet = true;
      config.deltaExchange.enforceTestnet = true;
      config.riskManagement.maxLeverage = 5;
      config.riskManagement.maxRiskPerTrade = 2;
      break;

    case 'testing':
      config.dataSource = 'enhanced-mock'; // Allow mock data for testing
      config.allowMockData = true;
      config.enforceDataConsistency = false; // Relaxed for testing
      config.deltaExchange.testnet = true;
      config.deltaExchange.enforceTestnet = true;
      config.validation.requireDataSourceValidation = false;
      break;

    case 'staging':
      config.dataSource = 'delta-exchange';
      config.allowMockData = false;
      config.enforceDataConsistency = true;
      config.deltaExchange.testnet = true; // Still use testnet in staging
      config.deltaExchange.enforceTestnet = true;
      config.riskManagement.maxLeverage = 20;
      config.riskManagement.maxRiskPerTrade = 3;
      break;

    case 'production':
      config.dataSource = 'delta-exchange';
      config.allowMockData = false; // NEVER allow mock data in production
      config.enforceDataConsistency = true;
      config.deltaExchange.testnet = forceTestnet; // Allow production trading only if explicitly configured
      config.deltaExchange.enforceTestnet = false;
      config.riskManagement.maxLeverage = 100;
      config.riskManagement.maxRiskPerTrade = 10;
      config.riskManagement.maxDailyLoss = 50;
      break;
  }

  // Log configuration
  logger.info(`🔧 Trading Environment Configuration:`, {
    mode: config.mode,
    dataSource: config.dataSource,
    allowMockData: config.allowMockData,
    testnet: config.deltaExchange.testnet,
    enforceDataConsistency: config.enforceDataConsistency,
  });

  return config;
}

/**
 * Validate trading environment configuration
 */
export function validateTradingEnvironment(config: TradingEnvironmentConfig): void {
  const errors: string[] = [];
  const warnings: string[] = [];

  // Critical validations
  if (config.mode === 'production' && config.allowMockData) {
    errors.push('🚨 CRITICAL: Mock data is not allowed in production mode');
  }

  if (config.mode === 'production' && config.dataSource.includes('mock')) {
    errors.push('🚨 CRITICAL: Mock data source is not allowed in production mode');
  }

  if (config.mode === 'production' && !config.enforceDataConsistency) {
    errors.push('🚨 CRITICAL: Data consistency enforcement is required in production mode');
  }

  // Warning validations
  if (config.mode === 'production' && config.deltaExchange.testnet) {
    warnings.push('⚠️ WARNING: Production mode is using testnet - ensure this is intentional');
  }

  if (config.riskManagement.maxLeverage > 50) {
    warnings.push(`⚠️ WARNING: High maximum leverage (${config.riskManagement.maxLeverage}x) configured`);
  }

  if (config.riskManagement.maxRiskPerTrade > 10) {
    warnings.push(`⚠️ WARNING: High risk per trade (${config.riskManagement.maxRiskPerTrade}%) configured`);
  }

  // Log results
  if (errors.length > 0) {
    logger.error('❌ Trading environment validation failed:');
    errors.forEach(error => logger.error(`  ${error}`));
    throw new Error(`Trading environment validation failed: ${errors.join(', ')}`);
  }

  if (warnings.length > 0) {
    logger.warn('⚠️ Trading environment warnings:');
    warnings.forEach(warning => logger.warn(`  ${warning}`));
  }

  logger.info('✅ Trading environment validation passed');
}

/**
 * Apply trading environment configuration to services
 */
export function applyTradingEnvironmentConfig(config: TradingEnvironmentConfig): void {
  // Set environment variables for other services to use
  process.env.TRADING_DATA_SOURCE = config.dataSource;
  process.env.TRADING_ALLOW_MOCK_DATA = config.allowMockData.toString();
  process.env.TRADING_ENFORCE_CONSISTENCY = config.enforceDataConsistency.toString();
  process.env.DELTA_TESTNET = config.deltaExchange.testnet.toString();
  
  logger.info('🔧 Applied trading environment configuration to system');
}

/**
 * Get current trading environment status
 */
export function getTradingEnvironmentStatus(): {
  isProduction: boolean;
  isSafeForLiveTrading: boolean;
  dataSourceSafe: boolean;
  riskLevelAcceptable: boolean;
  warnings: string[];
} {
  const config = getTradingEnvironmentConfig();
  
  const isProduction = config.mode === 'production';
  const dataSourceSafe = !config.dataSource.includes('mock');
  const riskLevelAcceptable = config.riskManagement.maxRiskPerTrade <= 10;
  
  const warnings: string[] = [];
  
  if (config.allowMockData && isProduction) {
    warnings.push('Mock data allowed in production environment');
  }
  
  if (config.riskManagement.maxLeverage > 50) {
    warnings.push(`High leverage limit: ${config.riskManagement.maxLeverage}x`);
  }
  
  const isSafeForLiveTrading = dataSourceSafe && 
                               config.enforceDataConsistency && 
                               !config.allowMockData &&
                               riskLevelAcceptable;

  return {
    isProduction,
    isSafeForLiveTrading,
    dataSourceSafe,
    riskLevelAcceptable,
    warnings,
  };
}

// Initialize and validate environment on module load
const environmentConfig = getTradingEnvironmentConfig();
validateTradingEnvironment(environmentConfig);
applyTradingEnvironmentConfig(environmentConfig);

export { environmentConfig };
