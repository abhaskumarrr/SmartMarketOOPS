/**
 * Trading Environment Configuration
 * Ensures proper data source selection based on environment
 */
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
export declare function getTradingEnvironmentConfig(): TradingEnvironmentConfig;
/**
 * Validate trading environment configuration
 */
export declare function validateTradingEnvironment(config: TradingEnvironmentConfig): void;
/**
 * Apply trading environment configuration to services
 */
export declare function applyTradingEnvironmentConfig(config: TradingEnvironmentConfig): void;
/**
 * Get current trading environment status
 */
export declare function getTradingEnvironmentStatus(): {
    isProduction: boolean;
    isSafeForLiveTrading: boolean;
    dataSourceSafe: boolean;
    riskLevelAcceptable: boolean;
    warnings: string[];
};
declare const environmentConfig: TradingEnvironmentConfig;
export { environmentConfig };
