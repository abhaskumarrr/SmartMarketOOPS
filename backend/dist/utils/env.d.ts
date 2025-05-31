/**
 * Environment configuration utility for SMOOPs backend
 * Handles environment variable validation and provides defaults
 */
interface EnvironmentConfig {
    NODE_ENV: string;
    PORT: number;
    HOST: string;
    DATABASE_URL: string;
    JWT_SECRET: string;
    JWT_EXPIRES_IN: string;
    JWT_REFRESH_SECRET: string;
    CORS_ORIGIN: string;
    CLIENT_URL: string;
    EMAIL_HOST: string;
    EMAIL_PORT: string;
    EMAIL_USER: string;
    EMAIL_PASSWORD: string;
    EMAIL_FROM: string;
    EMAIL_FROM_NAME: string;
    ENCRYPTION_MASTER_KEY: string;
    DELTA_EXCHANGE_TESTNET: boolean;
    DELTA_EXCHANGE_API_URL: string;
    ML_SERVICE_URL: string;
    LOG_LEVEL: string;
    COOKIE_DOMAIN?: string;
    COOKIE_SECRET: string;
}
declare const env: EnvironmentConfig;
export default env;
