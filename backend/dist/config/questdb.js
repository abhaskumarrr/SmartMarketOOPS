"use strict";
/**
 * QuestDB Configuration
 * High-performance time-series database configuration for SmartMarketOOPS
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.questdbClientOptions = exports.questdbConfig = exports.questdbConnection = exports.QuestDBConnection = void 0;
exports.createQuestDBClient = createQuestDBClient;
exports.validateQuestDBEnvironment = validateQuestDBEnvironment;
exports.connectWithRetry = connectWithRetry;
const nodejs_client_1 = require("@questdb/nodejs-client");
const logger_1 = require("../utils/logger");
// Default QuestDB configuration
const defaultConfig = {
    host: process.env.QUESTDB_HOST || 'localhost',
    port: parseInt(process.env.QUESTDB_PORT || '9000', 10),
    username: process.env.QUESTDB_USERNAME,
    password: process.env.QUESTDB_PASSWORD,
    database: process.env.QUESTDB_DATABASE || 'qdb',
    ssl: process.env.QUESTDB_SSL === 'true',
    connectionTimeout: parseInt(process.env.QUESTDB_CONNECTION_TIMEOUT || '30000', 10),
    queryTimeout: parseInt(process.env.QUESTDB_QUERY_TIMEOUT || '60000', 10),
    maxConnections: parseInt(process.env.QUESTDB_MAX_CONNECTIONS || '10', 10),
    retryAttempts: parseInt(process.env.QUESTDB_RETRY_ATTEMPTS || '3', 10),
    retryDelay: parseInt(process.env.QUESTDB_RETRY_DELAY || '1000', 10),
};
exports.questdbConfig = defaultConfig;
// QuestDB client configuration for line protocol (high-performance ingestion)
const defaultClientOptions = {
    host: process.env.QUESTDB_ILP_HOST || 'localhost',
    port: parseInt(process.env.QUESTDB_ILP_PORT || '9009', 10),
    username: process.env.QUESTDB_ILP_USERNAME,
    password: process.env.QUESTDB_ILP_PASSWORD,
    token: process.env.QUESTDB_ILP_TOKEN,
    tls: process.env.QUESTDB_ILP_TLS === 'true',
    autoFlush: true,
    autoFlushRows: parseInt(process.env.QUESTDB_AUTO_FLUSH_ROWS || '1000', 10),
    autoFlushInterval: parseInt(process.env.QUESTDB_AUTO_FLUSH_INTERVAL || '1000', 10),
    requestMinThroughput: parseInt(process.env.QUESTDB_MIN_THROUGHPUT || '1024', 10),
    requestTimeout: parseInt(process.env.QUESTDB_REQUEST_TIMEOUT || '10000', 10),
    retryTimeout: parseInt(process.env.QUESTDB_RETRY_TIMEOUT || '60000', 10),
    maxBufferSize: parseInt(process.env.QUESTDB_MAX_BUFFER_SIZE || '65536', 10),
};
exports.questdbClientOptions = defaultClientOptions;
class QuestDBConnection {
    constructor() {
        this.client = null;
        this.isConnected = false;
        this.connectionPromise = null;
        this.config = { ...defaultConfig };
        this.clientOptions = { ...defaultClientOptions };
    }
    static getInstance() {
        if (!QuestDBConnection.instance) {
            QuestDBConnection.instance = new QuestDBConnection();
        }
        return QuestDBConnection.instance;
    }
    async connect() {
        if (this.isConnected && this.client) {
            return;
        }
        if (this.connectionPromise) {
            return this.connectionPromise;
        }
        this.connectionPromise = this._connect();
        return this.connectionPromise;
    }
    async _connect() {
        try {
            logger_1.logger.info('🔌 Connecting to QuestDB...', {
                host: this.clientOptions.host,
                port: this.clientOptions.port,
                tls: this.clientOptions.tls,
            });
            // Create QuestDB client using ILP (Influx Line Protocol) over TCP
            const configString = `tcp::addr=${this.clientOptions.host}:${this.clientOptions.port || 9009};auto_flush_rows=${this.clientOptions.autoFlushRows || 1000};auto_flush_interval=${this.clientOptions.autoFlushInterval || 1000};`;
            this.client = nodejs_client_1.Sender.fromConfig(configString);
            // Connect to QuestDB
            await this.client.connect();
            this.isConnected = true;
            logger_1.logger.info('✅ QuestDB connection established successfully');
        }
        catch (error) {
            this.isConnected = false;
            this.client = null;
            this.connectionPromise = null;
            logger_1.logger.error('❌ Failed to connect to QuestDB:', error);
            throw new Error(`QuestDB connection failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
    }
    async disconnect() {
        if (this.client) {
            try {
                await this.client.close();
                logger_1.logger.info('🔌 QuestDB connection closed');
            }
            catch (error) {
                logger_1.logger.error('❌ Error closing QuestDB connection:', error);
            }
            finally {
                this.client = null;
                this.isConnected = false;
                this.connectionPromise = null;
            }
        }
    }
    getClient() {
        if (!this.client || !this.isConnected) {
            throw new Error('QuestDB client not connected. Call connect() first.');
        }
        return this.client;
    }
    isReady() {
        return this.isConnected && this.client !== null;
    }
    getConfig() {
        return { ...this.config };
    }
    updateConfig(newConfig) {
        this.config = { ...this.config, ...newConfig };
    }
    updateClientOptions(newOptions) {
        this.clientOptions = { ...this.clientOptions, ...newOptions };
    }
    // Health check method
    async healthCheck() {
        try {
            if (!this.isConnected || !this.client) {
                return false;
            }
            // Simple health check - try to flush any pending data
            await this.client.flush();
            return true;
        }
        catch (error) {
            logger_1.logger.error('QuestDB health check failed:', error);
            return false;
        }
    }
    // Get connection statistics
    getStats() {
        return {
            isConnected: this.isConnected,
            config: this.config,
            clientOptions: this.clientOptions,
        };
    }
}
exports.QuestDBConnection = QuestDBConnection;
// Export singleton instance
exports.questdbConnection = QuestDBConnection.getInstance();
// Utility function to create a new client with custom options
function createQuestDBClient(options) {
    const clientOptions = { ...defaultClientOptions, ...options };
    const configString = `http::addr=${clientOptions.host}:${clientOptions.port};`;
    return nodejs_client_1.Sender.fromConfig(configString);
}
// Environment validation
function validateQuestDBEnvironment() {
    const errors = [];
    // Check required environment variables
    if (!process.env.QUESTDB_HOST && !process.env.QUESTDB_ILP_HOST) {
        errors.push('QuestDB host not configured (QUESTDB_HOST or QUESTDB_ILP_HOST)');
    }
    // Validate port numbers
    const port = parseInt(process.env.QUESTDB_ILP_PORT || '9009', 10);
    if (isNaN(port) || port < 1 || port > 65535) {
        errors.push('Invalid QuestDB ILP port number');
    }
    // Validate timeout values
    const timeouts = [
        'QUESTDB_CONNECTION_TIMEOUT',
        'QUESTDB_QUERY_TIMEOUT',
        'QUESTDB_REQUEST_TIMEOUT',
        'QUESTDB_RETRY_TIMEOUT',
    ];
    timeouts.forEach(envVar => {
        const value = process.env[envVar];
        if (value && (isNaN(parseInt(value, 10)) || parseInt(value, 10) < 0)) {
            errors.push(`Invalid ${envVar} value: ${value}`);
        }
    });
    return {
        valid: errors.length === 0,
        errors,
    };
}
// Connection retry utility
async function connectWithRetry(maxAttempts = 3, delay = 1000) {
    let lastError = null;
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
            await exports.questdbConnection.connect();
            return;
        }
        catch (error) {
            lastError = error instanceof Error ? error : new Error('Unknown connection error');
            logger_1.logger.warn(`QuestDB connection attempt ${attempt}/${maxAttempts} failed: ${lastError.message}`);
            if (attempt < maxAttempts) {
                await new Promise(resolve => setTimeout(resolve, delay * attempt));
            }
        }
    }
    throw new Error(`Failed to connect to QuestDB after ${maxAttempts} attempts. Last error: ${lastError?.message}`);
}
//# sourceMappingURL=questdb.js.map