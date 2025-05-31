"use strict";
/**
 * Database Health Check Utility
 * Functions for checking database connectivity and health
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.checkDbConnection = checkDbConnection;
exports.performDatabaseHealthCheck = performDatabaseHealthCheck;
const prismaClient_1 = __importDefault(require("./prismaClient"));
const logger_1 = require("./logger");
// Create logger
const logger = (0, logger_1.createLogger)('DBHealthCheck');
/**
 * Checks the database connection health
 * @returns {Promise<ConnectionStatus>} The health check result
 */
async function checkDbConnection() {
    const startTime = Date.now();
    try {
        // Execute a simple query to verify database connection
        await prismaClient_1.default.$queryRaw `SELECT 1`;
        // Calculate query execution time
        const responseTime = Date.now() - startTime;
        return {
            success: true,
            message: 'Database connection is healthy',
            responseTime: `${responseTime}ms`
        };
    }
    catch (error) {
        return {
            success: false,
            message: 'Database connection failed',
            error: error.message,
            responseTime: `${Date.now() - startTime}ms`
        };
    }
}
/**
 * Performs a comprehensive database health check
 * @returns {Promise<HealthCheckResult>} Detailed health check result
 */
async function performDatabaseHealthCheck() {
    try {
        // Check connection
        const connectionStatus = await checkDbConnection();
        // If connection failed, return early
        if (!connectionStatus.success) {
            return {
                success: false,
                connection: connectionStatus,
                tables: null,
                migrations: null
            };
        }
        // Check tables (users)
        let userTableStatus = { success: true };
        try {
            await prismaClient_1.default.user.findFirst();
        }
        catch (error) {
            userTableStatus = {
                success: false,
                error: error.message
            };
        }
        // Check migrations table
        let migrationsStatus = { success: true };
        try {
            await prismaClient_1.default.$queryRaw `SELECT * FROM "_prisma_migrations" LIMIT 1`;
        }
        catch (error) {
            migrationsStatus = {
                success: false,
                error: error.message
            };
        }
        return {
            success: connectionStatus.success && userTableStatus.success && migrationsStatus.success,
            connection: connectionStatus,
            tables: {
                users: userTableStatus
            },
            migrations: migrationsStatus
        };
    }
    catch (error) {
        return {
            success: false,
            message: 'Comprehensive health check failed',
            error: error.message,
            connection: {
                success: false,
                message: 'Health check error',
                responseTime: '0ms'
            },
            tables: null,
            migrations: null
        };
    }
}
//# sourceMappingURL=dbHealthCheck.js.map