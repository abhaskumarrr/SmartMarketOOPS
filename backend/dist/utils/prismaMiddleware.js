"use strict";
/**
 * Prisma Middleware Utilities
 * Middleware for logging and validation of database operations
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.applyMiddleware = applyMiddleware;
const fs_1 = __importDefault(require("fs"));
const path_1 = __importDefault(require("path"));
const fs_2 = require("fs");
// Log directory
const LOG_DIR = path_1.default.join(__dirname, '../../logs');
const QUERY_LOG_PATH = path_1.default.join(LOG_DIR, 'prisma-queries.log');
// Ensure log directory exists
if (!fs_1.default.existsSync(LOG_DIR)) {
    fs_1.default.mkdirSync(LOG_DIR, { recursive: true });
}
// Create write stream for query logging
const queryLogStream = (0, fs_2.createWriteStream)(QUERY_LOG_PATH, { flags: 'a' });
/**
 * Apply middleware to a Prisma client instance
 * @param {PrismaClient} prisma - Prisma client instance
 * @returns {PrismaClient} - Prisma client with middleware applied
 */
function applyMiddleware(prisma) {
    // Query logging middleware
    prisma.$use(async (params, next) => {
        const before = Date.now();
        const result = await next(params);
        const after = Date.now();
        const duration = after - before;
        // Log query information
        const log = `${new Date().toISOString()} | ${params.model}.${params.action} | ${duration}ms\n`;
        queryLogStream.write(log);
        // Also log to console in development
        if (process.env.NODE_ENV === 'development') {
            console.log(`Prisma Query: ${params.model}.${params.action} (${duration}ms)`);
        }
        return result;
    });
    // Data validation middleware for User model
    prisma.$use(async (params, next) => {
        if (params.model === 'User') {
            if (params.action === 'create' || params.action === 'update') {
                validateUserData(params.args.data);
            }
        }
        return next(params);
    });
    // Data validation middleware for Bot model
    prisma.$use(async (params, next) => {
        if (params.model === 'Bot') {
            if (params.action === 'create' || params.action === 'update') {
                validateBotData(params.args.data);
            }
        }
        return next(params);
    });
    // Soft delete middleware for specific models
    prisma.$use(async (params, next) => {
        if (params.action === 'delete' && ['User', 'Bot'].includes(params.model)) {
            // Convert delete operations to updates with a deletedAt timestamp
            params.action = 'update';
            params.args.data = { deletedAt: new Date() };
            // Log soft delete
            console.log(`Soft delete: ${params.model} ${JSON.stringify(params.args.where)}`);
        }
        return next(params);
    });
    // Automatic timestamp updater
    prisma.$use(async (params, next) => {
        if (params.action === 'update' && params.model) {
            if (!params.args.data.updatedAt) {
                params.args.data.updatedAt = new Date();
            }
        }
        return next(params);
    });
    // Return the prisma client with middleware applied
    return prisma;
}
/**
 * Validate user data
 * @param {UserData} data - User data to validate
 * @throws {Error} If validation fails
 */
function validateUserData(data) {
    // Validate email
    if (data.email && !isValidEmail(data.email)) {
        throw new Error('Invalid email format');
    }
    // Validate password if present
    if (data.password && data.password.length < 8) {
        throw new Error('Password must be at least 8 characters long');
    }
}
/**
 * Validate bot data
 * @param {BotData} data - Bot data to validate
 * @throws {Error} If validation fails
 */
function validateBotData(data) {
    // Validate symbol format
    if (data.symbol && !isValidSymbol(data.symbol)) {
        throw new Error('Invalid trading symbol format');
    }
    // Validate timeframe
    if (data.timeframe && !isValidTimeframe(data.timeframe)) {
        throw new Error('Invalid timeframe. Must be one of: 1m, 5m, 15m, 30m, 1h, 4h, 1d');
    }
}
/**
 * Check if email is valid
 * @param {string} email - Email to validate
 * @returns {boolean} - True if valid
 */
function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}
/**
 * Check if trading symbol is valid
 * @param {string} symbol - Symbol to validate
 * @returns {boolean} - True if valid
 */
function isValidSymbol(symbol) {
    // Simple validation: letters followed by letters (e.g. BTCUSDT)
    const symbolRegex = /^[A-Z0-9]+[A-Z0-9]+$/;
    return symbolRegex.test(symbol);
}
/**
 * Check if timeframe is valid
 * @param {string} timeframe - Timeframe to validate
 * @returns {boolean} - True if valid
 */
function isValidTimeframe(timeframe) {
    const validTimeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d'];
    return validTimeframes.includes(timeframe);
}
//# sourceMappingURL=prismaMiddleware.js.map