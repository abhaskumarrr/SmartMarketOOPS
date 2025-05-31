"use strict";
/**
 * Audit Log Utility
 * Logs important user and system actions for security and compliance
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.createAuditLog = createAuditLog;
exports.getUserAuditLogs = getUserAuditLogs;
exports.getResourceAuditLogs = getResourceAuditLogs;
const prismaClient_1 = __importDefault(require("./prismaClient"));
const logger_1 = require("./logger");
// Create logger
const logger = (0, logger_1.createLogger)('AuditLog');
/**
 * Create a new audit log entry
 * @param {AuditLogEntry} entry - The audit log entry to create
 * @returns {Promise<any>} The created audit log
 */
async function createAuditLog(entry) {
    try {
        return await prismaClient_1.default.auditLog.create({
            data: {
                userId: entry.userId,
                action: entry.action,
                details: entry.details || {},
                ipAddress: entry.ipAddress || null,
                userAgent: entry.userAgent || null,
                timestamp: new Date()
            }
        });
    }
    catch (error) {
        console.error('Error creating audit log:', error);
        // Don't throw - audit logs should never block operation
        return null;
    }
}
/**
 * Get audit logs for a user
 * @param {string} userId - User ID to get logs for
 * @param {object} filters - Additional filters
 * @returns {Promise<any[]>} Array of audit logs
 */
async function getUserAuditLogs(userId, filters = {}) {
    try {
        return await prismaClient_1.default.auditLog.findMany({
            where: {
                userId,
                action: filters.action ? { startsWith: filters.action } : undefined,
                timestamp: {
                    gte: filters.startDate,
                    lte: filters.endDate
                }
            },
            orderBy: {
                timestamp: 'desc'
            },
            skip: filters.offset || 0,
            take: filters.limit || 50
        });
    }
    catch (error) {
        console.error('Error fetching audit logs:', error);
        return [];
    }
}
/**
 * Get audit logs for a specific resource
 * @param {string} resourceType - Type of resource (e.g., 'api_key', 'user')
 * @param {string} resourceId - ID of the resource
 * @returns {Promise<any[]>} Array of audit logs
 */
async function getResourceAuditLogs(resourceType, resourceId) {
    try {
        return await prismaClient_1.default.auditLog.findMany({
            where: {
                action: {
                    startsWith: `${resourceType}.`
                },
                details: {
                    path: [`${resourceType}Id`],
                    equals: resourceId
                }
            },
            orderBy: {
                timestamp: 'desc'
            },
            take: 50
        });
    }
    catch (error) {
        console.error('Error fetching resource audit logs:', error);
        return [];
    }
}
//# sourceMappingURL=auditLog.js.map