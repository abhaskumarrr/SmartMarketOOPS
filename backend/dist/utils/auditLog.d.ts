/**
 * Audit Log Utility
 * Logs important user and system actions for security and compliance
 */
/**
 * Audit log entry interface
 */
interface AuditLogEntry {
    userId: string;
    action: string;
    details?: any;
    ipAddress?: string;
    userAgent?: string;
}
/**
 * Create a new audit log entry
 * @param {AuditLogEntry} entry - The audit log entry to create
 * @returns {Promise<any>} The created audit log
 */
declare function createAuditLog(entry: AuditLogEntry): Promise<any>;
/**
 * Get audit logs for a user
 * @param {string} userId - User ID to get logs for
 * @param {object} filters - Additional filters
 * @returns {Promise<any[]>} Array of audit logs
 */
declare function getUserAuditLogs(userId: string, filters?: {
    action?: string;
    startDate?: Date;
    endDate?: Date;
    limit?: number;
    offset?: number;
}): Promise<any[]>;
/**
 * Get audit logs for a specific resource
 * @param {string} resourceType - Type of resource (e.g., 'api_key', 'user')
 * @param {string} resourceId - ID of the resource
 * @returns {Promise<any[]>} Array of audit logs
 */
declare function getResourceAuditLogs(resourceType: string, resourceId: string): Promise<any[]>;
export { AuditLogEntry, createAuditLog, getUserAuditLogs, getResourceAuditLogs };
