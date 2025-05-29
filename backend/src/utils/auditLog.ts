/**
 * Audit Log Utility
 * Logs important user and system actions for security and compliance
 */

import prisma from './prismaClient';
import { v4 as uuidv4 } from 'uuid';
import { createLogger } from './logger';

// Create logger
const logger = createLogger('AuditLog');

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
async function createAuditLog(entry: AuditLogEntry): Promise<any> {
  try {
    return await prisma.auditLog.create({
      data: {
        userId: entry.userId,
        action: entry.action,
        details: entry.details || {},
        ipAddress: entry.ipAddress || null,
        userAgent: entry.userAgent || null,
        timestamp: new Date()
      }
    });
  } catch (error) {
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
async function getUserAuditLogs(
  userId: string,
  filters: {
    action?: string;
    startDate?: Date;
    endDate?: Date;
    limit?: number;
    offset?: number;
  } = {}
): Promise<any[]> {
  try {
    return await prisma.auditLog.findMany({
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
  } catch (error) {
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
async function getResourceAuditLogs(
  resourceType: string,
  resourceId: string
): Promise<any[]> {
  try {
    return await prisma.auditLog.findMany({
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
  } catch (error) {
    console.error('Error fetching resource audit logs:', error);
    return [];
  }
}

export {
  AuditLogEntry,
  createAuditLog,
  getUserAuditLogs,
  getResourceAuditLogs
}; 