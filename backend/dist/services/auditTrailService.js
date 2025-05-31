"use strict";
/**
 * Audit Trail Service
 * Handles audit logs for user actions and system events
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.auditTrailService = exports.AuditTrailService = exports.LogSeverity = void 0;
const uuid_1 = require("uuid");
const prismaClient_1 = __importDefault(require("../utils/prismaClient"));
const logger_1 = require("../utils/logger");
const auditLog_1 = require("../types/auditLog");
// Create logger
const logger = (0, logger_1.createLogger)('AuditTrailService');
/**
 * Log severity levels
 */
var LogSeverity;
(function (LogSeverity) {
    LogSeverity["DEBUG"] = "DEBUG";
    LogSeverity["INFO"] = "INFO";
    LogSeverity["WARN"] = "WARN";
    LogSeverity["ERROR"] = "ERROR";
    LogSeverity["CRITICAL"] = "CRITICAL";
})(LogSeverity || (exports.LogSeverity = LogSeverity = {}));
/**
 * Audit Trail Service class
 * Handles comprehensive audit trail functionality for tracking system actions
 */
class AuditTrailService {
    /**
     * Create a new audit trail
     * @param data - Audit trail creation request
     * @returns Created audit trail
     */
    async createAuditTrail(data) {
        try {
            logger.info('Creating audit trail', { data });
            // Set default values if not provided
            const tags = data.tags || [];
            // Create audit trail in database
            const auditTrail = await prismaClient_1.default.auditTrail.create({
                data: {
                    id: (0, uuid_1.v4)(),
                    trailType: data.trailType,
                    entityId: data.entityId,
                    entityType: data.entityType,
                    startTime: new Date(),
                    status: auditLog_1.AuditTrailStatus.ACTIVE,
                    summary: data.summary,
                    userId: data.userId,
                    orderId: data.orderId,
                    tags,
                    metadata: data.metadata
                }
            });
            logger.info('Audit trail created', { id: auditTrail.id });
            // Map to response format
            return this.mapAuditTrailFromDb(auditTrail);
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            logger.error(`Failed to create audit trail: ${errorMessage}`, { error, data });
            throw error;
        }
    }
    /**
     * Get an audit trail by ID
     * @param id - Audit trail ID
     * @param includeEvents - Whether to include events
     * @param includeDecisionLogs - Whether to include decision logs
     * @returns Audit trail
     */
    async getAuditTrail(id, includeEvents = false, includeDecisionLogs = false) {
        try {
            logger.info(`Getting audit trail ${id}`, { includeEvents, includeDecisionLogs });
            // Build include object
            const include = {};
            if (includeEvents) {
                include.events = {
                    orderBy: {
                        timestamp: 'asc'
                    }
                };
            }
            if (includeDecisionLogs) {
                include.decisionLogs = {
                    orderBy: {
                        timestamp: 'asc'
                    }
                };
            }
            // Get audit trail from database
            const auditTrail = await prismaClient_1.default.auditTrail.findUnique({
                where: { id },
                include: Object.keys(include).length > 0 ? include : undefined
            });
            if (!auditTrail) {
                throw new Error(`Audit trail ${id} not found`);
            }
            logger.info('Audit trail retrieved', { id });
            // Map to response format
            return this.mapAuditTrailFromDb(auditTrail);
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            logger.error(`Failed to get audit trail ${id}: ${errorMessage}`, { error, id });
            throw error;
        }
    }
    /**
     * Update an audit trail
     * @param id - Audit trail ID
     * @param data - Updated audit trail data
     * @returns Updated audit trail
     */
    async updateAuditTrail(id, data) {
        try {
            logger.info(`Updating audit trail ${id}`, { data });
            // Update audit trail in database
            const auditTrail = await prismaClient_1.default.auditTrail.update({
                where: { id },
                data: {
                    status: data.status,
                    endTime: data.endTime ? new Date(data.endTime) : undefined,
                    summary: data.summary,
                    tags: data.tags,
                    metadata: data.metadata
                }
            });
            logger.info('Audit trail updated', { id });
            // Map to response format
            return this.mapAuditTrailFromDb(auditTrail);
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            logger.error(`Failed to update audit trail ${id}: ${errorMessage}`, { error, id, data });
            throw error;
        }
    }
    /**
     * Complete an audit trail
     * @param id - Audit trail ID
     * @param status - Status to set (default: COMPLETED)
     * @returns Completed audit trail
     */
    async completeAuditTrail(id, status = auditLog_1.AuditTrailStatus.COMPLETED) {
        try {
            logger.info(`Completing audit trail ${id}`, { status });
            // Update audit trail in database
            const auditTrail = await prismaClient_1.default.auditTrail.update({
                where: { id },
                data: {
                    status,
                    endTime: new Date()
                }
            });
            logger.info('Audit trail completed', { id, status });
            // Map to response format
            return this.mapAuditTrailFromDb(auditTrail);
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            logger.error(`Failed to complete audit trail ${id}: ${errorMessage}`, { error, id, status });
            throw error;
        }
    }
    /**
     * Delete an audit trail
     * @param id - Audit trail ID
     * @returns Whether the deletion was successful
     */
    async deleteAuditTrail(id) {
        try {
            logger.info(`Deleting audit trail ${id}`);
            // Delete audit trail from database
            await prismaClient_1.default.auditTrail.delete({
                where: { id }
            });
            logger.info('Audit trail deleted', { id });
            return true;
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            logger.error(`Failed to delete audit trail ${id}: ${errorMessage}`, { error, id });
            throw error;
        }
    }
    /**
     * Create an audit event
     * @param data - Audit event creation request
     * @returns Created audit event
     */
    async createAuditEvent(data) {
        try {
            logger.info('Creating audit event', { data });
            // Create audit event in database
            const auditEvent = await prismaClient_1.default.auditEvent.create({
                data: {
                    id: (0, uuid_1.v4)(),
                    auditTrailId: data.auditTrailId,
                    timestamp: new Date(),
                    eventType: data.eventType,
                    component: data.component,
                    action: data.action,
                    status: data.status,
                    details: data.details,
                    dataBefore: data.dataBefore,
                    dataAfter: data.dataAfter,
                    metadata: data.metadata
                }
            });
            logger.info('Audit event created', { id: auditEvent.id });
            // Map to response format
            return this.mapAuditEventFromDb(auditEvent);
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            logger.error(`Failed to create audit event: ${errorMessage}`, { error, data });
            throw error;
        }
    }
    /**
     * Get an audit event by ID
     * @param id - Audit event ID
     * @returns Audit event
     */
    async getAuditEvent(id) {
        try {
            logger.info(`Getting audit event ${id}`);
            // Get audit event from database
            const auditEvent = await prismaClient_1.default.auditEvent.findUnique({
                where: { id }
            });
            if (!auditEvent) {
                throw new Error(`Audit event ${id} not found`);
            }
            logger.info('Audit event retrieved', { id });
            // Map to response format
            return this.mapAuditEventFromDb(auditEvent);
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            logger.error(`Failed to get audit event ${id}: ${errorMessage}`, { error, id });
            throw error;
        }
    }
    /**
     * Query audit trails based on parameters
     * @param params - Query parameters
     * @returns Matching audit trails
     */
    async queryAuditTrails(params) {
        try {
            logger.info('Querying audit trails', { params });
            // Build where clause based on parameters
            const where = {};
            if (params.startDate || params.endDate) {
                where.startTime = {};
                if (params.startDate) {
                    where.startTime.gte = new Date(params.startDate);
                }
                if (params.endDate) {
                    where.startTime.lte = new Date(params.endDate);
                }
            }
            if (params.trailType) {
                where.trailType = params.trailType;
            }
            if (params.entityId) {
                where.entityId = params.entityId;
            }
            if (params.entityType) {
                where.entityType = params.entityType;
            }
            if (params.status) {
                where.status = params.status;
            }
            if (params.userId) {
                where.userId = params.userId;
            }
            if (params.orderId) {
                where.orderId = params.orderId;
            }
            if (params.tags && params.tags.length > 0) {
                where.tags = {
                    hasSome: params.tags
                };
            }
            // Build include object
            const include = {};
            if (params.includeEvents) {
                include.events = {
                    orderBy: {
                        timestamp: 'asc'
                    }
                };
            }
            if (params.includeDecisionLogs) {
                include.decisionLogs = {
                    orderBy: {
                        timestamp: 'asc'
                    }
                };
            }
            // Determine sorting
            const orderBy = {};
            if (params.sortBy) {
                orderBy[params.sortBy] = params.sortDirection || 'desc';
            }
            else {
                orderBy.startTime = 'desc'; // Default sort by start time
            }
            // Query audit trails from database
            const auditTrails = await prismaClient_1.default.auditTrail.findMany({
                where,
                include: Object.keys(include).length > 0 ? include : undefined,
                orderBy,
                skip: params.offset || 0,
                take: params.limit || 100
            });
            logger.info(`Found ${auditTrails.length} audit trails`);
            // Map to response format
            return auditTrails.map((trail) => this.mapAuditTrailFromDb(trail));
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            logger.error(`Failed to query audit trails: ${errorMessage}`, { error, params });
            throw error;
        }
    }
    /**
     * Count audit trails based on parameters
     * @param params - Query parameters
     * @returns Number of matching audit trails
     */
    async countAuditTrails(params) {
        try {
            logger.info('Counting audit trails', { params });
            // Build where clause based on parameters
            const where = {};
            if (params.startDate || params.endDate) {
                where.startTime = {};
                if (params.startDate) {
                    where.startTime.gte = new Date(params.startDate);
                }
                if (params.endDate) {
                    where.startTime.lte = new Date(params.endDate);
                }
            }
            if (params.trailType) {
                where.trailType = params.trailType;
            }
            if (params.entityId) {
                where.entityId = params.entityId;
            }
            if (params.entityType) {
                where.entityType = params.entityType;
            }
            if (params.status) {
                where.status = params.status;
            }
            if (params.userId) {
                where.userId = params.userId;
            }
            if (params.orderId) {
                where.orderId = params.orderId;
            }
            if (params.tags && params.tags.length > 0) {
                where.tags = {
                    hasSome: params.tags
                };
            }
            // Count audit trails from database
            const count = await prismaClient_1.default.auditTrail.count({
                where
            });
            logger.info(`Counted ${count} audit trails`);
            return count;
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            logger.error(`Failed to count audit trails: ${errorMessage}`, { error, params });
            throw error;
        }
    }
    /**
     * Query audit events based on parameters
     * @param params - Query parameters
     * @returns Matching audit events
     */
    async queryAuditEvents(params) {
        try {
            logger.info('Querying audit events', { params });
            // Build where clause based on parameters
            const where = {};
            if (params.auditTrailId) {
                where.auditTrailId = params.auditTrailId;
            }
            if (params.startDate || params.endDate) {
                where.timestamp = {};
                if (params.startDate) {
                    where.timestamp.gte = new Date(params.startDate);
                }
                if (params.endDate) {
                    where.timestamp.lte = new Date(params.endDate);
                }
            }
            if (params.eventType) {
                where.eventType = params.eventType;
            }
            if (params.component) {
                where.component = params.component;
            }
            if (params.status) {
                where.status = params.status;
            }
            // Determine sorting
            const orderBy = {};
            if (params.sortBy) {
                orderBy[params.sortBy] = params.sortDirection || 'desc';
            }
            else {
                orderBy.timestamp = 'desc'; // Default sort by timestamp
            }
            // Query audit events from database
            const auditEvents = await prismaClient_1.default.auditEvent.findMany({
                where,
                orderBy,
                skip: params.offset || 0,
                take: params.limit || 100
            });
            logger.info(`Found ${auditEvents.length} audit events`);
            // Map to response format
            return auditEvents.map((event) => this.mapAuditEventFromDb(event));
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            logger.error(`Failed to query audit events: ${errorMessage}`, { error, params });
            throw error;
        }
    }
    /**
     * Count audit events based on parameters
     * @param params - Query parameters
     * @returns Number of matching audit events
     */
    async countAuditEvents(params) {
        try {
            logger.info('Counting audit events', { params });
            // Build where clause based on parameters
            const where = {};
            if (params.auditTrailId) {
                where.auditTrailId = params.auditTrailId;
            }
            if (params.startDate || params.endDate) {
                where.timestamp = {};
                if (params.startDate) {
                    where.timestamp.gte = new Date(params.startDate);
                }
                if (params.endDate) {
                    where.timestamp.lte = new Date(params.endDate);
                }
            }
            if (params.eventType) {
                where.eventType = params.eventType;
            }
            if (params.component) {
                where.component = params.component;
            }
            if (params.status) {
                where.status = params.status;
            }
            // Count audit events from database
            const count = await prismaClient_1.default.auditEvent.count({
                where
            });
            logger.info(`Counted ${count} audit events`);
            return count;
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            logger.error(`Failed to count audit events: ${errorMessage}`, { error, params });
            throw error;
        }
    }
    /**
     * Map an audit trail from database to response format
     * @param dbTrail - Audit trail from database
     * @returns Mapped audit trail
     */
    mapAuditTrailFromDb(dbTrail) {
        const result = {
            id: dbTrail.id,
            trailType: dbTrail.trailType,
            entityId: dbTrail.entityId,
            entityType: dbTrail.entityType,
            startTime: dbTrail.startTime.toISOString(),
            endTime: dbTrail.endTime ? dbTrail.endTime.toISOString() : undefined,
            status: dbTrail.status,
            summary: dbTrail.summary,
            userId: dbTrail.userId,
            orderId: dbTrail.orderId,
            tags: dbTrail.tags,
            metadata: dbTrail.metadata
        };
        // Add events if present
        if (dbTrail.events) {
            result.events = dbTrail.events.map((event) => this.mapAuditEventFromDb(event));
        }
        // Add decision logs if present
        if (dbTrail.decisionLogs) {
            result.decisionLogs = dbTrail.decisionLogs.map((log) => ({
                id: log.id,
                timestamp: log.timestamp.toISOString(),
                source: log.source,
                actionType: log.actionType,
                decision: log.decision,
                reasonCode: log.reasonCode,
                reasonDetails: log.reasonDetails,
                confidence: log.confidence,
                dataSnapshot: log.dataSnapshot,
                parameters: log.parameters,
                modelVersion: log.modelVersion,
                userId: log.userId,
                strategyId: log.strategyId,
                botId: log.botId,
                signalId: log.signalId,
                orderId: log.orderId,
                symbol: log.symbol,
                outcome: log.outcome,
                outcomeDetails: log.outcomeDetails,
                pnl: log.pnl,
                evaluatedAt: log.evaluatedAt ? log.evaluatedAt.toISOString() : undefined,
                tags: log.tags,
                importance: log.importance,
                notes: log.notes,
                auditTrailId: log.auditTrailId
            }));
        }
        return result;
    }
    /**
     * Map an audit event from database to response format
     * @param dbEvent - Audit event from database
     * @returns Mapped audit event
     */
    mapAuditEventFromDb(dbEvent) {
        return {
            id: dbEvent.id,
            auditTrailId: dbEvent.auditTrailId,
            timestamp: dbEvent.timestamp.toISOString(),
            eventType: dbEvent.eventType,
            component: dbEvent.component,
            action: dbEvent.action,
            status: dbEvent.status,
            details: dbEvent.details,
            dataBefore: dbEvent.dataBefore,
            dataAfter: dbEvent.dataAfter,
            metadata: dbEvent.metadata
        };
    }
}
exports.AuditTrailService = AuditTrailService;
// Export singleton instance
exports.auditTrailService = new AuditTrailService();
// Export default for dependency injection
exports.default = exports.auditTrailService;
//# sourceMappingURL=auditTrailService.js.map