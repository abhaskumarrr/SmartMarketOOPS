"use strict";
/**
 * Decision Log Service
 * Logs important decisions made by the system or users
 * Used for auditing, debugging, and analyzing trading strategy behavior
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.createDecisionLog = exports.decisionLogService = exports.DecisionLogService = void 0;
const prismaClient_1 = __importDefault(require("../utils/prismaClient"));
const uuid_1 = require("uuid");
const logger_1 = require("../utils/logger");
// Create logger
const logger = (0, logger_1.createLogger)('DecisionLogService');
/**
 * Decision Log Service class
 * Handles logging of trading decisions and actions
 */
class DecisionLogService {
    /**
     * Create a new decision log
     * @param data - Decision log creation request
     * @returns Created decision log
     */
    async createDecisionLog(data) {
        try {
            logger.info('Creating decision log', { data });
            // Set default values if not provided
            const tags = data.tags || [];
            const importance = data.importance || 'NORMAL';
            // Create decision log in database
            const decisionLog = await prismaClient_1.default.decisionLog.create({
                data: {
                    id: (0, uuid_1.v4)(),
                    timestamp: new Date(),
                    source: data.source,
                    actionType: data.actionType,
                    decision: data.decision,
                    reasonCode: data.reasonCode,
                    reasonDetails: data.reasonDetails,
                    confidence: data.confidence,
                    dataSnapshot: data.dataSnapshot,
                    parameters: data.parameters,
                    modelVersion: data.modelVersion,
                    userId: data.userId,
                    strategyId: data.strategyId,
                    botId: data.botId,
                    signalId: data.signalId,
                    orderId: data.orderId,
                    symbol: data.symbol,
                    tags,
                    importance,
                    notes: data.notes,
                    auditTrailId: data.auditTrailId
                }
            });
            logger.info('Decision log created', { id: decisionLog.id });
            // Map to response format
            return this.mapDecisionLogFromDb(decisionLog);
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            logger.error(`Failed to create decision log: ${errorMessage}`, { error, data });
            throw error;
        }
    }
    /**
     * Get a decision log by ID
     * @param id - Decision log ID
     * @returns Decision log
     */
    async getDecisionLog(id) {
        try {
            logger.info(`Getting decision log ${id}`);
            // Get decision log from database
            const decisionLog = await prismaClient_1.default.decisionLog.findUnique({
                where: { id }
            });
            if (!decisionLog) {
                throw new Error(`Decision log ${id} not found`);
            }
            logger.info('Decision log retrieved', { id });
            // Map to response format
            return this.mapDecisionLogFromDb(decisionLog);
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            logger.error(`Failed to get decision log ${id}: ${errorMessage}`, { error, id });
            throw error;
        }
    }
    /**
     * Update a decision log
     * @param id - Decision log ID
     * @param data - Updated decision log data
     * @returns Updated decision log
     */
    async updateDecisionLog(id, data) {
        try {
            logger.info(`Updating decision log ${id}`, { data });
            // Update decision log in database
            const decisionLog = await prismaClient_1.default.decisionLog.update({
                where: { id },
                data: {
                    outcome: data.outcome,
                    outcomeDetails: data.outcomeDetails,
                    pnl: data.pnl,
                    evaluatedAt: data.evaluatedAt ? new Date(data.evaluatedAt) : undefined,
                    tags: data.tags,
                    importance: data.importance,
                    notes: data.notes
                }
            });
            logger.info('Decision log updated', { id });
            // Map to response format
            return this.mapDecisionLogFromDb(decisionLog);
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            logger.error(`Failed to update decision log ${id}: ${errorMessage}`, { error, id, data });
            throw error;
        }
    }
    /**
     * Delete a decision log
     * @param id - Decision log ID
     * @returns Whether the deletion was successful
     */
    async deleteDecisionLog(id) {
        try {
            logger.info(`Deleting decision log ${id}`);
            // Delete decision log from database
            await prismaClient_1.default.decisionLog.delete({
                where: { id }
            });
            logger.info('Decision log deleted', { id });
            return true;
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            logger.error(`Failed to delete decision log ${id}: ${errorMessage}`, { error, id });
            throw error;
        }
    }
    /**
     * Query decision logs based on parameters
     * @param params - Query parameters
     * @returns Matching decision logs
     */
    async queryDecisionLogs(params) {
        try {
            logger.info('Querying decision logs', { params });
            // Build where clause based on parameters
            const where = {};
            if (params.startDate || params.endDate) {
                where.timestamp = {};
                if (params.startDate) {
                    where.timestamp.gte = new Date(params.startDate);
                }
                if (params.endDate) {
                    where.timestamp.lte = new Date(params.endDate);
                }
            }
            if (params.source) {
                where.source = params.source;
            }
            if (params.actionType) {
                where.actionType = params.actionType;
            }
            if (params.userId) {
                where.userId = params.userId;
            }
            if (params.strategyId) {
                where.strategyId = params.strategyId;
            }
            if (params.botId) {
                where.botId = params.botId;
            }
            if (params.signalId) {
                where.signalId = params.signalId;
            }
            if (params.orderId) {
                where.orderId = params.orderId;
            }
            if (params.symbol) {
                where.symbol = params.symbol;
            }
            if (params.outcome) {
                where.outcome = params.outcome;
            }
            if (params.importance) {
                where.importance = params.importance;
            }
            if (params.tags && params.tags.length > 0) {
                where.tags = {
                    hasSome: params.tags
                };
            }
            // Determine sorting
            const orderBy = {};
            if (params.sortBy) {
                orderBy[params.sortBy] = params.sortDirection || 'desc';
            }
            else {
                orderBy.timestamp = 'desc'; // Default sort by timestamp
            }
            // Query decision logs from database
            const decisionLogs = await prismaClient_1.default.decisionLog.findMany({
                where,
                orderBy,
                skip: params.offset || 0,
                take: params.limit || 100
            });
            logger.info(`Found ${decisionLogs.length} decision logs`);
            // Map to response format
            return decisionLogs.map((log) => this.mapDecisionLogFromDb(log));
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            logger.error(`Failed to query decision logs: ${errorMessage}`, { error, params });
            throw error;
        }
    }
    /**
     * Count decision logs based on parameters
     * @param params - Query parameters
     * @returns Number of matching decision logs
     */
    async countDecisionLogs(params) {
        try {
            logger.info('Counting decision logs', { params });
            // Build where clause based on parameters
            const where = {};
            if (params.startDate || params.endDate) {
                where.timestamp = {};
                if (params.startDate) {
                    where.timestamp.gte = new Date(params.startDate);
                }
                if (params.endDate) {
                    where.timestamp.lte = new Date(params.endDate);
                }
            }
            if (params.source) {
                where.source = params.source;
            }
            if (params.actionType) {
                where.actionType = params.actionType;
            }
            if (params.userId) {
                where.userId = params.userId;
            }
            if (params.strategyId) {
                where.strategyId = params.strategyId;
            }
            if (params.botId) {
                where.botId = params.botId;
            }
            if (params.signalId) {
                where.signalId = params.signalId;
            }
            if (params.orderId) {
                where.orderId = params.orderId;
            }
            if (params.symbol) {
                where.symbol = params.symbol;
            }
            if (params.outcome) {
                where.outcome = params.outcome;
            }
            if (params.importance) {
                where.importance = params.importance;
            }
            if (params.tags && params.tags.length > 0) {
                where.tags = {
                    hasSome: params.tags
                };
            }
            // Count decision logs from database
            const count = await prismaClient_1.default.decisionLog.count({
                where
            });
            logger.info(`Counted ${count} decision logs`);
            return count;
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            logger.error(`Failed to count decision logs: ${errorMessage}`, { error, params });
            throw error;
        }
    }
    /**
     * Map a decision log from database to response format
     * @param dbLog - Decision log from database
     * @returns Mapped decision log
     */
    mapDecisionLogFromDb(dbLog) {
        return {
            id: dbLog.id,
            timestamp: dbLog.timestamp.toISOString(),
            source: dbLog.source,
            actionType: dbLog.actionType,
            decision: dbLog.decision,
            reasonCode: dbLog.reasonCode,
            reasonDetails: dbLog.reasonDetails,
            confidence: dbLog.confidence,
            dataSnapshot: dbLog.dataSnapshot,
            parameters: dbLog.parameters,
            modelVersion: dbLog.modelVersion,
            userId: dbLog.userId,
            strategyId: dbLog.strategyId,
            botId: dbLog.botId,
            signalId: dbLog.signalId,
            orderId: dbLog.orderId,
            symbol: dbLog.symbol,
            outcome: dbLog.outcome,
            outcomeDetails: dbLog.outcomeDetails,
            pnl: dbLog.pnl,
            evaluatedAt: dbLog.evaluatedAt ? dbLog.evaluatedAt.toISOString() : undefined,
            tags: dbLog.tags,
            importance: dbLog.importance,
            notes: dbLog.notes,
            auditTrailId: dbLog.auditTrailId
        };
    }
}
exports.DecisionLogService = DecisionLogService;
// Export singleton instance
exports.decisionLogService = new DecisionLogService();
// Export default for dependency injection
exports.default = exports.decisionLogService;
/**
 * Standalone function to create a decision log
 * Wrapper around the decisionLogService.createDecisionLog method
 * @param data - Decision log entry data
 * @returns Promise resolving to the created decision log
 */
const createDecisionLog = (data) => {
    return exports.decisionLogService.createDecisionLog({
        source: data.source,
        actionType: data.actionType,
        decision: data.decision,
        reasonDetails: data.reasonDetails,
        userId: data.userId,
        botId: data.botId,
        strategyId: data.strategyId,
        symbol: data.symbol,
        orderId: data.orderId,
        importance: data.importance,
        tags: data.tags
    });
};
exports.createDecisionLog = createDecisionLog;
//# sourceMappingURL=decisionLogService.js.map