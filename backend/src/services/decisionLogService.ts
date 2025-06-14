/**
 * Decision Log Service
 * Logs important decisions made by the system or users
 * Used for auditing, debugging, and analyzing trading strategy behavior
 */

import prisma from '../utils/prismaClient';
import prismaReadOnly from '../config/prisma-readonly';
import { v4 as uuidv4 } from 'uuid';
import { createLogger } from '../utils/logger';
import {
  DecisionLog,
  IDecisionLogService,
  DecisionLogQueryParams,
  CreateDecisionLogRequest,
  DecisionOutcome
} from '../types/auditLog';

// Create logger
const logger = createLogger('DecisionLogService');

/**
 * Decision importance levels
 */
export type DecisionImportance = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';

/**
 * Decision source types
 */
export type DecisionSource = 'System' | 'User' | 'Strategy' | 'ML' | 'RiskManagement';

/**
 * Decision action types
 */
export type DecisionActionType = 
  | 'OrderExecution' 
  | 'RiskAdjustment' 
  | 'StrategySwitch' 
  | 'PositionClose' 
  | 'BotControl'
  | 'MarketAnalysis'
  | 'SystemAlert';

/**
 * Decision log entry
 */
export interface DecisionLogEntry {
  source: DecisionSource;
  actionType: DecisionActionType;
  decision: string;
  reasonDetails?: string;
  userId?: string;
  botId?: string;
  strategyId?: string;
  symbol?: string;
  orderId?: string;
  positionId?: string;
  importance: DecisionImportance;
  metadata?: Record<string, any>;
  tags?: string[];
}

/**
 * Decision Log Service class
 * Handles logging of trading decisions and actions
 */
export class DecisionLogService implements IDecisionLogService {
  /**
   * Create a new decision log
   * @param data - Decision log creation request
   * @returns Created decision log
   */
  async createDecisionLog(data: CreateDecisionLogRequest): Promise<DecisionLog> {
    try {
      logger.info('Creating decision log', { data });
      
      const createData: any = { ...data };
      if (!createData.tags) createData.tags = [];
      if (!createData.importance) createData.importance = 'NORMAL';
      
      const decisionLog = await prisma.decisionLog.create({
        data: {
          id: uuidv4(),
          timestamp: new Date(),
          ...createData,
        }
      });
      
      logger.info('Decision log created', { id: decisionLog.id });
      
      // Map to response format
      return this.mapDecisionLogFromDb(decisionLog);
    } catch (error) {
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
  async getDecisionLog(id: string): Promise<DecisionLog> {
    try {
      logger.info(`Getting decision log ${id}`);
      
      // Get decision log from read-only database
      const decisionLog = await prismaReadOnly.decisionLog.findUnique({
        where: { id }
      });
      
      if (!decisionLog) {
        throw new Error(`Decision log ${id} not found`);
      }
      
      logger.info('Decision log retrieved', { id });
      
      // Map to response format
      return this.mapDecisionLogFromDb(decisionLog);
    } catch (error) {
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
  async updateDecisionLog(id: string, data: Partial<DecisionLog>): Promise<DecisionLog> {
    try {
      logger.info(`Updating decision log ${id}`, { data });
      
      const updateData: any = { ...data };
      if (updateData.evaluatedAt) {
        updateData.evaluatedAt = new Date(updateData.evaluatedAt);
      }

      const decisionLog = await prisma.decisionLog.update({
        where: { id },
        data: updateData,
      });
      
      logger.info('Decision log updated', { id });
      
      // Map to response format
      return this.mapDecisionLogFromDb(decisionLog);
    } catch (error) {
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
  async deleteDecisionLog(id: string): Promise<boolean> {
    try {
      logger.info(`Deleting decision log ${id}`);
      
      // Delete decision log from database
      await prisma.decisionLog.delete({
        where: { id }
      });
      
      logger.info('Decision log deleted', { id });
      
      return true;
    } catch (error) {
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
  async queryDecisionLogs(params: DecisionLogQueryParams): Promise<DecisionLog[]> {
    try {
      logger.info('Querying decision logs', { params });
      
      // Build where clause based on parameters
      const where: any = {};
      
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
      const orderBy: any = {};
      if (params.sortBy) {
        orderBy[params.sortBy] = params.sortDirection || 'desc';
      } else {
        orderBy.timestamp = 'desc'; // Default sort by timestamp
      }
      
      // Get decision logs from read-only database
      const decisionLogs = await prismaReadOnly.decisionLog.findMany({
        where,
        orderBy: { timestamp: 'desc' },
        skip: params.offset,
        take: params.limit
      });
      
      logger.info('Decision logs retrieved', { count: decisionLogs.length });
      
      // Map to response format
      return decisionLogs.map((log: any) => this.mapDecisionLogFromDb(log));
    } catch (error) {
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
  async countDecisionLogs(params: DecisionLogQueryParams): Promise<number> {
    try {
      logger.info('Counting decision logs', { params });
      
      // Build where clause based on parameters
      const where: any = {};
      
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
      
      // Get count from read-only database
      const count = await prismaReadOnly.decisionLog.count({ where });
      
      logger.info('Decision logs counted', { count });
      
      return count;
    } catch (error) {
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
  private mapDecisionLogFromDb(dbLog: any): DecisionLog {
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

// Export singleton instance
export const decisionLogService = new DecisionLogService();

// Export default for dependency injection
export default decisionLogService; 

/**
 * Standalone function to create a decision log
 * Wrapper around the decisionLogService.createDecisionLog method
 * @param data - Decision log entry data
 * @returns Promise resolving to the created decision log
 */
export const createDecisionLog = (data: DecisionLogEntry): Promise<DecisionLog> => {
  return decisionLogService.createDecisionLog({
    source: data.source as any,
    actionType: data.actionType as any,
    decision: data.decision,
    reasonDetails: data.reasonDetails,
    userId: data.userId,
    botId: data.botId,
    strategyId: data.strategyId,
    symbol: data.symbol,
    orderId: data.orderId,
    importance: data.importance as any,
    tags: data.tags
  });
}; 