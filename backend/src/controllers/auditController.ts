/**
 * Audit Controller
 * Handles requests related to decision logs and audit trails
 */

import { Request, Response } from 'express';
import { createLogger } from '../utils/logger';
import { decisionLogService } from '../services/decisionLogService';
import { auditTrailService } from '../services/auditTrailService';
import {
  DecisionSource,
  DecisionActionType,
  DecisionImportance,
  AuditTrailType,
  AuditEventType,
  AuditEventStatus
} from '../types/auditLog';

// Create logger
const logger = createLogger('AuditController');

/**
 * Audit Controller class
 * Handles requests related to decision logs and audit trails
 */
export class AuditController {
  /**
   * Create a decision log
   * @param req - Request
   * @param res - Response
   */
  async createDecisionLog(req: Request, res: Response): Promise<void> {
    try {
      logger.info('Creating decision log', { body: req.body });
      
      const {
        source,
        actionType,
        decision,
        reasonCode,
        reasonDetails,
        confidence,
        dataSnapshot,
        parameters,
        modelVersion,
        userId,
        strategyId,
        botId,
        signalId,
        orderId,
        symbol,
        tags,
        importance,
        notes,
        auditTrailId
      } = req.body;
      
      // Validate required fields
      if (!source || !actionType || !decision) {
        logger.warn('Missing required fields for decision log', { body: req.body });
        
        res.status(400).json({
          success: false,
          error: {
            code: 'MISSING_REQUIRED_FIELDS',
            message: 'Missing required fields: source, actionType, decision'
          },
          timestamp: new Date().toISOString()
        });
        
        return;
      }
      
      // Create decision log
      const decisionLog = await decisionLogService.createDecisionLog({
        source,
        actionType,
        decision,
        reasonCode,
        reasonDetails,
        confidence,
        dataSnapshot,
        parameters,
        modelVersion,
        userId,
        strategyId,
        botId,
        signalId,
        orderId,
        symbol,
        tags,
        importance,
        notes,
        auditTrailId
      });
      
      logger.info('Decision log created', { id: decisionLog.id });
      
      // Return response
      res.status(201).json({
        success: true,
        data: decisionLog,
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      logger.error(`Failed to create decision log: ${errorMessage}`, { error });
      
      res.status(500).json({
        success: false,
        error: {
          code: 'INTERNAL_SERVER_ERROR',
          message: 'Failed to create decision log'
        },
        timestamp: new Date().toISOString()
      });
    }
  }
  
  /**
   * Get a decision log by ID
   * @param req - Request
   * @param res - Response
   */
  async getDecisionLog(req: Request, res: Response): Promise<void> {
    try {
      const { id } = req.params;
      
      logger.info(`Getting decision log ${id}`);
      
      // Get decision log
      const decisionLog = await decisionLogService.getDecisionLog(id);
      
      logger.info('Decision log retrieved', { id });
      
      // Return response
      res.status(200).json({
        success: true,
        data: decisionLog,
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      logger.error(`Failed to get decision log: ${errorMessage}`, { error });
      
      if (errorMessage.includes('not found')) {
        res.status(404).json({
          success: false,
          error: {
            code: 'DECISION_LOG_NOT_FOUND',
            message: `Decision log not found`
          },
          timestamp: new Date().toISOString()
        });
      } else {
        res.status(500).json({
          success: false,
          error: {
            code: 'INTERNAL_SERVER_ERROR',
            message: 'Failed to get decision log'
          },
          timestamp: new Date().toISOString()
        });
      }
    }
  }
  
  /**
   * Query decision logs
   * @param req - Request
   * @param res - Response
   */
  async queryDecisionLogs(req: Request, res: Response): Promise<void> {
    try {
      const {
        startDate,
        endDate,
        source,
        actionType,
        userId,
        strategyId,
        botId,
        signalId,
        orderId,
        symbol,
        outcome,
        importance,
        tags,
        limit,
        offset,
        sortBy,
        sortDirection,
        count
      } = req.query;
      
      logger.info('Querying decision logs', { query: req.query });
      
      // Parse query parameters
      const params: any = {};
      
      if (startDate) {
        params.startDate = startDate;
      }
      
      if (endDate) {
        params.endDate = endDate;
      }
      
      if (source) {
        params.source = source;
      }
      
      if (actionType) {
        params.actionType = actionType;
      }
      
      if (userId) {
        params.userId = userId;
      }
      
      if (strategyId) {
        params.strategyId = strategyId;
      }
      
      if (botId) {
        params.botId = botId;
      }
      
      if (signalId) {
        params.signalId = signalId;
      }
      
      if (orderId) {
        params.orderId = orderId;
      }
      
      if (symbol) {
        params.symbol = symbol;
      }
      
      if (outcome) {
        params.outcome = outcome;
      }
      
      if (importance) {
        params.importance = importance;
      }
      
      if (tags) {
        params.tags = Array.isArray(tags) ? tags : [tags];
      }
      
      if (limit) {
        params.limit = parseInt(limit as string, 10);
      }
      
      if (offset) {
        params.offset = parseInt(offset as string, 10);
      }
      
      if (sortBy) {
        params.sortBy = sortBy;
      }
      
      if (sortDirection) {
        params.sortDirection = sortDirection;
      }
      
      // Determine whether to count or query
      if (count === 'true') {
        // Count decision logs
        const total = await decisionLogService.countDecisionLogs(params);
        
        logger.info(`Counted ${total} decision logs`);
        
        // Return response
        res.status(200).json({
          success: true,
          data: { total },
          timestamp: new Date().toISOString()
        });
      } else {
        // Query decision logs
        const decisionLogs = await decisionLogService.queryDecisionLogs(params);
        
        logger.info(`Found ${decisionLogs.length} decision logs`);
        
        // Return response
        res.status(200).json({
          success: true,
          data: decisionLogs,
          timestamp: new Date().toISOString()
        });
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      logger.error(`Failed to query decision logs: ${errorMessage}`, { error });
      
      res.status(500).json({
        success: false,
        error: {
          code: 'INTERNAL_SERVER_ERROR',
          message: 'Failed to query decision logs'
        },
        timestamp: new Date().toISOString()
      });
    }
  }
  
  /**
   * Create an audit trail
   * @param req - Request
   * @param res - Response
   */
  async createAuditTrail(req: Request, res: Response): Promise<void> {
    try {
      logger.info('Creating audit trail', { body: req.body });
      
      const {
        trailType,
        entityId,
        entityType,
        summary,
        userId,
        orderId,
        tags,
        metadata
      } = req.body;
      
      // Validate required fields
      if (!trailType || !entityId || !entityType) {
        logger.warn('Missing required fields for audit trail', { body: req.body });
        
        res.status(400).json({
          success: false,
          error: {
            code: 'MISSING_REQUIRED_FIELDS',
            message: 'Missing required fields: trailType, entityId, entityType'
          },
          timestamp: new Date().toISOString()
        });
        
        return;
      }
      
      // Create audit trail
      const auditTrail = await auditTrailService.createAuditTrail({
        trailType,
        entityId,
        entityType,
        summary,
        userId,
        orderId,
        tags,
        metadata
      });
      
      logger.info('Audit trail created', { id: auditTrail.id });
      
      // Return response
      res.status(201).json({
        success: true,
        data: auditTrail,
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      logger.error(`Failed to create audit trail: ${errorMessage}`, { error });
      
      res.status(500).json({
        success: false,
        error: {
          code: 'INTERNAL_SERVER_ERROR',
          message: 'Failed to create audit trail'
        },
        timestamp: new Date().toISOString()
      });
    }
  }
  
  /**
   * Get an audit trail by ID
   * @param req - Request
   * @param res - Response
   */
  async getAuditTrail(req: Request, res: Response): Promise<void> {
    try {
      const { id } = req.params;
      const { includeEvents, includeDecisionLogs } = req.query;
      
      logger.info(`Getting audit trail ${id}`, { includeEvents, includeDecisionLogs });
      
      // Get audit trail
      const auditTrail = await auditTrailService.getAuditTrail(
        id,
        includeEvents === 'true',
        includeDecisionLogs === 'true'
      );
      
      logger.info('Audit trail retrieved', { id });
      
      // Return response
      res.status(200).json({
        success: true,
        data: auditTrail,
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      logger.error(`Failed to get audit trail: ${errorMessage}`, { error });
      
      if (errorMessage.includes('not found')) {
        res.status(404).json({
          success: false,
          error: {
            code: 'AUDIT_TRAIL_NOT_FOUND',
            message: `Audit trail not found`
          },
          timestamp: new Date().toISOString()
        });
      } else {
        res.status(500).json({
          success: false,
          error: {
            code: 'INTERNAL_SERVER_ERROR',
            message: 'Failed to get audit trail'
          },
          timestamp: new Date().toISOString()
        });
      }
    }
  }
  
  /**
   * Query audit trails
   * @param req - Request
   * @param res - Response
   */
  async queryAuditTrails(req: Request, res: Response): Promise<void> {
    try {
      const {
        startDate,
        endDate,
        trailType,
        entityId,
        entityType,
        status,
        userId,
        orderId,
        tags,
        includeEvents,
        includeDecisionLogs,
        limit,
        offset,
        sortBy,
        sortDirection,
        count
      } = req.query;
      
      logger.info('Querying audit trails', { query: req.query });
      
      // Parse query parameters
      const params: any = {};
      
      if (startDate) {
        params.startDate = startDate;
      }
      
      if (endDate) {
        params.endDate = endDate;
      }
      
      if (trailType) {
        params.trailType = trailType;
      }
      
      if (entityId) {
        params.entityId = entityId;
      }
      
      if (entityType) {
        params.entityType = entityType;
      }
      
      if (status) {
        params.status = status;
      }
      
      if (userId) {
        params.userId = userId;
      }
      
      if (orderId) {
        params.orderId = orderId;
      }
      
      if (tags) {
        params.tags = Array.isArray(tags) ? tags : [tags];
      }
      
      if (includeEvents === 'true') {
        params.includeEvents = true;
      }
      
      if (includeDecisionLogs === 'true') {
        params.includeDecisionLogs = true;
      }
      
      if (limit) {
        params.limit = parseInt(limit as string, 10);
      }
      
      if (offset) {
        params.offset = parseInt(offset as string, 10);
      }
      
      if (sortBy) {
        params.sortBy = sortBy;
      }
      
      if (sortDirection) {
        params.sortDirection = sortDirection;
      }
      
      // Determine whether to count or query
      if (count === 'true') {
        // Count audit trails
        const total = await auditTrailService.countAuditTrails(params);
        
        logger.info(`Counted ${total} audit trails`);
        
        // Return response
        res.status(200).json({
          success: true,
          data: { total },
          timestamp: new Date().toISOString()
        });
      } else {
        // Query audit trails
        const auditTrails = await auditTrailService.queryAuditTrails(params);
        
        logger.info(`Found ${auditTrails.length} audit trails`);
        
        // Return response
        res.status(200).json({
          success: true,
          data: auditTrails,
          timestamp: new Date().toISOString()
        });
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      logger.error(`Failed to query audit trails: ${errorMessage}`, { error });
      
      res.status(500).json({
        success: false,
        error: {
          code: 'INTERNAL_SERVER_ERROR',
          message: 'Failed to query audit trails'
        },
        timestamp: new Date().toISOString()
      });
    }
  }
  
  /**
   * Complete an audit trail
   * @param req - Request
   * @param res - Response
   */
  async completeAuditTrail(req: Request, res: Response): Promise<void> {
    try {
      const { id } = req.params;
      const { status } = req.body;
      
      logger.info(`Completing audit trail ${id}`, { status });
      
      // Complete audit trail
      const auditTrail = await auditTrailService.completeAuditTrail(id, status);
      
      logger.info('Audit trail completed', { id, status });
      
      // Return response
      res.status(200).json({
        success: true,
        data: auditTrail,
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      logger.error(`Failed to complete audit trail: ${errorMessage}`, { error });
      
      if (errorMessage.includes('not found')) {
        res.status(404).json({
          success: false,
          error: {
            code: 'AUDIT_TRAIL_NOT_FOUND',
            message: `Audit trail not found`
          },
          timestamp: new Date().toISOString()
        });
      } else {
        res.status(500).json({
          success: false,
          error: {
            code: 'INTERNAL_SERVER_ERROR',
            message: 'Failed to complete audit trail'
          },
          timestamp: new Date().toISOString()
        });
      }
    }
  }
  
  /**
   * Create an audit event
   * @param req - Request
   * @param res - Response
   */
  async createAuditEvent(req: Request, res: Response): Promise<void> {
    try {
      logger.info('Creating audit event', { body: req.body });
      
      const {
        auditTrailId,
        eventType,
        component,
        action,
        status,
        details,
        dataBefore,
        dataAfter,
        metadata
      } = req.body;
      
      // Validate required fields
      if (!auditTrailId || !eventType || !component || !action || !status) {
        logger.warn('Missing required fields for audit event', { body: req.body });
        
        res.status(400).json({
          success: false,
          error: {
            code: 'MISSING_REQUIRED_FIELDS',
            message: 'Missing required fields: auditTrailId, eventType, component, action, status'
          },
          timestamp: new Date().toISOString()
        });
        
        return;
      }
      
      // Create audit event
      const auditEvent = await auditTrailService.createAuditEvent({
        auditTrailId,
        eventType,
        component,
        action,
        status,
        details,
        dataBefore,
        dataAfter,
        metadata
      });
      
      logger.info('Audit event created', { id: auditEvent.id });
      
      // Return response
      res.status(201).json({
        success: true,
        data: auditEvent,
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      logger.error(`Failed to create audit event: ${errorMessage}`, { error });
      
      res.status(500).json({
        success: false,
        error: {
          code: 'INTERNAL_SERVER_ERROR',
          message: 'Failed to create audit event'
        },
        timestamp: new Date().toISOString()
      });
    }
  }
  
  /**
   * Get all supported enum values
   * @param req - Request
   * @param res - Response
   */
  async getEnumValues(req: Request, res: Response): Promise<void> {
    try {
      logger.info('Getting enum values');
      
      // Create response with all enum values
      const enumValues = {
        decisionSources: Object.values(DecisionSource),
        decisionActionTypes: Object.values(DecisionActionType),
        decisionImportance: Object.values(DecisionImportance),
        auditTrailTypes: Object.values(AuditTrailType),
        auditEventTypes: Object.values(AuditEventType),
        auditEventStatuses: Object.values(AuditEventStatus)
      };
      
      logger.info('Enum values retrieved');
      
      // Return response
      res.status(200).json({
        success: true,
        data: enumValues,
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      logger.error(`Failed to get enum values: ${errorMessage}`, { error });
      
      res.status(500).json({
        success: false,
        error: {
          code: 'INTERNAL_SERVER_ERROR',
          message: 'Failed to get enum values'
        },
        timestamp: new Date().toISOString()
      });
    }
  }
}

// Create an instance of the controller
export const auditController = new AuditController();

// Export default for dependency injection
export default auditController; 