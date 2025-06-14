/**
 * Audit Trail Service
 * Handles audit logs for user actions and system events
 */

import { v4 as uuidv4 } from 'uuid';
import prisma from '../utils/prismaClient';
import prismaReadOnly from '../config/prisma-readonly'; // Import the read-only client
import { createLogger, LogData } from '../utils/logger';
import {
  AuditTrail,
  AuditEvent,
  AuditEventStatus,
  AuditEventType,
  AuditTrailStatus,
  AuditTrailType,
  IAuditTrailService,
  AuditTrailQueryParams,
  CreateAuditTrailRequest,
  CreateAuditEventRequest
} from '../types/auditLog';

// Create logger
const logger = createLogger('AuditTrailService');

/**
 * Log severity levels
 */
export enum LogSeverity {
  DEBUG = 'DEBUG',
  INFO = 'INFO',
  WARN = 'WARN',
  ERROR = 'ERROR',
  CRITICAL = 'CRITICAL'
}

/**
 * Audit Trail Service class
 * Handles comprehensive audit trail functionality for tracking system actions
 */
export class AuditTrailService implements IAuditTrailService {
  /**
   * Create a new audit trail
   * @param data - Audit trail creation request
   * @returns Created audit trail
   */
  async createAuditTrail(data: CreateAuditTrailRequest): Promise<AuditTrail> {
    try {
      logger.info('Creating audit trail', { data });
      
      // Set default values if not provided
      const tags = data.tags || [];
      
      // Create audit trail in database
      const auditTrail = await prisma.auditTrail.create({
        data: {
          id: uuidv4(),
          trailType: data.trailType,
          entityId: data.entityId,
          entityType: data.entityType,
          startTime: new Date(),
          status: AuditTrailStatus.ACTIVE,
          summary: data.summary,
          userId: data.userId,
          orderId: data.orderId,
          tags,
          metadata: data.metadata as any
        }
      });
      
      logger.info('Audit trail created', { id: auditTrail.id });
      
      // Map to response format
      return this.mapAuditTrailFromDb(auditTrail);
    } catch (error) {
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
  async getAuditTrail(id: string, includeEvents = false, includeDecisionLogs = false): Promise<AuditTrail> {
    try {
      logger.info(`Getting audit trail ${id}`, { includeEvents, includeDecisionLogs });
      
      // Build include object
      const include: any = {};
      
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
      
      // Get audit trail from read-only database
      const auditTrail = await prismaReadOnly.auditTrail.findUnique({
        where: { id },
        include: Object.keys(include).length > 0 ? include : undefined
      });
      
      if (!auditTrail) {
        throw new Error(`Audit trail ${id} not found`);
      }
      
      logger.info('Audit trail retrieved', { id });
      
      // Map to response format
      return this.mapAuditTrailFromDb(auditTrail);
    } catch (error) {
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
  async updateAuditTrail(id: string, data: Partial<AuditTrail>): Promise<AuditTrail> {
    try {
      logger.info(`Updating audit trail ${id}`, { data });
      
      // Update audit trail in database
      const auditTrail = await prisma.auditTrail.update({
        where: { id },
        data: {
          status: data.status,
          endTime: data.endTime ? new Date(data.endTime) : undefined,
          summary: data.summary,
          tags: data.tags,
          metadata: data.metadata as any
        }
      });
      
      logger.info('Audit trail updated', { id });
      
      // Map to response format
      return this.mapAuditTrailFromDb(auditTrail);
    } catch (error) {
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
  async completeAuditTrail(id: string, status = AuditTrailStatus.COMPLETED): Promise<AuditTrail> {
    try {
      logger.info(`Completing audit trail ${id}`, { status });
      
      // Update audit trail in database
      const auditTrail = await prisma.auditTrail.update({
        where: { id },
        data: {
          status,
          endTime: new Date()
        }
      });
      
      logger.info('Audit trail completed', { id, status });
      
      // Map to response format
      return this.mapAuditTrailFromDb(auditTrail);
    } catch (error) {
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
  async deleteAuditTrail(id: string): Promise<boolean> {
    try {
      logger.info(`Deleting audit trail ${id}`);
      
      // Delete audit trail from database
      await prisma.auditTrail.delete({
        where: { id }
      });
      
      logger.info('Audit trail deleted', { id });
      
      return true;
    } catch (error) {
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
  async createAuditEvent(data: CreateAuditEventRequest): Promise<AuditEvent> {
    try {
      logger.info('Creating audit event', { data });
      
      // Create audit event in database
      const auditEvent = await prisma.auditEvent.create({
        data: {
          id: uuidv4(),
          auditTrailId: data.auditTrailId,
          timestamp: new Date(),
          eventType: data.eventType,
          component: data.component,
          action: data.action,
          status: data.status,
          details: data.details as any,
          dataBefore: data.dataBefore as any,
          dataAfter: data.dataAfter as any,
          metadata: data.metadata as any
        }
      });
      
      logger.info('Audit event created', { id: auditEvent.id });
      
      // Map to response format
      return this.mapAuditEventFromDb(auditEvent);
    } catch (error) {
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
  async getAuditEvent(id: string): Promise<AuditEvent> {
    try {
      logger.info(`Getting audit event ${id}`);
      
      // Get audit event from read-only database
      const auditEvent = await prismaReadOnly.auditEvent.findUnique({
        where: { id }
      });
      
      if (!auditEvent) {
        throw new Error(`Audit event ${id} not found`);
      }
      
      logger.info('Audit event retrieved', { id });
      
      // Map to response format
      return this.mapAuditEventFromDb(auditEvent);
    } catch (error) {
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
  async queryAuditTrails(params: AuditTrailQueryParams): Promise<AuditTrail[]> {
    try {
      logger.info('Querying audit trails', { params });
      
      // Build where clause based on parameters
      const where: any = {};
      
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
      const include: any = {};
      
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
      const orderBy: any = {};
      if (params.sortBy) {
        orderBy[params.sortBy] = params.sortDirection || 'desc';
      } else {
        orderBy.startTime = 'desc'; // Default sort by start time
      }
      
      // Get audit trails from read-only database
      const auditTrails = await prismaReadOnly.auditTrail.findMany({
        where: where,
        orderBy,
        skip: params.offset || 0,
        take: params.limit || 100
      });
      
      logger.info(`Found ${auditTrails.length} audit trails`);
      
      // Map to response format
      return auditTrails.map((trail: any) => this.mapAuditTrailFromDb(trail));
    } catch (error) {
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
  async countAuditTrails(params: AuditTrailQueryParams): Promise<number> {
    try {
      logger.info('Counting audit trails', { params });
      
      // Build where clause based on parameters
      const where: any = {};
      
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
      
      // Get count from read-only database
      const count = await prismaReadOnly.auditTrail.count({
        where
      });
      
      logger.info(`Counted ${count} audit trails`);
      
      return count;
    } catch (error) {
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
  async queryAuditEvents(params: any): Promise<AuditEvent[]> {
    try {
      logger.info('Querying audit events', { params });
      
      // Build where clause based on parameters
      const where: any = {};
      
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
      const orderBy: any = {};
      if (params.sortBy) {
        orderBy[params.sortBy] = params.sortDirection || 'desc';
      } else {
        orderBy.timestamp = 'desc'; // Default sort by timestamp
      }
      
      // Get audit events from read-only database
      const auditEvents = await prismaReadOnly.auditEvent.findMany({
        where: where,
        orderBy,
        skip: params.offset || 0,
        take: params.limit || 100
      });
      
      logger.info(`Found ${auditEvents.length} audit events`);
      
      // Map to response format
      return auditEvents.map((event: any) => this.mapAuditEventFromDb(event));
    } catch (error) {
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
  async countAuditEvents(params: any): Promise<number> {
    try {
      logger.info('Counting audit events', { params });
      
      // Build where clause based on parameters
      const where: any = {};
      
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
      
      // Get count from read-only database
      const count = await prismaReadOnly.auditEvent.count({
        where
      });
      
      logger.info(`Counted ${count} audit events`);
      
      return count;
    } catch (error) {
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
  private mapAuditTrailFromDb(dbTrail: any): AuditTrail {
    const result: AuditTrail = {
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
      result.events = dbTrail.events.map((event: any) => this.mapAuditEventFromDb(event));
    }
    
    // Add decision logs if present
    if (dbTrail.decisionLogs) {
      result.decisionLogs = dbTrail.decisionLogs.map((log: any) => ({
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
  private mapAuditEventFromDb(dbEvent: any): AuditEvent {
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

// Export singleton instance
export const auditTrailService = new AuditTrailService();

// Export default for dependency injection
export default auditTrailService; 