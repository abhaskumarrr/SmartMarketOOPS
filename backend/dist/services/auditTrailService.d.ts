/**
 * Audit Trail Service
 * Handles audit logs for user actions and system events
 */
import { AuditTrail, AuditEvent, AuditTrailStatus, IAuditTrailService, AuditTrailQueryParams, CreateAuditTrailRequest, CreateAuditEventRequest } from '../types/auditLog';
/**
 * Log severity levels
 */
export declare enum LogSeverity {
    DEBUG = "DEBUG",
    INFO = "INFO",
    WARN = "WARN",
    ERROR = "ERROR",
    CRITICAL = "CRITICAL"
}
/**
 * Audit Trail Service class
 * Handles comprehensive audit trail functionality for tracking system actions
 */
export declare class AuditTrailService implements IAuditTrailService {
    /**
     * Create a new audit trail
     * @param data - Audit trail creation request
     * @returns Created audit trail
     */
    createAuditTrail(data: CreateAuditTrailRequest): Promise<AuditTrail>;
    /**
     * Get an audit trail by ID
     * @param id - Audit trail ID
     * @param includeEvents - Whether to include events
     * @param includeDecisionLogs - Whether to include decision logs
     * @returns Audit trail
     */
    getAuditTrail(id: string, includeEvents?: boolean, includeDecisionLogs?: boolean): Promise<AuditTrail>;
    /**
     * Update an audit trail
     * @param id - Audit trail ID
     * @param data - Updated audit trail data
     * @returns Updated audit trail
     */
    updateAuditTrail(id: string, data: Partial<AuditTrail>): Promise<AuditTrail>;
    /**
     * Complete an audit trail
     * @param id - Audit trail ID
     * @param status - Status to set (default: COMPLETED)
     * @returns Completed audit trail
     */
    completeAuditTrail(id: string, status?: AuditTrailStatus): Promise<AuditTrail>;
    /**
     * Delete an audit trail
     * @param id - Audit trail ID
     * @returns Whether the deletion was successful
     */
    deleteAuditTrail(id: string): Promise<boolean>;
    /**
     * Create an audit event
     * @param data - Audit event creation request
     * @returns Created audit event
     */
    createAuditEvent(data: CreateAuditEventRequest): Promise<AuditEvent>;
    /**
     * Get an audit event by ID
     * @param id - Audit event ID
     * @returns Audit event
     */
    getAuditEvent(id: string): Promise<AuditEvent>;
    /**
     * Query audit trails based on parameters
     * @param params - Query parameters
     * @returns Matching audit trails
     */
    queryAuditTrails(params: AuditTrailQueryParams): Promise<AuditTrail[]>;
    /**
     * Count audit trails based on parameters
     * @param params - Query parameters
     * @returns Number of matching audit trails
     */
    countAuditTrails(params: AuditTrailQueryParams): Promise<number>;
    /**
     * Query audit events based on parameters
     * @param params - Query parameters
     * @returns Matching audit events
     */
    queryAuditEvents(params: any): Promise<AuditEvent[]>;
    /**
     * Count audit events based on parameters
     * @param params - Query parameters
     * @returns Number of matching audit events
     */
    countAuditEvents(params: any): Promise<number>;
    /**
     * Map an audit trail from database to response format
     * @param dbTrail - Audit trail from database
     * @returns Mapped audit trail
     */
    private mapAuditTrailFromDb;
    /**
     * Map an audit event from database to response format
     * @param dbEvent - Audit event from database
     * @returns Mapped audit event
     */
    private mapAuditEventFromDb;
}
export declare const auditTrailService: AuditTrailService;
export default auditTrailService;
