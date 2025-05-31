/**
 * Audit Log Types
 * Type definitions for the audit logging and decision logging system
 */
import { Timestamp, UUID } from './common';
/**
 * Decision source types
 */
export declare enum DecisionSource {
    SIGNAL = "SIGNAL",
    STRATEGY = "STRATEGY",
    BOT = "BOT",
    USER = "USER",
    SYSTEM = "SYSTEM",
    RISK_MANAGEMENT = "RISK_MANAGEMENT",
    ML_MODEL = "ML_MODEL",
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
}
/**
 * Decision action types
 */
export declare enum DecisionActionType {
    ENTRY = "ENTRY",
    EXIT = "EXIT",
    ADJUSTMENT = "ADJUSTMENT",
    CANCELLATION = "CANCELLATION",
    RISK_OVERRIDE = "RISK_OVERRIDE",
    LEVERAGE_CHANGE = "LEVERAGE_CHANGE",
    POSITION_CLOSE = "POSITION_CLOSE",
    STOP_LOSS_MOVE = "STOP_LOSS_MOVE",
    TAKE_PROFIT_MOVE = "TAKE_PROFIT_MOVE",
    ORDER_REJECTION = "ORDER_REJECTION",
    STRATEGY_ACTIVATION = "STRATEGY_ACTIVATION",
    STRATEGY_DEACTIVATION = "STRATEGY_DEACTIVATION",
    BOT_ACTIVATION = "BOT_ACTIVATION",
    BOT_DEACTIVATION = "BOT_DEACTIVATION",
    SIGNAL_GENERATION = "SIGNAL_GENERATION",
    SIGNAL_VALIDATION = "SIGNAL_VALIDATION"
}
/**
 * Decision outcome types
 */
export declare enum DecisionOutcome {
    SUCCESS = "SUCCESS",
    FAILURE = "FAILURE",
    PARTIAL = "PARTIAL",
    PENDING = "PENDING",
    CANCELLED = "CANCELLED",
    UNKNOWN = "UNKNOWN"
}
/**
 * Decision importance levels
 */
export declare enum DecisionImportance {
    HIGH = "HIGH",
    NORMAL = "NORMAL",
    LOW = "LOW"
}
/**
 * Audit trail types
 */
export declare enum AuditTrailType {
    ORDER = "ORDER",
    SIGNAL = "SIGNAL",
    STRATEGY = "STRATEGY",
    BOT = "BOT",
    RISK = "RISK",
    AUTHENTICATION = "AUTHENTICATION",
    USER_MANAGEMENT = "USER_MANAGEMENT",
    SYSTEM = "SYSTEM",
    ML_MODEL = "ML_MODEL",
    PERFORMANCE = "PERFORMANCE"
}
/**
 * Audit trail status
 */
export declare enum AuditTrailStatus {
    ACTIVE = "ACTIVE",
    COMPLETED = "COMPLETED",
    CANCELLED = "CANCELLED",
    FAILED = "FAILED"
}
/**
 * Audit event types
 */
export declare enum AuditEventType {
    CREATE = "CREATE",
    READ = "READ",
    UPDATE = "UPDATE",
    DELETE = "DELETE",
    LOGIN = "LOGIN",
    LOGOUT = "LOGOUT",
    AUTHENTICATION = "AUTHENTICATION",
    AUTHORIZATION = "AUTHORIZATION",
    EXPORT = "EXPORT",
    IMPORT = "IMPORT",
    VALIDATION = "VALIDATION",
    CALCULATION = "CALCULATION",
    EXECUTION = "EXECUTION",
    API_CALL = "API_CALL",
    ERROR = "ERROR",
    WARNING = "WARNING",
    INFO = "INFO"
}
/**
 * Audit event status
 */
export declare enum AuditEventStatus {
    SUCCESS = "SUCCESS",
    FAILURE = "FAILURE",
    WARNING = "WARNING",
    INFO = "INFO"
}
/**
 * Decision log data structure
 */
export interface DecisionLog {
    id: UUID;
    timestamp: Timestamp;
    source: DecisionSource;
    actionType: DecisionActionType;
    decision: string;
    reasonCode?: string;
    reasonDetails?: string;
    confidence?: number;
    dataSnapshot?: Record<string, any>;
    parameters?: Record<string, any>;
    modelVersion?: string;
    userId?: string;
    strategyId?: string;
    botId?: string;
    signalId?: string;
    orderId?: string;
    symbol?: string;
    outcome?: DecisionOutcome;
    outcomeDetails?: Record<string, any>;
    pnl?: number;
    evaluatedAt?: Timestamp;
    tags: string[];
    importance: DecisionImportance;
    notes?: string;
    auditTrailId?: UUID;
}
/**
 * Audit trail data structure
 */
export interface AuditTrail {
    id: UUID;
    trailType: AuditTrailType;
    entityId: string;
    entityType: string;
    startTime: Timestamp;
    endTime?: Timestamp;
    status: AuditTrailStatus;
    summary?: string;
    userId?: string;
    orderId?: string;
    tags: string[];
    metadata?: Record<string, any>;
    events?: AuditEvent[];
    decisionLogs?: DecisionLog[];
}
/**
 * Audit event data structure
 */
export interface AuditEvent {
    id: UUID;
    auditTrailId: UUID;
    timestamp: Timestamp;
    eventType: AuditEventType;
    component: string;
    action: string;
    status: AuditEventStatus;
    details?: Record<string, any>;
    dataBefore?: Record<string, any>;
    dataAfter?: Record<string, any>;
    metadata?: Record<string, any>;
}
/**
 * Decision log creation request
 */
export interface CreateDecisionLogRequest {
    source: DecisionSource;
    actionType: DecisionActionType;
    decision: string;
    reasonCode?: string;
    reasonDetails?: string;
    confidence?: number;
    dataSnapshot?: Record<string, any>;
    parameters?: Record<string, any>;
    modelVersion?: string;
    userId?: string;
    strategyId?: string;
    botId?: string;
    signalId?: string;
    orderId?: string;
    symbol?: string;
    tags?: string[];
    importance?: DecisionImportance;
    notes?: string;
    auditTrailId?: UUID;
}
/**
 * Audit trail creation request
 */
export interface CreateAuditTrailRequest {
    trailType: AuditTrailType;
    entityId: string;
    entityType: string;
    summary?: string;
    userId?: string;
    orderId?: string;
    tags?: string[];
    metadata?: Record<string, any>;
}
/**
 * Audit event creation request
 */
export interface CreateAuditEventRequest {
    auditTrailId: UUID;
    eventType: AuditEventType;
    component: string;
    action: string;
    status: AuditEventStatus;
    details?: Record<string, any>;
    dataBefore?: Record<string, any>;
    dataAfter?: Record<string, any>;
    metadata?: Record<string, any>;
}
/**
 * Query parameters for decision logs
 */
export interface DecisionLogQueryParams {
    startDate?: Timestamp;
    endDate?: Timestamp;
    source?: DecisionSource;
    actionType?: DecisionActionType;
    userId?: string;
    strategyId?: string;
    botId?: string;
    signalId?: string;
    orderId?: string;
    symbol?: string;
    outcome?: DecisionOutcome;
    importance?: DecisionImportance;
    tags?: string[];
    limit?: number;
    offset?: number;
    sortBy?: string;
    sortDirection?: 'asc' | 'desc';
}
/**
 * Query parameters for audit trails
 */
export interface AuditTrailQueryParams {
    startDate?: Timestamp;
    endDate?: Timestamp;
    trailType?: AuditTrailType;
    entityId?: string;
    entityType?: string;
    status?: AuditTrailStatus;
    userId?: string;
    orderId?: string;
    tags?: string[];
    includeEvents?: boolean;
    includeDecisionLogs?: boolean;
    limit?: number;
    offset?: number;
    sortBy?: string;
    sortDirection?: 'asc' | 'desc';
}
/**
 * Query parameters for audit events
 */
export interface AuditEventQueryParams {
    auditTrailId?: UUID;
    startDate?: Timestamp;
    endDate?: Timestamp;
    eventType?: AuditEventType;
    component?: string;
    status?: AuditEventStatus;
    limit?: number;
    offset?: number;
    sortBy?: string;
    sortDirection?: 'asc' | 'desc';
}
/**
 * Decision log service interface
 */
export interface IDecisionLogService {
    createDecisionLog(data: CreateDecisionLogRequest): Promise<DecisionLog>;
    getDecisionLog(id: UUID): Promise<DecisionLog>;
    updateDecisionLog(id: UUID, data: Partial<DecisionLog>): Promise<DecisionLog>;
    deleteDecisionLog(id: UUID): Promise<boolean>;
    queryDecisionLogs(params: DecisionLogQueryParams): Promise<DecisionLog[]>;
    countDecisionLogs(params: DecisionLogQueryParams): Promise<number>;
}
/**
 * Audit trail service interface
 */
export interface IAuditTrailService {
    createAuditTrail(data: CreateAuditTrailRequest): Promise<AuditTrail>;
    getAuditTrail(id: UUID, includeEvents?: boolean, includeDecisionLogs?: boolean): Promise<AuditTrail>;
    updateAuditTrail(id: UUID, data: Partial<AuditTrail>): Promise<AuditTrail>;
    completeAuditTrail(id: UUID, status?: AuditTrailStatus): Promise<AuditTrail>;
    deleteAuditTrail(id: UUID): Promise<boolean>;
    queryAuditTrails(params: AuditTrailQueryParams): Promise<AuditTrail[]>;
    countAuditTrails(params: AuditTrailQueryParams): Promise<number>;
}
/**
 * Audit event service interface
 */
export interface IAuditEventService {
    createAuditEvent(data: CreateAuditEventRequest): Promise<AuditEvent>;
    getAuditEvent(id: UUID): Promise<AuditEvent>;
    queryAuditEvents(params: AuditEventQueryParams): Promise<AuditEvent[]>;
    countAuditEvents(params: AuditEventQueryParams): Promise<number>;
}
