/**
 * Audit Controller
 * Handles requests related to decision logs and audit trails
 */
import { Request, Response } from 'express';
/**
 * Audit Controller class
 * Handles requests related to decision logs and audit trails
 */
export declare class AuditController {
    /**
     * Create a decision log
     * @param req - Request
     * @param res - Response
     */
    createDecisionLog(req: Request, res: Response): Promise<void>;
    /**
     * Get a decision log by ID
     * @param req - Request
     * @param res - Response
     */
    getDecisionLog(req: Request, res: Response): Promise<void>;
    /**
     * Query decision logs
     * @param req - Request
     * @param res - Response
     */
    queryDecisionLogs(req: Request, res: Response): Promise<void>;
    /**
     * Create an audit trail
     * @param req - Request
     * @param res - Response
     */
    createAuditTrail(req: Request, res: Response): Promise<void>;
    /**
     * Get an audit trail by ID
     * @param req - Request
     * @param res - Response
     */
    getAuditTrail(req: Request, res: Response): Promise<void>;
    /**
     * Query audit trails
     * @param req - Request
     * @param res - Response
     */
    queryAuditTrails(req: Request, res: Response): Promise<void>;
    /**
     * Complete an audit trail
     * @param req - Request
     * @param res - Response
     */
    completeAuditTrail(req: Request, res: Response): Promise<void>;
    /**
     * Create an audit event
     * @param req - Request
     * @param res - Response
     */
    createAuditEvent(req: Request, res: Response): Promise<void>;
    /**
     * Get all supported enum values
     * @param req - Request
     * @param res - Response
     */
    getEnumValues(req: Request, res: Response): Promise<void>;
}
export declare const auditController: AuditController;
export default auditController;
