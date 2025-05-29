/**
 * Audit Routes
 * Routes for decision logs and audit trails
 */

import { Router } from 'express';
import { auditController } from '../controllers/auditController';
import { authenticateJWT } from '../middleware/authMiddleware';

// Create router
const router = Router();

/**
 * Decision Log Routes
 */

// Create a decision log
router.post(
  '/decision-logs',
  authenticateJWT,
  auditController.createDecisionLog
);

// Get a decision log by ID
router.get(
  '/decision-logs/:id',
  authenticateJWT,
  auditController.getDecisionLog
);

// Query decision logs
router.get(
  '/decision-logs',
  authenticateJWT,
  auditController.queryDecisionLogs
);

/**
 * Audit Trail Routes
 */

// Create an audit trail
router.post(
  '/audit-trails',
  authenticateJWT,
  auditController.createAuditTrail
);

// Get an audit trail by ID
router.get(
  '/audit-trails/:id',
  authenticateJWT,
  auditController.getAuditTrail
);

// Query audit trails
router.get(
  '/audit-trails',
  authenticateJWT,
  auditController.queryAuditTrails
);

// Complete an audit trail
router.post(
  '/audit-trails/:id/complete',
  authenticateJWT,
  auditController.completeAuditTrail
);

/**
 * Audit Event Routes
 */

// Create an audit event
router.post(
  '/audit-events',
  authenticateJWT,
  auditController.createAuditEvent
);

/**
 * Utility Routes
 */

// Get all supported enum values
router.get(
  '/enum-values',
  authenticateJWT,
  auditController.getEnumValues
);

// Export router
export default router; 