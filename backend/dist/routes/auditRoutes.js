"use strict";
/**
 * Audit Routes
 * Routes for decision logs and audit trails
 */
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = require("express");
const auditController_1 = require("../controllers/auditController");
const authMiddleware_1 = require("../middleware/authMiddleware");
// Create router
const router = (0, express_1.Router)();
/**
 * Decision Log Routes
 */
// Create a decision log
router.post('/decision-logs', authMiddleware_1.authenticateJWT, auditController_1.auditController.createDecisionLog);
// Get a decision log by ID
router.get('/decision-logs/:id', authMiddleware_1.authenticateJWT, auditController_1.auditController.getDecisionLog);
// Query decision logs
router.get('/decision-logs', authMiddleware_1.authenticateJWT, auditController_1.auditController.queryDecisionLogs);
/**
 * Audit Trail Routes
 */
// Create an audit trail
router.post('/audit-trails', authMiddleware_1.authenticateJWT, auditController_1.auditController.createAuditTrail);
// Get an audit trail by ID
router.get('/audit-trails/:id', authMiddleware_1.authenticateJWT, auditController_1.auditController.getAuditTrail);
// Query audit trails
router.get('/audit-trails', authMiddleware_1.authenticateJWT, auditController_1.auditController.queryAuditTrails);
// Complete an audit trail
router.post('/audit-trails/:id/complete', authMiddleware_1.authenticateJWT, auditController_1.auditController.completeAuditTrail);
/**
 * Audit Event Routes
 */
// Create an audit event
router.post('/audit-events', authMiddleware_1.authenticateJWT, auditController_1.auditController.createAuditEvent);
/**
 * Utility Routes
 */
// Get all supported enum values
router.get('/enum-values', authMiddleware_1.authenticateJWT, auditController_1.auditController.getEnumValues);
// Export router
exports.default = router;
//# sourceMappingURL=auditRoutes.js.map