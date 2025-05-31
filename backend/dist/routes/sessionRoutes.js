"use strict";
/**
 * Session Routes
 * Endpoints for session management and device tracking
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const auth_1 = require("../middleware/auth");
const sessionController_1 = require("../controllers/sessionController");
const router = express_1.default.Router();
// All routes require authentication
router.use(auth_1.protect);
// Get all active sessions for the current user
router.get('/', sessionController_1.getUserSessions);
// Check current session status
router.get('/check', sessionController_1.checkSession);
// Revoke a specific session by ID
router.delete('/:sessionId', sessionController_1.revokeSession);
// Revoke all other sessions except current one
router.delete('/', sessionController_1.revokeAllSessions);
exports.default = router;
//# sourceMappingURL=sessionRoutes.js.map