/**
 * Session Routes
 * Endpoints for session management and device tracking
 */

import express from 'express';
import { protect } from '../middleware/auth';
import { validateUserSession } from '../middleware/sessionMiddleware';
import { 
  getUserSessions, 
  revokeSession, 
  revokeAllSessions,
  checkSession
} from '../controllers/sessionController';

const router = express.Router();

// All routes require authentication
router.use(protect);

// Get all active sessions for the current user
router.get('/', getUserSessions);

// Check current session status
router.get('/check', checkSession);

// Revoke a specific session by ID
router.delete('/:sessionId', revokeSession);

// Revoke all other sessions except current one
router.delete('/', revokeAllSessions);

export default router; 