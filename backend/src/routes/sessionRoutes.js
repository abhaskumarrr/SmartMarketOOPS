/**
 * Session Management Routes
 * Provides endpoints for managing user sessions
 */

const express = require('express');
const router = express.Router();
const { auth } = require('../middleware/auth');
const sessionController = require('../controllers/sessionController');

// Get all active sessions for the current user
router.get('/', auth, sessionController.getUserSessions);

// Get details of a specific session
router.get('/:id', auth, sessionController.getSessionById);

// Revoke a specific session
router.delete('/:id', auth, sessionController.revokeSession);

// Revoke all sessions except the current one
router.delete('/', auth, sessionController.revokeAllSessions);

module.exports = router; 