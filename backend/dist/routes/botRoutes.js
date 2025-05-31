/**
 * Trading Bot Routes
 * Endpoints for configuring and managing trading bots
 */
const express = require('express');
const router = express.Router();
const { createBot, getBots, getBot, updateBot, deleteBot, startBot, stopBot, getBotStatus } = require('../controllers/botController');
const { auth } = require('../middleware/auth');
// All routes require authentication
router.use(auth);
// Bot configuration endpoints
router.post('/', createBot);
router.get('/', getBots);
router.get('/:id', getBot);
router.put('/:id', updateBot);
router.delete('/:id', deleteBot);
// Bot control endpoints
router.post('/:id/start', startBot);
router.post('/:id/stop', stopBot);
router.get('/:id/status', getBotStatus);
module.exports = router;
//# sourceMappingURL=botRoutes.js.map