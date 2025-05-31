"use strict";
/**
 * Bot Routes
 * Simple placeholder implementation
 */
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = require("express");
const authMiddleware_1 = require("../../middleware/authMiddleware");
const router = (0, express_1.Router)();
// Apply authentication middleware
router.use(authMiddleware_1.authenticateJWT);
// Simple placeholder response for all bot routes
router.all('*', (req, res) => {
    res.status(501).json({
        success: false,
        message: 'Bot functionality not yet implemented'
    });
});
exports.default = router;
//# sourceMappingURL=botRoutes.js.map