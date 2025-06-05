"use strict";
/**
 * Metrics Routes
 * Endpoints for system metrics and monitoring
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const router = express_1.default.Router();
// Health check for metrics routes
router.get('/metrics/health', (req, res) => {
    res.json({
        status: 'Metrics routes working',
        timestamp: new Date().toISOString(),
        uptime: process.uptime(),
        memory: process.memoryUsage()
    });
});
// Basic system metrics
router.get('/metrics/system', (req, res) => {
    res.json({
        success: true,
        data: {
            uptime: process.uptime(),
            memory: process.memoryUsage(),
            cpu: process.cpuUsage(),
            platform: process.platform,
            nodeVersion: process.version,
            timestamp: new Date().toISOString()
        }
    });
});
exports.default = router;
//# sourceMappingURL=metricsRoutes.js.map