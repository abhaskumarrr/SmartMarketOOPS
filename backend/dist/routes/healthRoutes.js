"use strict";
/**
 * Health Check Routes
 * Endpoints for system monitoring and health status
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const dbHealthCheck_1 = require("../utils/dbHealthCheck");
const router = express_1.default.Router();
// Basic health check
router.get('/', (req, res) => {
    res.status(200).json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        uptime: process.uptime()
    });
});
// Database connection health check
router.get('/db', async (req, res) => {
    try {
        const dbStatus = await (0, dbHealthCheck_1.checkDbConnection)();
        res.status(200).json({
            status: dbStatus.success ? 'healthy' : 'unhealthy',
            timestamp: new Date().toISOString(),
            database: dbStatus
        });
    }
    catch (error) {
        res.status(500).json({
            status: 'unhealthy',
            timestamp: new Date().toISOString(),
            error: error.message
        });
    }
});
// Detailed system health check
router.get('/system', (req, res) => {
    const memoryUsage = process.memoryUsage();
    res.status(200).json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        system: {
            uptime: process.uptime(),
            nodeVersion: process.version,
            platform: process.platform,
            memory: {
                rss: `${Math.round(memoryUsage.rss / 1024 / 1024)}MB`,
                heapTotal: `${Math.round(memoryUsage.heapTotal / 1024 / 1024)}MB`,
                heapUsed: `${Math.round(memoryUsage.heapUsed / 1024 / 1024)}MB`,
                external: `${Math.round(memoryUsage.external / 1024 / 1024)}MB`
            },
            cpu: process.cpuUsage()
        }
    });
});
exports.default = router;
//# sourceMappingURL=healthRoutes.js.map