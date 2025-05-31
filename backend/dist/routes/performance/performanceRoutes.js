"use strict";
/**
 * Performance Routes
 * Routes for performance testing
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const authMiddleware_1 = require("../../middleware/authMiddleware");
const performanceController_1 = require("../../controllers/performance/performanceController");
const router = express_1.default.Router();
// All routes require authentication
router.use(authMiddleware_1.authenticateJWT);
// Performance test routes
router.post('/tests', performanceController_1.createTest);
router.get('/tests', performanceController_1.getAllTests);
router.get('/tests/:id', performanceController_1.getTest);
router.post('/tests/:id/start', performanceController_1.startTest);
router.get('/tests/:id/results', performanceController_1.getTestResults);
// Test result routes
router.get('/results/:id', performanceController_1.getTestResult);
// Load test routes
router.post('/load-test', performanceController_1.runLoadTest);
exports.default = router;
//# sourceMappingURL=performanceRoutes.js.map