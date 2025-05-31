"use strict";
/**
 * Strategy Routes
 * Routes for strategy management and execution
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const authMiddleware_1 = require("../middleware/authMiddleware");
const strategyController_1 = require("../controllers/strategyController");
const router = express_1.default.Router();
// Strategy routes
router.get('/strategies', authMiddleware_1.authenticateJWT, strategyController_1.getAllStrategies);
router.get('/strategies/:id', authMiddleware_1.authenticateJWT, strategyController_1.getStrategyById);
router.post('/strategies', authMiddleware_1.authenticateJWT, strategyController_1.createStrategy);
router.put('/strategies/:id', authMiddleware_1.authenticateJWT, strategyController_1.updateStrategy);
router.delete('/strategies/:id', authMiddleware_1.authenticateJWT, strategyController_1.deleteStrategy);
router.post('/strategies/validate', authMiddleware_1.authenticateJWT, strategyController_1.validateStrategy);
router.post('/strategies/:id/execute', authMiddleware_1.authenticateJWT, strategyController_1.startStrategyExecution);
// Execution routes
router.get('/executions', authMiddleware_1.authenticateJWT, strategyController_1.getUserExecutions);
router.get('/executions/:id', authMiddleware_1.authenticateJWT, strategyController_1.getExecutionById);
router.get('/executions/:id/results', authMiddleware_1.authenticateJWT, strategyController_1.getExecutionResults);
router.post('/executions/:id/stop', authMiddleware_1.authenticateJWT, strategyController_1.stopStrategyExecution);
exports.default = router;
//# sourceMappingURL=strategyRoutes.js.map