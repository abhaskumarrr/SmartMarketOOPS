"use strict";
/**
 * Bridge Routes
 * Routes for the ML-Trading bridge API
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const authMiddleware_1 = require("../../middleware/authMiddleware");
const bridgeController_1 = require("../../controllers/bridge/bridgeController");
const router = express_1.default.Router();
// Prediction routes
router.post('/predict-and-signal', authMiddleware_1.authenticateJWT, bridgeController_1.getPredictionAndGenerateSignal);
router.post('/predict', authMiddleware_1.authenticateJWT, bridgeController_1.getPrediction);
router.post('/predict-batch', authMiddleware_1.authenticateJWT, bridgeController_1.getBatchPredictions);
router.get('/predictions/:id', authMiddleware_1.authenticateJWT, bridgeController_1.getPredictionById);
// Model routes
router.get('/models', authMiddleware_1.authenticateJWT, bridgeController_1.getAllModels);
router.get('/models/:id', authMiddleware_1.authenticateJWT, bridgeController_1.getModelById);
router.get('/models/:id/features', authMiddleware_1.authenticateJWT, bridgeController_1.getFeatureImportance);
// Training routes
router.post('/training', authMiddleware_1.authenticateJWT, bridgeController_1.startTraining);
router.get('/training/:id', authMiddleware_1.authenticateJWT, bridgeController_1.getTrainingStatus);
router.delete('/training/:id', authMiddleware_1.authenticateJWT, bridgeController_1.cancelTraining);
// Backtest routes
router.post('/backtest', authMiddleware_1.authenticateJWT, bridgeController_1.runBacktest);
// Health routes
router.get('/health', authMiddleware_1.authenticateJWT, bridgeController_1.getBridgeHealth);
router.get('/ml-health', authMiddleware_1.authenticateJWT, bridgeController_1.checkMLConnection);
exports.default = router;
//# sourceMappingURL=bridgeRoutes.js.map