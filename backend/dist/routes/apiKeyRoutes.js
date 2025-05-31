"use strict";
/**
 * API Key Routes
 * Endpoints for managing Delta Exchange API keys
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const apiKeyController_1 = require("../controllers/apiKeyController");
const auth_1 = require("../middleware/auth");
const router = express_1.default.Router();
// All routes require authentication
router.use(auth_1.protect);
// All routes require verified email
router.use(auth_1.requireVerified);
// Create a new API key
router.post('/', (0, auth_1.requirePermission)(['apiKeys:create']), apiKeyController_1.createApiKey);
// Get all API keys for current user
router.get('/', (0, auth_1.requirePermission)(['apiKeys:read']), apiKeyController_1.getApiKeys);
// Get a specific API key
router.get('/:id', (0, auth_1.requirePermission)(['apiKeys:read']), apiKeyController_1.getApiKey);
// Delete an API key
router.delete('/:id', (0, auth_1.requirePermission)(['apiKeys:delete']), apiKeyController_1.deleteApiKey);
// Validate an API key with Delta Exchange
router.post('/validate', (0, auth_1.requirePermission)(['apiKeys:read']), apiKeyController_1.validateApiKey);
exports.default = router;
//# sourceMappingURL=apiKeyRoutes.js.map