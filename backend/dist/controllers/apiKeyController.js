"use strict";
/**
 * API Key Controller
 * Wrapper controller that delegates to the trading API key controller
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.validateApiKey = exports.deleteApiKey = exports.getApiKey = exports.getApiKeys = exports.createApiKey = void 0;
const apiKeyController_1 = require("./trading/apiKeyController");
/**
 * Create a new API key
 */
const createApiKey = async (req, res) => {
    await (0, apiKeyController_1.createApiKey)(req, res);
};
exports.createApiKey = createApiKey;
/**
 * Get all API keys for current user
 */
const getApiKeys = async (req, res) => {
    await (0, apiKeyController_1.getAllApiKeys)(req, res);
};
exports.getApiKeys = getApiKeys;
/**
 * Get a specific API key
 */
const getApiKey = async (req, res) => {
    await (0, apiKeyController_1.getApiKeyById)(req, res);
};
exports.getApiKey = getApiKey;
/**
 * Delete an API key
 */
const deleteApiKey = async (req, res) => {
    await (0, apiKeyController_1.revokeApiKey)(req, res);
};
exports.deleteApiKey = deleteApiKey;
/**
 * Validate an API key with Delta Exchange
 */
const validateApiKey = async (req, res) => {
    await (0, apiKeyController_1.validateApiKey)(req, res);
};
exports.validateApiKey = validateApiKey;
//# sourceMappingURL=apiKeyController.js.map