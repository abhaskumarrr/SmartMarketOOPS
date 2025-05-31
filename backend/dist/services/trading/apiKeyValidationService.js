"use strict";
/**
 * API Key Validation Service
 * Validates Delta Exchange API keys against the exchange API
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.validateDeltaApiKey = validateDeltaApiKey;
exports.validateTradingPermissions = validateTradingPermissions;
const axios_1 = __importDefault(require("axios"));
const crypto_1 = __importDefault(require("crypto"));
// Delta Exchange API URLs
const DELTA_TESTNET_URL = 'https://testnet-api.delta.exchange';
const DELTA_MAINNET_URL = 'https://api.delta.exchange';
/**
 * Get Delta Exchange base URL based on environment
 * @param {boolean} isTestnet - Whether to use testnet
 * @returns {string} The base URL
 */
function getDeltaBaseUrl(isTestnet) {
    return isTestnet ? DELTA_TESTNET_URL : DELTA_MAINNET_URL;
}
/**
 * Validates an API key and secret against the Delta Exchange API
 * @param {string} apiKey - The API key to validate
 * @param {string} apiSecret - The API secret to validate
 * @param {boolean} isTestnet - Whether to use testnet
 * @returns {Promise<ValidationResult>} Validation result
 */
async function validateDeltaApiKey(apiKey, apiSecret, isTestnet = true) {
    if (!apiKey || !apiSecret) {
        return {
            isValid: false,
            message: 'API key and secret are required'
        };
    }
    try {
        const baseUrl = getDeltaBaseUrl(isTestnet);
        const endpoint = '/v2/wallet/balances';
        const timestamp = Date.now();
        const method = 'GET';
        const requestPath = endpoint;
        // Create signature according to Delta Exchange API documentation
        const message = `${timestamp}${method}${requestPath}`;
        const signature = crypto_1.default
            .createHmac('sha256', apiSecret)
            .update(message)
            .digest('hex');
        // Make API call to validate key
        const response = await (0, axios_1.default)({
            method,
            url: `${baseUrl}${endpoint}`,
            headers: {
                'api-key': apiKey,
                'timestamp': timestamp.toString(),
                'signature': signature
            },
            timeout: 10000 // 10 seconds timeout
        });
        if (response.status === 200) {
            return {
                isValid: true,
                message: 'API key is valid',
                accountInfo: response.data
            };
        }
        else {
            return {
                isValid: false,
                message: `Invalid response: ${response.status} ${response.statusText}`,
                error: response.data
            };
        }
    }
    catch (error) {
        const axiosError = error;
        // Check for specific authorization errors
        if (axiosError.response) {
            if (axiosError.response.status === 401) {
                return {
                    isValid: false,
                    message: 'Invalid API key or secret',
                    error: axiosError.response.data
                };
            }
            else if (axiosError.response.status === 403) {
                return {
                    isValid: false,
                    message: 'API key does not have sufficient permissions',
                    error: axiosError.response.data
                };
            }
            return {
                isValid: false,
                message: `API error: ${axiosError.response.status} ${axiosError.response.statusText}`,
                error: axiosError.response.data
            };
        }
        return {
            isValid: false,
            message: `Connection error: ${axiosError.message || 'Unknown error'}`,
            error: axiosError
        };
    }
}
/**
 * Validates API key trading permissions
 * @param {string} apiKey - The API key to validate
 * @param {string} apiSecret - The API secret to validate
 * @param {boolean} isTestnet - Whether to use testnet
 * @returns {Promise<ValidationResult>} Validation result with permissions
 */
async function validateTradingPermissions(apiKey, apiSecret, isTestnet = true) {
    if (!apiKey || !apiSecret) {
        return {
            isValid: false,
            message: 'API key and secret are required'
        };
    }
    try {
        const baseUrl = getDeltaBaseUrl(isTestnet);
        const endpoint = '/v2/products';
        const timestamp = Date.now();
        const method = 'GET';
        const requestPath = endpoint;
        // Create signature
        const message = `${timestamp}${method}${requestPath}`;
        const signature = crypto_1.default
            .createHmac('sha256', apiSecret)
            .update(message)
            .digest('hex');
        // Test a simple order placement
        const orderEndpoint = '/v2/orders/test';
        const orderMethod = 'POST';
        const orderPath = orderEndpoint;
        const orderTimestamp = Date.now();
        const orderParams = {
            product_id: 1, // BTC-USD for example
            size: 0.001,
            side: 'buy',
            order_type: 'limit',
            limit_price: 1000,
            time_in_force: 'gtc'
        };
        const orderMessage = `${orderTimestamp}${orderMethod}${orderPath}${JSON.stringify(orderParams)}`;
        const orderSignature = crypto_1.default
            .createHmac('sha256', apiSecret)
            .update(orderMessage)
            .digest('hex');
        // Make test order API call to check trading permissions
        const orderResponse = await (0, axios_1.default)({
            method: orderMethod,
            url: `${baseUrl}${orderEndpoint}`,
            headers: {
                'api-key': apiKey,
                'timestamp': orderTimestamp.toString(),
                'signature': orderSignature,
                'Content-Type': 'application/json'
            },
            data: orderParams,
            timeout: 10000 // 10 seconds timeout
        });
        if (orderResponse.status === 200) {
            return {
                isValid: true,
                message: 'API key has trading permissions',
                accountInfo: orderResponse.data
            };
        }
        else {
            return {
                isValid: false,
                message: `Invalid response for trading: ${orderResponse.status} ${orderResponse.statusText}`,
                error: orderResponse.data
            };
        }
    }
    catch (error) {
        const axiosError = error;
        // Check for specific permission errors
        if (axiosError.response) {
            if (axiosError.response.status === 401) {
                return {
                    isValid: false,
                    message: 'Invalid API key or secret',
                    error: axiosError.response.data
                };
            }
            else if (axiosError.response.status === 403) {
                return {
                    isValid: false,
                    message: 'API key does not have trading permissions',
                    error: axiosError.response.data
                };
            }
            return {
                isValid: false,
                message: `Trading API error: ${axiosError.response.status} ${axiosError.response.statusText}`,
                error: axiosError.response.data
            };
        }
        return {
            isValid: false,
            message: `Connection error: ${axiosError.message || 'Unknown error'}`,
            error: axiosError
        };
    }
}
//# sourceMappingURL=apiKeyValidationService.js.map