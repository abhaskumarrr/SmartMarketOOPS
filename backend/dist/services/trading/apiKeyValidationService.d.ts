/**
 * API Key Validation Service
 * Validates Delta Exchange API keys against the exchange API
 */
/**
 * API key validation response
 */
interface ValidationResult {
    isValid: boolean;
    message: string;
    accountInfo?: any;
    error?: any;
}
/**
 * Validates an API key and secret against the Delta Exchange API
 * @param {string} apiKey - The API key to validate
 * @param {string} apiSecret - The API secret to validate
 * @param {boolean} isTestnet - Whether to use testnet
 * @returns {Promise<ValidationResult>} Validation result
 */
declare function validateDeltaApiKey(apiKey: string, apiSecret: string, isTestnet?: boolean): Promise<ValidationResult>;
/**
 * Validates API key trading permissions
 * @param {string} apiKey - The API key to validate
 * @param {string} apiSecret - The API secret to validate
 * @param {boolean} isTestnet - Whether to use testnet
 * @returns {Promise<ValidationResult>} Validation result with permissions
 */
declare function validateTradingPermissions(apiKey: string, apiSecret: string, isTestnet?: boolean): Promise<ValidationResult>;
export { validateDeltaApiKey, validateTradingPermissions, ValidationResult };
