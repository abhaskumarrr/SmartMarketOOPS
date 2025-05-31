/**
 * Key management utility for securely handling encryption keys
 * Handles master key retrieval and validation
 */
declare const MASTER_KEY_LENGTH = 32;
/**
 * Retrieves the master encryption key from environment or creates a secure one
 * @returns {Buffer} The master encryption key as a Buffer
 */
export declare function getMasterKey(): Buffer;
/**
 * Derives a specific key for a particular purpose to avoid key reuse
 * @param {Buffer} masterKey - The master encryption key
 * @param {string} purpose - The purpose identifier (e.g., 'api-keys', 'user-data')
 * @param {string} [id=''] - Optional ID to further distinguish the derived key
 * @returns {Buffer} A derived key specific to the purpose
 */
export declare function deriveKey(masterKey: Buffer, purpose: string, id?: string): Buffer;
export { MASTER_KEY_LENGTH };
