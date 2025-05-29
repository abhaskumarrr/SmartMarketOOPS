/**
 * Encryption utility for secure data storage
 * Provides functions to encrypt and decrypt sensitive data
 */

const crypto = require('crypto');
const dotenv = require('dotenv');
const path = require('path');

// Load environment variables from .env file
dotenv.config({
  path: path.resolve(__dirname, '../../../.env')
});

// Algorithm constants
const ALGORITHM = 'aes-256-gcm';
const IV_LENGTH = 16; // For AES, this is always 16 bytes
const AUTH_TAG_LENGTH = 16;
const SALT_LENGTH = 64;
const KEY_LENGTH = 32; // 256 bits
const ITERATIONS = 100000;

/**
 * Get the encryption key from environment variable
 * @returns {Buffer} The encryption key as a Buffer
 */
const getEncryptionKey = () => {
  const key = process.env.ENCRYPTION_KEY;
  if (!key) {
    throw new Error('ENCRYPTION_KEY environment variable is not set');
  }
  return Buffer.from(key, 'hex');
};

/**
 * Encrypt a string or object using AES-256-GCM
 * @param {string|object} data - Data to encrypt (strings or objects that can be JSON stringified)
 * @param {Buffer} [customKey] - Optional custom encryption key
 * @returns {string} Encrypted data as a hex string
 */
const encrypt = (data, customKey) => {
  try {
    // Convert objects to strings
    const dataString = typeof data === 'object' ? JSON.stringify(data) : String(data);
    
    // Generate a random initialization vector
    const iv = crypto.randomBytes(IV_LENGTH);
    
    // Get the encryption key
    const key = customKey || getEncryptionKey();
    
    // Create cipher with key and IV
    const cipher = crypto.createCipheriv(ALGORITHM, key, iv);
    
    // Encrypt the data
    let encrypted = cipher.update(dataString, 'utf8', 'hex');
    encrypted += cipher.final('hex');
    
    // Get the auth tag (for GCM mode)
    const authTag = cipher.getAuthTag().toString('hex');
    
    // Format: iv:authTag:encryptedData
    return `${iv.toString('hex')}:${authTag}:${encrypted}`;
  } catch (error) {
    console.error('Encryption error:', error);
    throw new Error('Failed to encrypt data');
  }
};

/**
 * Decrypt data that was encrypted with the encrypt function
 * @param {string} encryptedData - The encrypted data string (iv:authTag:encryptedData)
 * @param {Buffer} [customKey] - Optional custom decryption key
 * @returns {string|object} The decrypted data (parsed as JSON if it was an object)
 */
const decrypt = (encryptedData, customKey) => {
  try {
    // Split the encrypted data into components
    const [ivHex, authTagHex, encryptedHex] = encryptedData.split(':');
    
    if (!ivHex || !authTagHex || !encryptedHex) {
      throw new Error('Invalid encrypted data format');
    }
    
    // Convert hex strings back to buffers
    const iv = Buffer.from(ivHex, 'hex');
    const authTag = Buffer.from(authTagHex, 'hex');
    
    // Get the encryption key
    const key = customKey || getEncryptionKey();
    
    // Create decipher
    const decipher = crypto.createDecipheriv(ALGORITHM, key, iv);
    decipher.setAuthTag(authTag);
    
    // Decrypt the data
    let decrypted = decipher.update(encryptedHex, 'hex', 'utf8');
    decrypted += decipher.final('utf8');
    
    // Try to parse as JSON if it looks like a JSON string
    if (decrypted.startsWith('{') || decrypted.startsWith('[')) {
      try {
        return JSON.parse(decrypted);
      } catch (e) {
        // If it's not valid JSON, return as is
        return decrypted;
      }
    }
    
    return decrypted;
  } catch (error) {
    console.error('Decryption error:', error);
    throw new Error('Failed to decrypt data');
  }
};

/**
 * Hash a password with salt for secure storage
 * @param {string} password - The password to hash
 * @returns {string} The salted hash in format salt:hash
 */
const hashPassword = (password) => {
  // Generate a random salt
  const salt = crypto.randomBytes(SALT_LENGTH).toString('hex');
  
  // Hash the password with the salt
  const hash = crypto.pbkdf2Sync(
    password,
    salt,
    ITERATIONS,
    KEY_LENGTH,
    'sha512'
  ).toString('hex');
  
  // Return salt:hash
  return `${salt}:${hash}`;
};

/**
 * Verify a password against a stored hash
 * @param {string} password - The password to verify
 * @param {string} storedHash - The stored hash (salt:hash)
 * @returns {boolean} True if the password matches
 */
const verifyPassword = (password, storedHash) => {
  const [salt, hash] = storedHash.split(':');
  
  if (!salt || !hash) {
    return false;
  }
  
  // Hash the password with the same salt
  const calculatedHash = crypto.pbkdf2Sync(
    password,
    salt,
    ITERATIONS,
    KEY_LENGTH,
    'sha512'
  ).toString('hex');
  
  // Compare calculated hash with stored hash
  return calculatedHash === hash;
};

module.exports = {
  encrypt,
  decrypt,
  hashPassword,
  verifyPassword
}; 