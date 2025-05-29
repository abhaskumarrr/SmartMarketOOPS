/**
 * Key management utility for securely handling encryption keys
 * Handles master key retrieval and validation
 */

import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

// Expected length of the master key (32 bytes / 256 bits)
const MASTER_KEY_LENGTH = 32;

/**
 * Retrieves the master encryption key from environment or creates a secure one
 * @returns {Buffer} The master encryption key as a Buffer
 */
export function getMasterKey(): Buffer {
  // Try to get the key from environment variables first
  const envKey = process.env.ENCRYPTION_MASTER_KEY;
  
  if (envKey) {
    // Convert hex string to Buffer
    const masterKey = Buffer.from(envKey, 'hex');
    
    // Validate the key length
    if (masterKey.length !== MASTER_KEY_LENGTH) {
      throw new Error(`Master encryption key must be ${MASTER_KEY_LENGTH} bytes (${MASTER_KEY_LENGTH * 8} bits)`);
    }
    
    return masterKey;
  }
  
  // If not in environment, check for a key file
  const keyFilePath = path.join(__dirname, '..', '..', '.keys', 'master.key');
  
  try {
    // Check if the key file exists and try to read it
    if (fs.existsSync(keyFilePath)) {
      const fileKey = fs.readFileSync(keyFilePath);
      
      if (fileKey.length !== MASTER_KEY_LENGTH) {
        throw new Error(`Master key file contains invalid key length: ${fileKey.length}, expected ${MASTER_KEY_LENGTH}`);
      }
      
      return fileKey;
    }
  } catch (error) {
    console.error('Error reading master key file:', error instanceof Error ? error.message : error);
  }
  
  // If we reach here, we need to generate a new key
  console.warn('No master encryption key found. Generating a new one.');
  
  // Generate a secure random key
  const newKey = crypto.randomBytes(MASTER_KEY_LENGTH);
  
  try {
    // Create directory if it doesn't exist
    const dirPath = path.join(__dirname, '..', '..', '.keys');
    if (!fs.existsSync(dirPath)) {
      fs.mkdirSync(dirPath, { recursive: true });
    }
    
    // Write the key to the file
    fs.writeFileSync(keyFilePath, newKey);
    
    // Output the key to console for setting in environment variables
    console.info('New master encryption key generated. Please set this in your environment variables:');
    console.info(`ENCRYPTION_MASTER_KEY=${newKey.toString('hex')}`);
    
    // Also output instructions to set the key in .env
    console.info('Add this line to your .env file:');
    console.info(`ENCRYPTION_MASTER_KEY=${newKey.toString('hex')}`);
    
    return newKey;
  } catch (error) {
    console.error('Failed to save master key:', error instanceof Error ? error.message : error);
    console.info('Using generated key for this session only.');
    return newKey;
  }
}

/**
 * Derives a specific key for a particular purpose to avoid key reuse
 * @param {Buffer} masterKey - The master encryption key
 * @param {string} purpose - The purpose identifier (e.g., 'api-keys', 'user-data')
 * @param {string} [id=''] - Optional ID to further distinguish the derived key
 * @returns {Buffer} A derived key specific to the purpose
 */
export function deriveKey(masterKey: Buffer, purpose: string, id: string = ''): Buffer {
  if (!masterKey || masterKey.length !== MASTER_KEY_LENGTH) {
    throw new Error('Valid master key required');
  }
  
  if (!purpose) {
    throw new Error('Purpose is required for key derivation');
  }
  
  // Use a deterministic salt based on purpose and ID
  // This ensures the same key is derived for the same inputs
  const saltInput = `${purpose}:${id}:salt`;
  const salt = crypto.createHash('sha256').update(saltInput).digest().slice(0, 16);
  
  // Use HKDF (HMAC-based Key Derivation Function) for deriving keys
  const info = Buffer.from(`${purpose}:${id}`);
  
  // In Node.js, we can implement a simplified HKDF
  const hmac = crypto.createHmac('sha256', salt);
  hmac.update(masterKey);
  hmac.update(info);
  
  return hmac.digest().slice(0, MASTER_KEY_LENGTH);
}

export {
  MASTER_KEY_LENGTH
}; 