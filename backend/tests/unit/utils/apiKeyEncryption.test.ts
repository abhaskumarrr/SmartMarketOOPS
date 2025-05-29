/**
 * API Key Encryption Utility Tests
 */

import * as apiKeyEncryption from '../../../src/utils/apiKeyEncryption';
import crypto from 'crypto';
import dotenv from 'dotenv';
import path from 'path';

// Load environment variables for the tests
dotenv.config({
  path: path.resolve(__dirname, '../../../../.env')
});

// Ensure ENCRYPTION_KEY is set for tests
if (!process.env.ENCRYPTION_KEY) {
  process.env.ENCRYPTION_KEY = 'aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899';
}

describe('API Key Encryption Utility', () => {
  const testUserId = 'test-user-123';
  const testApiKey = 'abcd1234efgh5678ijkl9012mnop3456qrst7890uvwx';
  const testApiSecret = 'a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2g3h4i5j6k7l8m9n0';
  
  // Test API key data
  const testApiKeyData = {
    key: testApiKey,
    secret: testApiSecret,
    environment: 'testnet',
    label: 'Test API Key'
  };

  describe('hashApiCredential', () => {
    it('should generate a hash from value and salt', () => {
      const salt = 'test-salt';
      const hash = apiKeyEncryption.hashApiCredential(testApiKey, salt);
      
      // Hash should be a hex string of length 64 (SHA-256)
      expect(hash).toMatch(/^[a-f0-9]{64}$/);
      
      // Same input should produce same hash
      const hash2 = apiKeyEncryption.hashApiCredential(testApiKey, salt);
      expect(hash).toEqual(hash2);
      
      // Different salt should produce different hash
      const hash3 = apiKeyEncryption.hashApiCredential(testApiKey, 'different-salt');
      expect(hash).not.toEqual(hash3);
    });
  });

  describe('encryptApiKeyData and decryptApiKeyData', () => {
    it('should encrypt and decrypt API key data correctly', () => {
      // Encrypt the data
      const encrypted = apiKeyEncryption.encryptApiKeyData(testApiKeyData, testUserId);
      
      // Verify encrypted data structure
      expect(encrypted).toHaveProperty('encryptedKey');
      expect(encrypted).toHaveProperty('encryptedSecret');
      expect(encrypted).toHaveProperty('keyHash');
      expect(encrypted).toHaveProperty('secretHash');
      expect(encrypted).toHaveProperty('iv');
      
      // Decrypt the data
      const decrypted = apiKeyEncryption.decryptApiKeyData(encrypted, testUserId);
      
      // Verify decrypted data matches original
      expect(decrypted.key).toEqual(testApiKey);
      expect(decrypted.secret).toEqual(testApiSecret);
    });
    
    it('should throw error when attempting to decrypt with wrong user ID', () => {
      // Encrypt with one user ID
      const encrypted = apiKeyEncryption.encryptApiKeyData(testApiKeyData, testUserId);
      
      // Attempt to decrypt with different user ID
      expect(() => {
        apiKeyEncryption.decryptApiKeyData(encrypted, 'wrong-user-id');
      }).toThrow();
    });
  });

  describe('validateApiKeyFormat', () => {
    it('should validate correct API key formats', () => {
      // Valid formats
      expect(apiKeyEncryption.validateApiKeyFormat(testApiKey)).toBe(true);
      expect(apiKeyEncryption.validateApiKeyFormat('abcd1234-efgh-5678-ijkl-9012mnop3456')).toBe(true);
      
      // Invalid formats
      expect(apiKeyEncryption.validateApiKeyFormat('')).toBe(false);
      expect(apiKeyEncryption.validateApiKeyFormat('too-short')).toBe(false);
      expect(apiKeyEncryption.validateApiKeyFormat('invalid@characters!')).toBe(false);
      expect(apiKeyEncryption.validateApiKeyFormat(null as any)).toBe(false);
      expect(apiKeyEncryption.validateApiKeyFormat(undefined as any)).toBe(false);
    });
  });

  describe('validateApiSecretFormat', () => {
    it('should validate correct API secret formats', () => {
      // Valid formats
      expect(apiKeyEncryption.validateApiSecretFormat(testApiSecret)).toBe(true);
      
      // Invalid formats
      expect(apiKeyEncryption.validateApiSecretFormat('')).toBe(false);
      expect(apiKeyEncryption.validateApiSecretFormat('too-short')).toBe(false);
      expect(apiKeyEncryption.validateApiSecretFormat('invalid@characters!')).toBe(false);
      expect(apiKeyEncryption.validateApiSecretFormat(null as any)).toBe(false);
      expect(apiKeyEncryption.validateApiSecretFormat(undefined as any)).toBe(false);
    });
  });

  describe('maskApiKey', () => {
    it('should mask API keys correctly', () => {
      // Test with a standard key
      const masked = apiKeyEncryption.maskApiKey('abcdefghijklmnop');
      expect(masked).toBe('abcd****mnop');
      
      // Test with shorter key
      expect(apiKeyEncryption.maskApiKey('abcdef')).toBe('****');
      
      // Test with null/undefined
      expect(apiKeyEncryption.maskApiKey(null as any)).toBe('****');
      expect(apiKeyEncryption.maskApiKey(undefined as any)).toBe('****');
    });
  });
}); 