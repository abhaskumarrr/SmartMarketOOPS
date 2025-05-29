# API Key Management Security Documentation

## Overview

This document outlines the security measures implemented in the SmartMarketOOPS API Key Management system. It covers encryption, storage, validation, audit logging, and best practices for secure API key handling.

## Security Architecture

The API key management system follows a defense-in-depth approach with multiple layers of security:

1. **Encryption Layer**: All sensitive API key data is encrypted at rest
2. **Access Control Layer**: Authentication and authorization for all API key operations
3. **Validation Layer**: Verification of API keys before storing or using them
4. **Audit Layer**: Comprehensive logging of all API key operations
5. **Rate Limiting Layer**: Protection against brute force and enumeration attacks

## Encryption Implementation

### Key Material Storage

- The main encryption key (ENCRYPTION_KEY) is stored in environment variables, not in the codebase
- For production, consider using a key management service like AWS KMS or HashiCorp Vault
- The encryption key should be at least 32 bytes of high-entropy data

### Encryption Algorithm

- AES-256-GCM encryption is used for all API key secrets
- Deterministic authenticated encryption (using stable initialization vectors derived from the user ID and key name) 
- This approach enables searching encrypted data while maintaining strong security

### Code Example

```typescript
// In apiKeyEncryption.ts
export async function encryptApiKeyData(
  userId: string,
  keyName: string,
  data: ApiKeySecretData
): Promise<string> {
  // Deterministic IV generation from user ID and key name
  const iv = await generateDeterministicIv(userId, keyName);
  
  // Encrypt the data with AES-256-GCM
  const encryptedData = encrypt(
    JSON.stringify(data),
    process.env.ENCRYPTION_KEY,
    iv
  );
  
  return encryptedData;
}
```

## Data Storage Security

### Database Schema Security

- API key secrets are never stored in plaintext
- The `encryptedData` field contains encrypted JSON with the API secret and other sensitive information
- The visible `key` field only contains a masked version of the API key

### Data Minimization

- Only essential data is stored
- Secrets are stored in encrypted form only
- API responses never include sensitive data

## API Key Lifecycle Management

### Creation

1. Validate the API key against the exchange API
2. Mask the API key for display and storage
3. Encrypt the secret and other sensitive data
4. Store the encrypted data and metadata
5. Log the creation event with an audit trail

### Rotation

1. Validate the new API key credentials
2. Keep the old key active until validation succeeds
3. Encrypt and store the new key data
4. Revoke the old key
5. Log the rotation event with both old and new key references

### Revocation

1. Mark the key as revoked in the database
2. Record the reason for revocation
3. Log the revocation event
4. Optionally notify the user
5. Keep the key record for audit purposes, but prevent its use

## Security Controls

### Rate Limiting

Rate limits are implemented for sensitive operations:

- Standard operations: 100 requests per minute
- API key validation: 10 requests per 5 minutes
- API key management: 20 requests per 10 minutes
- Sensitive operations (rotation): 3 requests per hour

### Input Validation

- All user inputs are validated using express-validator
- Input sanitization is applied to prevent XSS attacks
- Environment values are restricted to an allowed list

### Authentication and Authorization

- All API key endpoints require authentication
- Users can only access their own API keys
- Special operations like rotation require additional verification

## Audit Logging

### Log Format

Each audit log entry includes:

- Timestamp
- User ID
- Action type (create, update, rotate, revoke)
- Resource ID
- Resource type ("ApiKey")
- IP address
- User agent
- Request details (sanitized)
- Success/failure indicator

### Log Storage

- Audit logs are stored in the database
- Critical security events trigger alerts
- Logs are retained according to data retention policies

## Security Testing

Regular security testing includes:

1. Unit tests for encryption functionality
2. Integration tests for API key endpoints
3. Security-specific tests for access control
4. Automated security scans
5. Periodic security audits

## Security Monitoring

The system monitors for suspicious activities:

- Failed API key validations
- Unusual API key creation patterns
- Access attempts from new IP addresses
- Abnormal usage patterns

## Security Best Practices for Users

Users should follow these best practices:

1. Use unique API keys for different applications or services
2. Set appropriate scopes for each key (principle of least privilege)
3. Rotate API keys regularly (at least every 6 months)
4. Revoke unused or compromised keys immediately
5. Use IP restrictions where possible
6. Monitor API key usage regularly

## Incident Response

In case of a security incident:

1. Revoke affected API keys
2. Notify affected users
3. Investigate the root cause
4. Document the incident
5. Implement corrective measures

## Security Audit Script

The system includes a security audit script (`scripts/auditApiKeyStorage.js`) that checks:

- Encryption configuration
- Proper data masking
- Expired keys
- Unrevoked keys
- API key rotation requirements
- Audit logging completeness

To run the audit:

```bash
node scripts/auditApiKeyStorage.js
```

## Regular Security Reviews

The API key management system undergoes regular security reviews:

- Code reviews for security considerations
- Dependency updates and vulnerability scanning
- Encryption and security algorithm reviews
- Access control policy reviews
- Audit logging effectiveness reviews 