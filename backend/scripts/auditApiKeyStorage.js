/**
 * API Key Storage Security Audit Script
 * 
 * This script performs a security audit on the API key storage
 * It checks for any security issues and provides recommendations
 * 
 * Usage: node scripts/auditApiKeyStorage.js
 */

const { PrismaClient } = require('@prisma/client');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

const prisma = new PrismaClient();

/**
 * Colors for console output
 */
const colors = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m',
  white: '\x1b[37m',
};

/**
 * Check if a string contains a raw (unencrypted) API key
 * @param {string} str - String to check
 * @returns {boolean} - True if the string contains a possible API key
 */
function containsRawApiKey(str) {
  // Pattern for API keys - this should be adjusted based on your specific format
  const apiKeyPattern = /\b[A-Za-z0-9]{20,64}\b/;
  return apiKeyPattern.test(str);
}

/**
 * Check if data is encrypted
 * @param {string} data - Data to check
 * @returns {boolean} - True if the data appears to be encrypted
 */
function isEncrypted(data) {
  // Encrypted data usually has high entropy and contains non-printable characters
  const entropy = calculateEntropy(data);
  
  // Encrypted data typically has an entropy value close to 8.0
  return entropy > 7.0 && /[^\x20-\x7E]/.test(data);
}

/**
 * Calculate Shannon entropy of a string
 * @param {string} str - String to calculate entropy for
 * @returns {number} - Entropy value
 */
function calculateEntropy(str) {
  const len = str.length;
  const frequencies = {};
  
  // Count frequency of each character
  for (let i = 0; i < len; i++) {
    const char = str.charAt(i);
    frequencies[char] = (frequencies[char] || 0) + 1;
  }
  
  // Calculate entropy
  return Object.values(frequencies).reduce((entropy, freq) => {
    const p = freq / len;
    return entropy - p * Math.log2(p);
  }, 0);
}

/**
 * Run the security audit
 */
async function runAudit() {
  console.log(`${colors.blue}Starting API Key Storage Security Audit${colors.reset}`);
  console.log('===========================================\n');
  
  const issues = [];
  const recommendations = [];
  
  try {
    // Check if encryption key is properly set
    console.log(`${colors.cyan}Checking encryption configuration...${colors.reset}`);
    
    if (!process.env.ENCRYPTION_KEY) {
      issues.push('ENCRYPTION_KEY environment variable is not set');
      recommendations.push('Set a strong ENCRYPTION_KEY in your environment variables');
    } else if (process.env.ENCRYPTION_KEY.length < 32) {
      issues.push('ENCRYPTION_KEY is too short (less than 32 characters)');
      recommendations.push('Use a stronger ENCRYPTION_KEY with at least 32 characters');
    }
    
    // Check database for API keys
    console.log(`${colors.cyan}Checking API keys in database...${colors.reset}`);
    
    const apiKeys = await prisma.apiKey.findMany({
      select: {
        id: true,
        key: true,
        encryptedData: true,
        createdAt: true,
        updatedAt: true,
        expiry: true,
        isRevoked: true,
        environment: true,
      }
    });
    
    console.log(`Found ${apiKeys.length} API keys in the database.`);
    
    // Check for unencrypted API keys
    const unencryptedKeys = apiKeys.filter(key => 
      !key.encryptedData || 
      !isEncrypted(key.encryptedData)
    );
    
    if (unencryptedKeys.length > 0) {
      issues.push(`Found ${unencryptedKeys.length} API keys with potentially unencrypted data`);
      recommendations.push('Rotate affected API keys and ensure encryption is working properly');
    }
    
    // Check for expired but not revoked keys
    const now = new Date();
    const expiredKeys = apiKeys.filter(key => 
      key.expiry && new Date(key.expiry) < now && !key.isRevoked
    );
    
    if (expiredKeys.length > 0) {
      issues.push(`Found ${expiredKeys.length} expired API keys that have not been revoked`);
      recommendations.push('Automatically revoke expired API keys');
    }
    
    // Check for unmasked API keys
    const potentiallyUnmaskedKeys = apiKeys.filter(key => 
      containsRawApiKey(key.key)
    );
    
    if (potentiallyUnmaskedKeys.length > 0) {
      issues.push(`Found ${potentiallyUnmaskedKeys.length} API keys that may not be properly masked`);
      recommendations.push('Ensure all API keys are properly masked before storing');
    }
    
    // Check for old keys that need rotation
    const sixMonthsAgo = new Date();
    sixMonthsAgo.setMonth(sixMonthsAgo.getMonth() - 6);
    
    const oldKeys = apiKeys.filter(key => 
      new Date(key.createdAt) < sixMonthsAgo && !key.isRevoked
    );
    
    if (oldKeys.length > 0) {
      issues.push(`Found ${oldKeys.length} API keys older than 6 months that should be rotated`);
      recommendations.push('Implement a key rotation policy for keys older than 6 months');
    }
    
    // Check database schema for sensitive fields
    console.log(`${colors.cyan}Checking database schema...${colors.reset}`);
    
    const prismaSchema = fs.readFileSync(
      path.join(__dirname, '../prisma/schema.prisma'),
      'utf8'
    );
    
    if (!prismaSchema.includes('@db.Encrypted') && !prismaSchema.includes('encryptedData')) {
      issues.push('Database schema may not be properly configured for encrypted data');
      recommendations.push('Ensure sensitive fields are encrypted or use encrypted JSON data');
    }
    
    // Check for audit logging
    console.log(`${colors.cyan}Checking audit logging...${colors.reset}`);
    
    const auditLogs = await prisma.auditLog.findMany({
      where: {
        resourceType: 'ApiKey'
      },
      take: 1
    });
    
    if (auditLogs.length === 0) {
      issues.push('No audit logs found for API key operations');
      recommendations.push('Implement comprehensive audit logging for all API key operations');
    }
    
    // Print results
    console.log('\n===========================================');
    if (issues.length === 0) {
      console.log(`${colors.green}No security issues found!${colors.reset}`);
    } else {
      console.log(`${colors.red}Found ${issues.length} security issues:${colors.reset}`);
      issues.forEach((issue, index) => {
        console.log(`${colors.red}${index + 1}. ${issue}${colors.reset}`);
      });
      
      console.log(`\n${colors.yellow}Recommendations:${colors.reset}`);
      recommendations.forEach((rec, index) => {
        console.log(`${colors.yellow}${index + 1}. ${rec}${colors.reset}`);
      });
    }
    
    // Save report to file
    const report = {
      timestamp: new Date().toISOString(),
      apiKeyCount: apiKeys.length,
      issues,
      recommendations
    };
    
    fs.writeFileSync(
      path.join(__dirname, '../logs/api-key-security-audit.json'),
      JSON.stringify(report, null, 2)
    );
    
    console.log(`\n${colors.green}Audit complete. Report saved to logs/api-key-security-audit.json${colors.reset}`);
    
  } catch (error) {
    console.error(`${colors.red}Error during audit:${colors.reset}`, error);
  } finally {
    await prisma.$disconnect();
  }
}

// Run the audit
runAudit(); 