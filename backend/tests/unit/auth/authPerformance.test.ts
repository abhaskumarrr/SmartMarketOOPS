/**
 * Authentication Performance Tests
 * Tests focused on performance aspects of the authentication system
 */

import { Request, Response } from 'express';
import { PrismaClient } from '@prisma/client';
import bcrypt from 'bcryptjs';
import * as authController from '../../../src/controllers/authController';
import { protect } from '../../../src/middleware/auth';
import * as jwt from '../../../src/utils/jwt';
import { AuthenticatedRequest } from '../../../src/types/auth';
import { performance } from 'perf_hooks';

// Mock dependencies
jest.mock('@prisma/client', () => {
  const mockPrismaClient = {
    user: {
      findUnique: jest.fn(),
      findFirst: jest.fn(),
      create: jest.fn(),
      update: jest.fn()
    },
    $disconnect: jest.fn()
  };
  return {
    PrismaClient: jest.fn(() => mockPrismaClient)
  };
});

jest.mock('bcryptjs', () => ({
  genSalt: jest.fn().mockResolvedValue('salt'),
  hash: jest.fn().mockResolvedValue('hashedPassword'),
  compare: jest.fn().mockResolvedValue(true)
}));

jest.mock('../../../src/utils/jwt', () => ({
  verifyToken: jest.fn(),
  generateToken: jest.fn().mockReturnValue('test-token'),
  generateRefreshToken: jest.fn().mockReturnValue('test-refresh-token')
}));

describe('Authentication Performance', () => {
  let mockRequest: Partial<AuthenticatedRequest>;
  let mockResponse: Partial<Response>;
  let nextFunction: jest.Mock;
  let prisma: any;

  beforeEach(() => {
    mockRequest = {
      headers: {},
      body: {},
      params: {}
    };
    
    mockResponse = {
      status: jest.fn().mockReturnThis(),
      json: jest.fn()
    };
    
    nextFunction = jest.fn();
    
    prisma = new PrismaClient();
    
    jest.clearAllMocks();
  });

  describe('Token Validation Performance', () => {
    it('should verify tokens efficiently under load', async () => {
      // Setup - prepare a valid token request
      mockRequest.headers = { authorization: 'Bearer valid.token.value' };
      
      // Mock the token verification to return a valid user payload
      (jwt.verifyToken as jest.Mock).mockReturnValue({ id: 'user-123' });
      
      // Mock finding the user
      (prisma.user.findUnique as jest.Mock).mockResolvedValue({
        id: 'user-123',
        name: 'Test User',
        email: 'test@example.com',
        role: 'user',
        isVerified: true
      });
      
      // Measure the performance of token validation under load
      const iterations = 100;
      const startTime = performance.now();
      
      // Run multiple iterations to simulate load
      for (let i = 0; i < iterations; i++) {
        await protect(mockRequest as AuthenticatedRequest, mockResponse as Response, nextFunction);
        // Reset the next function for the next iteration
        nextFunction.mockClear();
      }
      
      const endTime = performance.now();
      const executionTime = endTime - startTime;
      const avgExecutionTime = executionTime / iterations;
      
      console.log(`Average token validation time: ${avgExecutionTime.toFixed(2)} ms`);
      
      // Token validation should be fast (typically under 10ms per operation)
      // This threshold can be adjusted based on environment and requirements
      expect(avgExecutionTime).toBeLessThan(10); // less than 10ms per validation on average
    });
  });

  describe('Login Performance', () => {
    it('should handle concurrent login requests efficiently', async () => {
      // Setup - prepare login request data
      mockRequest.body = {
        email: 'test@example.com',
        password: 'Password123!'
      };
      
      // Mock user lookup
      (prisma.user.findUnique as jest.Mock).mockResolvedValue({
        id: 'user-123',
        name: 'Test User',
        email: 'test@example.com',
        password: 'hashedPassword',
        role: 'user',
        isVerified: true
      });
      
      // Measure the performance of login processing under load
      const iterations = 50;
      const startTime = performance.now();
      
      // Create promises for concurrent login requests
      const loginPromises = Array(iterations).fill(0).map(() => 
        authController.login(mockRequest as Request, mockResponse as Response)
      );
      
      // Wait for all login requests to complete
      await Promise.all(loginPromises);
      
      const endTime = performance.now();
      const executionTime = endTime - startTime;
      const avgExecutionTime = executionTime / iterations;
      
      console.log(`Average login time: ${avgExecutionTime.toFixed(2)} ms`);
      
      // Login should be relatively fast, but will be slower than token validation
      // due to password comparison and other operations
      expect(avgExecutionTime).toBeLessThan(100); // less than 100ms per login on average
    });
  });

  describe('Password Hashing Performance', () => {
    it('should hash passwords with appropriate cost factor', async () => {
      // Setup - mock bcrypt salt and hash functions
      const password = 'Password123!';
      
      // Reset the mock implementation to measure actual performance
      (bcrypt.genSalt as jest.Mock).mockRestore();
      (bcrypt.hash as jest.Mock).mockRestore();
      
      // Measure the performance of password hashing
      const startTime = performance.now();
      
      // Generate salt and hash the password
      const salt = await bcrypt.genSalt(10); // Default cost factor
      const hash = await bcrypt.hash(password, salt);
      
      const endTime = performance.now();
      const executionTime = endTime - startTime;
      
      console.log(`Password hashing time (cost factor 10): ${executionTime.toFixed(2)} ms`);
      
      // Password hashing should take between 200ms and 1000ms
      // Too fast means insufficient security, too slow means poor user experience
      expect(executionTime).toBeGreaterThan(10); // Should take some time for security
      expect(executionTime).toBeLessThan(2000); // But not too much time for UX
      
      // Verify the hash was generated (not just mocked)
      expect(hash).toBeTruthy();
      expect(hash).not.toBe(password);
      
      // Reset the mocks for other tests
      jest.mock('bcryptjs', () => ({
        genSalt: jest.fn().mockResolvedValue('salt'),
        hash: jest.fn().mockResolvedValue('hashedPassword'),
        compare: jest.fn().mockResolvedValue(true)
      }));
    });
  });

  describe('Authentication Scalability', () => {
    it('should handle many concurrent token verifications', async () => {
      // Setup - mock token verification
      (jwt.verifyToken as jest.Mock).mockReturnValue({ id: 'user-123' });
      
      // Mock finding the user
      (prisma.user.findUnique as jest.Mock).mockResolvedValue({
        id: 'user-123',
        name: 'Test User',
        email: 'test@example.com',
        role: 'user',
        isVerified: true
      });
      
      // Create many mock requests with auth headers
      const numRequests = 500;
      const requests = Array(numRequests).fill(0).map(() => ({
        headers: { authorization: 'Bearer valid.token.value' }
      }));
      
      // Measure performance of many concurrent verifications
      const startTime = performance.now();
      
      // Process all requests concurrently
      await Promise.all(
        requests.map(req => 
          protect(req as AuthenticatedRequest, mockResponse as Response, nextFunction)
        )
      );
      
      const endTime = performance.now();
      const totalTime = endTime - startTime;
      
      console.log(`Processed ${numRequests} concurrent auth requests in ${totalTime.toFixed(2)} ms`);
      
      // For 500 requests, total time should scale reasonably
      // This threshold can be adjusted based on environment
      expect(totalTime).toBeLessThan(5000); // Less than 5 seconds for 500 requests
      
      // Verify that user loading was properly called
      expect(prisma.user.findUnique).toHaveBeenCalledTimes(numRequests);
    });
  });

  // Optionally: memory usage tests could be added here if needed
}); 