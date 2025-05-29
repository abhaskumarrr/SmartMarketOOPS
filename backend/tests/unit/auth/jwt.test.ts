/**
 * JWT Utilities Tests
 * Tests for JWT token generation and validation
 */

import jwt from 'jsonwebtoken';
import { 
  generateToken, 
  generateRefreshToken, 
  verifyToken, 
  verifyRefreshToken 
} from '../../../src/utils/jwt';

// Mock the jsonwebtoken module
jest.mock('jsonwebtoken');

describe('JWT Utilities', () => {
  const userId = 'test-user-id';
  const mockToken = 'mock.jwt.token';
  const mockPayload = { id: userId };
  
  beforeEach(() => {
    jest.clearAllMocks();
    // Mock environment variables for testing
    process.env.JWT_SECRET = 'test-jwt-secret';
    process.env.JWT_REFRESH_SECRET = 'test-jwt-refresh-secret';
  });

  describe('generateToken', () => {
    it('should generate a JWT token with the correct payload and options', () => {
      // Mock implementation for sign
      (jwt.sign as jest.Mock).mockReturnValue(mockToken);

      // Call the function
      const token = generateToken(userId);

      // Assertions
      expect(token).toBe(mockToken);
      expect(jwt.sign).toHaveBeenCalledWith(
        { id: userId },
        'test-jwt-secret',
        { expiresIn: '1h' }
      );
    });

    it('should handle token generation errors', () => {
      // Mock implementation for sign to throw an error
      (jwt.sign as jest.Mock).mockImplementation(() => {
        throw new Error('Token generation failed');
      });

      // The function should handle the error gracefully or throw a specific error
      expect(() => generateToken(userId)).toThrow('Token generation failed');
    });
  });

  describe('generateRefreshToken', () => {
    it('should generate a refresh token with the correct payload and options', () => {
      // Mock implementation for sign
      (jwt.sign as jest.Mock).mockReturnValue(mockToken);

      // Call the function
      const token = generateRefreshToken(userId);

      // Assertions
      expect(token).toBe(mockToken);
      expect(jwt.sign).toHaveBeenCalledWith(
        { id: userId },
        'test-jwt-refresh-secret',
        { expiresIn: '7d' }
      );
    });

    it('should use JWT_SECRET if JWT_REFRESH_SECRET is not available', () => {
      // Remove refresh secret
      delete process.env.JWT_REFRESH_SECRET;
      
      // Mock implementation for sign
      (jwt.sign as jest.Mock).mockReturnValue(mockToken);

      // Call the function
      const token = generateRefreshToken(userId);

      // Assertions
      expect(token).toBe(mockToken);
      expect(jwt.sign).toHaveBeenCalledWith(
        { id: userId },
        'test-jwt-secret',  // Should fall back to JWT_SECRET
        { expiresIn: '7d' }
      );
    });
  });

  describe('verifyToken', () => {
    it('should verify a valid token and return the payload', () => {
      // Mock implementation for verify
      (jwt.verify as jest.Mock).mockReturnValue(mockPayload);

      // Call the function
      const result = verifyToken(mockToken);

      // Assertions
      expect(result).toEqual(mockPayload);
      expect(jwt.verify).toHaveBeenCalledWith(mockToken, 'test-jwt-secret');
    });

    it('should return null for an invalid token', () => {
      // Mock implementation for verify to throw an error
      (jwt.verify as jest.Mock).mockImplementation(() => {
        throw new Error('Invalid token');
      });

      // Call the function
      const result = verifyToken(mockToken);

      // Assertions
      expect(result).toBeNull();
      expect(jwt.verify).toHaveBeenCalledWith(mockToken, 'test-jwt-secret');
    });

    it('should handle different types of verification errors', () => {
      // Test expired token
      (jwt.verify as jest.Mock).mockImplementation(() => {
        const error = new Error('jwt expired') as any;
        error.name = 'TokenExpiredError';
        throw error;
      });

      expect(verifyToken(mockToken)).toBeNull();

      // Test malformed token
      (jwt.verify as jest.Mock).mockImplementation(() => {
        const error = new Error('jwt malformed') as any;
        error.name = 'JsonWebTokenError';
        throw error;
      });

      expect(verifyToken(mockToken)).toBeNull();
    });
  });

  describe('verifyRefreshToken', () => {
    it('should verify a valid refresh token and return the payload', () => {
      // Mock implementation for verify
      (jwt.verify as jest.Mock).mockReturnValue(mockPayload);

      // Call the function
      const result = verifyRefreshToken(mockToken);

      // Assertions
      expect(result).toEqual(mockPayload);
      expect(jwt.verify).toHaveBeenCalledWith(mockToken, 'test-jwt-refresh-secret');
    });

    it('should return null for an invalid refresh token', () => {
      // Mock implementation for verify to throw an error
      (jwt.verify as jest.Mock).mockImplementation(() => {
        throw new Error('Invalid token');
      });

      // Call the function
      const result = verifyRefreshToken(mockToken);

      // Assertions
      expect(result).toBeNull();
      expect(jwt.verify).toHaveBeenCalledWith(mockToken, 'test-jwt-refresh-secret');
    });

    it('should use JWT_SECRET if JWT_REFRESH_SECRET is not available', () => {
      // Remove refresh secret
      delete process.env.JWT_REFRESH_SECRET;
      
      // Mock implementation for verify
      (jwt.verify as jest.Mock).mockReturnValue(mockPayload);

      // Call the function
      const result = verifyRefreshToken(mockToken);

      // Assertions
      expect(result).toEqual(mockPayload);
      expect(jwt.verify).toHaveBeenCalledWith(mockToken, 'test-jwt-secret');
    });
  });
}); 