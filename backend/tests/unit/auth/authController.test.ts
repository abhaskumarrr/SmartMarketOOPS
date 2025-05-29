/**
 * Authentication Controller Tests
 * Tests for authentication-related controllers
 */

import { Request, Response } from 'express';
import bcrypt from 'bcryptjs';
import { PrismaClient } from '@prisma/client';
import * as jwt from '../../../src/utils/jwt';
import * as authController from '../../../src/controllers/authController';
import * as emailUtils from '../../../src/utils/email';

// Mock the dependencies
jest.mock('@prisma/client', () => {
  const mockPrismaClient = {
    user: {
      findUnique: jest.fn(),
      findFirst: jest.fn(),
      create: jest.fn(),
      update: jest.fn()
    },
    session: {
      create: jest.fn(),
      findUnique: jest.fn(),
      delete: jest.fn(),
      deleteMany: jest.fn()
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
  compare: jest.fn()
}));

jest.mock('../../../src/utils/jwt', () => ({
  generateToken: jest.fn().mockReturnValue('test-token'),
  generateRefreshToken: jest.fn().mockReturnValue('test-refresh-token'),
  verifyToken: jest.fn(),
  verifyRefreshToken: jest.fn()
}));

jest.mock('../../../src/utils/email', () => ({
  sendEmail: jest.fn().mockResolvedValue(true)
}));

jest.mock('../../../src/utils/sessionManager', () => ({
  createSession: jest.fn().mockResolvedValue({
    token: 'test-token',
    refreshToken: 'test-refresh-token',
    session: { id: 'test-session-id' }
  }),
  invalidateSession: jest.fn().mockResolvedValue(true),
  invalidateAllUserSessions: jest.fn().mockResolvedValue(true)
}));

jest.mock('crypto', () => ({
  randomBytes: jest.fn().mockReturnValue({
    toString: jest.fn().mockReturnValue('test-verification-token')
  })
}));

describe('Auth Controller', () => {
  let mockRequest: Partial<Request>;
  let mockResponse: Partial<Response>;
  let prisma: any;

  beforeEach(() => {
    // Setup request and response objects
    mockRequest = {
      body: {},
      params: {},
      headers: {},
      cookies: {}
    };
    
    mockResponse = {
      status: jest.fn().mockReturnThis(),
      json: jest.fn(),
      cookie: jest.fn()
    };
    
    // Get the mocked prisma instance
    prisma = new PrismaClient();
    
    // Reset all mocks before each test
    jest.clearAllMocks();
  });

  describe('register', () => {
    it('should register a new user successfully', async () => {
      // Setup
      mockRequest.body = {
        name: 'Test User',
        email: 'test@example.com',
        password: 'Password123!'
      };
      
      // Mock user doesn't exist yet
      (prisma.user.findUnique as jest.Mock).mockResolvedValue(null);
      
      // Mock user creation
      (prisma.user.create as jest.Mock).mockResolvedValue({
        id: 'user-id',
        name: 'Test User',
        email: 'test@example.com',
        isVerified: false
      });
      
      // Execute
      await authController.register(mockRequest as Request, mockResponse as Response);
      
      // Assert
      expect(prisma.user.findUnique).toHaveBeenCalledWith({
        where: { email: 'test@example.com' }
      });
      expect(bcrypt.genSalt).toHaveBeenCalledWith(10);
      expect(bcrypt.hash).toHaveBeenCalledWith('Password123!', 'salt');
      expect(prisma.user.create).toHaveBeenCalled();
      expect(jwt.generateToken).toHaveBeenCalledWith('user-id');
      expect(jwt.generateRefreshToken).toHaveBeenCalledWith('user-id');
      expect(emailUtils.sendEmail).toHaveBeenCalled();
      expect(mockResponse.status).toHaveBeenCalledWith(201);
      expect(mockResponse.json).toHaveBeenCalledWith(expect.objectContaining({
        success: true,
        data: expect.objectContaining({
          id: 'user-id',
          name: 'Test User',
          email: 'test@example.com',
          token: 'test-token',
          refreshToken: 'test-refresh-token',
          isVerified: false
        })
      }));
    });

    it('should return 400 if user already exists', async () => {
      // Setup
      mockRequest.body = {
        name: 'Test User',
        email: 'test@example.com',
        password: 'Password123!'
      };
      
      // Mock user already exists
      (prisma.user.findUnique as jest.Mock).mockResolvedValue({
        id: 'existing-user-id',
        email: 'test@example.com'
      });
      
      // Execute
      await authController.register(mockRequest as Request, mockResponse as Response);
      
      // Assert
      expect(prisma.user.findUnique).toHaveBeenCalledWith({
        where: { email: 'test@example.com' }
      });
      expect(mockResponse.status).toHaveBeenCalledWith(400);
      expect(mockResponse.json).toHaveBeenCalledWith(expect.objectContaining({
        success: false,
        message: 'User already exists'
      }));
      expect(prisma.user.create).not.toHaveBeenCalled();
    });

    it('should return 400 if required fields are missing', async () => {
      // Setup
      mockRequest.body = {
        name: 'Test User',
        // Missing email and password
      };
      
      // Execute
      await authController.register(mockRequest as Request, mockResponse as Response);
      
      // Assert
      expect(mockResponse.status).toHaveBeenCalledWith(400);
      expect(mockResponse.json).toHaveBeenCalledWith(expect.objectContaining({
        success: false,
        message: 'Please provide all required fields'
      }));
      expect(prisma.user.findUnique).not.toHaveBeenCalled();
      expect(prisma.user.create).not.toHaveBeenCalled();
    });
  });

  describe('login', () => {
    it('should login successfully with valid credentials', async () => {
      // Setup
      mockRequest.body = {
        email: 'test@example.com',
        password: 'Password123!'
      };
      
      // Mock user exists
      (prisma.user.findUnique as jest.Mock).mockResolvedValue({
        id: 'user-id',
        name: 'Test User',
        email: 'test@example.com',
        password: 'hashedPassword',
        isVerified: true,
        role: 'user'
      });
      
      // Mock password match
      (bcrypt.compare as jest.Mock).mockResolvedValue(true);
      
      // Mock user update
      (prisma.user.update as jest.Mock).mockResolvedValue({ id: 'user-id' });
      
      // Execute
      await authController.login(mockRequest as Request, mockResponse as Response);
      
      // Assert
      expect(prisma.user.findUnique).toHaveBeenCalledWith({
        where: { email: 'test@example.com' }
      });
      expect(bcrypt.compare).toHaveBeenCalledWith('Password123!', 'hashedPassword');
      expect(mockResponse.status).toHaveBeenCalledWith(200);
      expect(mockResponse.json).toHaveBeenCalledWith(expect.objectContaining({
        success: true,
        data: expect.objectContaining({
          token: 'test-token',
          refreshToken: 'test-refresh-token'
        })
      }));
    });

    it('should return 401 with invalid credentials (wrong password)', async () => {
      // Setup
      mockRequest.body = {
        email: 'test@example.com',
        password: 'WrongPassword123!'
      };
      
      // Mock user exists
      (prisma.user.findUnique as jest.Mock).mockResolvedValue({
        id: 'user-id',
        email: 'test@example.com',
        password: 'hashedPassword'
      });
      
      // Mock password doesn't match
      (bcrypt.compare as jest.Mock).mockResolvedValue(false);
      
      // Execute
      await authController.login(mockRequest as Request, mockResponse as Response);
      
      // Assert
      expect(prisma.user.findUnique).toHaveBeenCalledWith({
        where: { email: 'test@example.com' }
      });
      expect(bcrypt.compare).toHaveBeenCalledWith('WrongPassword123!', 'hashedPassword');
      expect(mockResponse.status).toHaveBeenCalledWith(401);
      expect(mockResponse.json).toHaveBeenCalledWith(expect.objectContaining({
        success: false,
        message: 'Invalid credentials'
      }));
    });

    it('should return 401 with invalid credentials (user not found)', async () => {
      // Setup
      mockRequest.body = {
        email: 'nonexistent@example.com',
        password: 'Password123!'
      };
      
      // Mock user doesn't exist
      (prisma.user.findUnique as jest.Mock).mockResolvedValue(null);
      
      // Execute
      await authController.login(mockRequest as Request, mockResponse as Response);
      
      // Assert
      expect(prisma.user.findUnique).toHaveBeenCalledWith({
        where: { email: 'nonexistent@example.com' }
      });
      expect(mockResponse.status).toHaveBeenCalledWith(401);
      expect(mockResponse.json).toHaveBeenCalledWith(expect.objectContaining({
        success: false,
        message: 'Invalid credentials'
      }));
      expect(bcrypt.compare).not.toHaveBeenCalled();
    });

    it('should return 400 if required fields are missing', async () => {
      // Setup
      mockRequest.body = {
        // Missing email and password
      };
      
      // Execute
      await authController.login(mockRequest as Request, mockResponse as Response);
      
      // Assert
      expect(mockResponse.status).toHaveBeenCalledWith(400);
      expect(mockResponse.json).toHaveBeenCalledWith(expect.objectContaining({
        success: false,
        message: 'Please provide email and password'
      }));
      expect(prisma.user.findUnique).not.toHaveBeenCalled();
    });
  });

  // Additional tests for verifyEmail, refreshToken, forgotPassword, resetPassword, etc.
  // would follow the same pattern
}); 