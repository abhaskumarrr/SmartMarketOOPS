/**
 * Error Handler Tests
 * Tests the centralized error handling module
 */

import { Request, Response } from 'express';
import { 
  ApiError, 
  asyncHandler, 
  errorMiddleware, 
  notFoundHandler,
  throwNotFound,
  throwUnauthorized,
  throwForbidden,
  throwBadRequest,
  throwConflict,
  throwServerError
} from '../../../src/utils/errorHandler';

// Mock Express request and response
const mockRequest = {} as Request;
const mockResponse = {
  status: jest.fn().mockReturnThis(),
  json: jest.fn()
} as unknown as Response;
const mockNext = jest.fn();

describe('Error Handler', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('ApiError', () => {
    it('should create an ApiError with default values', () => {
      const error = new ApiError('Test error');
      
      expect(error).toBeInstanceOf(Error);
      expect(error.message).toBe('Test error');
      expect(error.statusCode).toBe(500);
      expect(error.isOperational).toBe(true);
      expect(error.errorCode).toBeUndefined();
      expect(error.details).toBeUndefined();
    });

    it('should create an ApiError with custom values', () => {
      const error = new ApiError(
        'Not found', 
        404, 
        true, 
        'RESOURCE_NOT_FOUND', 
        { resource: 'user', id: '123' }
      );
      
      expect(error.message).toBe('Not found');
      expect(error.statusCode).toBe(404);
      expect(error.isOperational).toBe(true);
      expect(error.errorCode).toBe('RESOURCE_NOT_FOUND');
      expect(error.details).toEqual({ resource: 'user', id: '123' });
    });
  });

  describe('asyncHandler', () => {
    it('should call next with error when async function throws', async () => {
      const error = new Error('Async error');
      const asyncFn = jest.fn().mockRejectedValue(error);
      const handler = asyncHandler(asyncFn);
      
      await handler(mockRequest, mockResponse, mockNext);
      
      expect(asyncFn).toHaveBeenCalledWith(mockRequest, mockResponse, mockNext);
      expect(mockNext).toHaveBeenCalledWith(error);
    });

    it('should not call next when async function resolves', async () => {
      const asyncFn = jest.fn().mockResolvedValue('success');
      const handler = asyncHandler(asyncFn);
      
      await handler(mockRequest, mockResponse, mockNext);
      
      expect(asyncFn).toHaveBeenCalledWith(mockRequest, mockResponse, mockNext);
      expect(mockNext).not.toHaveBeenCalled();
    });
  });

  describe('errorMiddleware', () => {
    it('should handle ApiError correctly', () => {
      const error = new ApiError('Test error', 400, true, 'TEST_ERROR', { field: 'test' });
      
      errorMiddleware(error, mockRequest, mockResponse, mockNext);
      
      expect(mockResponse.status).toHaveBeenCalledWith(400);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: false,
        error: {
          message: 'Test error',
          code: 'TEST_ERROR',
          details: { field: 'test' }
        }
      });
    });

    it('should handle standard Error correctly', () => {
      const error = new Error('Standard error');
      
      errorMiddleware(error, mockRequest, mockResponse, mockNext);
      
      expect(mockResponse.status).toHaveBeenCalledWith(500);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: false,
        error: {
          message: 'Standard error',
          code: 'INTERNAL_ERROR'
        }
      });
    });

    it('should handle Prisma unique constraint error correctly', () => {
      const error = {
        name: 'PrismaClientKnownRequestError',
        code: 'P2002',
        meta: { target: ['email'] }
      };
      
      errorMiddleware(error, mockRequest, mockResponse, mockNext);
      
      expect(mockResponse.status).toHaveBeenCalledWith(409);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: false,
        error: {
          message: 'Resource already exists',
          code: 'RESOURCE_EXISTS',
          details: { fields: ['email'] }
        }
      });
    });

    it('should handle Prisma record not found error correctly', () => {
      const error = {
        name: 'PrismaClientKnownRequestError',
        code: 'P2025'
      };
      
      errorMiddleware(error, mockRequest, mockResponse, mockNext);
      
      expect(mockResponse.status).toHaveBeenCalledWith(404);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: false,
        error: {
          message: 'Resource not found',
          code: 'RESOURCE_NOT_FOUND'
        }
      });
    });
  });

  describe('notFoundHandler', () => {
    it('should create a 404 error and pass it to next', () => {
      mockRequest.originalUrl = '/not-found';
      
      notFoundHandler(mockRequest, mockResponse, mockNext);
      
      expect(mockNext).toHaveBeenCalledWith(expect.any(ApiError));
      const error = mockNext.mock.calls[0][0];
      expect(error.message).toBe('Not Found - /not-found');
      expect(error.statusCode).toBe(404);
      expect(error.errorCode).toBe('NOT_FOUND');
    });
  });

  describe('Error Utility Functions', () => {
    it('should throw NotFound error', () => {
      expect(() => throwNotFound()).toThrow(ApiError);
      try {
        throwNotFound('User not found', 'USER_NOT_FOUND');
      } catch (error) {
        expect(error).toBeInstanceOf(ApiError);
        expect(error.message).toBe('User not found');
        expect(error.statusCode).toBe(404);
        expect(error.errorCode).toBe('USER_NOT_FOUND');
      }
    });

    it('should throw Unauthorized error', () => {
      expect(() => throwUnauthorized()).toThrow(ApiError);
      try {
        throwUnauthorized('Invalid token', 'INVALID_TOKEN');
      } catch (error) {
        expect(error).toBeInstanceOf(ApiError);
        expect(error.message).toBe('Invalid token');
        expect(error.statusCode).toBe(401);
        expect(error.errorCode).toBe('INVALID_TOKEN');
      }
    });

    it('should throw Forbidden error', () => {
      expect(() => throwForbidden()).toThrow(ApiError);
      try {
        throwForbidden('Insufficient permissions', 'INSUFFICIENT_PERMISSIONS');
      } catch (error) {
        expect(error).toBeInstanceOf(ApiError);
        expect(error.message).toBe('Insufficient permissions');
        expect(error.statusCode).toBe(403);
        expect(error.errorCode).toBe('INSUFFICIENT_PERMISSIONS');
      }
    });

    it('should throw BadRequest error', () => {
      expect(() => throwBadRequest()).toThrow(ApiError);
      try {
        throwBadRequest('Invalid input', 'INVALID_INPUT', { field: 'email' });
      } catch (error) {
        expect(error).toBeInstanceOf(ApiError);
        expect(error.message).toBe('Invalid input');
        expect(error.statusCode).toBe(400);
        expect(error.errorCode).toBe('INVALID_INPUT');
        expect(error.details).toEqual({ field: 'email' });
      }
    });

    it('should throw Conflict error', () => {
      expect(() => throwConflict()).toThrow(ApiError);
      try {
        throwConflict('Email already exists', 'EMAIL_EXISTS');
      } catch (error) {
        expect(error).toBeInstanceOf(ApiError);
        expect(error.message).toBe('Email already exists');
        expect(error.statusCode).toBe(409);
        expect(error.errorCode).toBe('EMAIL_EXISTS');
      }
    });

    it('should throw ServerError', () => {
      expect(() => throwServerError()).toThrow(ApiError);
      try {
        throwServerError('Database connection failed', 'DB_CONNECTION_ERROR');
      } catch (error) {
        expect(error).toBeInstanceOf(ApiError);
        expect(error.message).toBe('Database connection failed');
        expect(error.statusCode).toBe(500);
        expect(error.isOperational).toBe(false);
        expect(error.errorCode).toBe('DB_CONNECTION_ERROR');
      }
    });
  });
});