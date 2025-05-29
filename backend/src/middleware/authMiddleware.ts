/**
 * Auth Middleware
 * Provides authentication and authorization middleware for routes
 */

import { Request, Response, NextFunction } from 'express';
import prisma from '../utils/prismaClient';
import jwt from 'jsonwebtoken';

// Interface for authenticated request
export interface AuthenticatedRequest extends Request {
  user?: {
    id: string;
    email?: string;
    role?: string;
    [key: string]: any;
  };
}

/**
 * Authenticate JWT token and attach user to request
 * @param req - Express request
 * @param res - Express response
 * @param next - Express next function
 */
export const authenticateJWT = async (req: AuthenticatedRequest, res: Response, next: NextFunction): Promise<void> => {
  try {
    const authHeader = req.headers.authorization;
    
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      res.status(401).json({ 
        error: 'Unauthorized',
        message: 'Authentication token is missing or invalid'
      });
      return;
    }
    
    const token = authHeader.split(' ')[1];
    
    // Verify token
    const decoded = jwt.verify(token, process.env.JWT_SECRET as string) as any;
    
    // Get user from database
    const user = await prisma.user.findUnique({
      where: { id: decoded.id || decoded.sub }
    });
    
    if (!user) {
      res.status(401).json({ 
        error: 'Unauthorized',
        message: 'User not found'
      });
      return;
    }
    
    // Attach user to request
    req.user = {
      id: user.id,
      email: user.email,
      role: user.role
    };
    
    next();
  } catch (error) {
    console.error('JWT Authentication error:', error);
    
    res.status(401).json({ 
      error: 'Unauthorized',
      message: 'Invalid or expired token'
    });
  }
};

/**
 * Authorize roles middleware
 * @param roles - Allowed roles
 * @returns Middleware function
 */
export const authorizeRoles = (roles: string[]) => {
  return (req: AuthenticatedRequest, res: Response, next: NextFunction): void => {
    if (!req.user) {
      res.status(401).json({ 
        error: 'Unauthorized',
        message: 'Authentication required'
      });
      return;
    }
    
    if (!roles.includes(req.user.role as string)) {
      res.status(403).json({ 
        error: 'Forbidden',
        message: 'You do not have permission to access this resource'
      });
      return;
    }
    
    next();
  };
}; 