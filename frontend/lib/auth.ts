import { NextRequest } from 'next/server';

export interface AuthUser {
  id: string;
  email: string;
  role: string;
}

/**
 * Simple authentication check for API routes
 * In a production environment, this would validate JWT tokens
 */
export async function authenticateRequest(request: NextRequest): Promise<AuthUser | null> {
  try {
    // Get authorization header
    const authHeader = request.headers.get('authorization');
    
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return null;
    }

    const token = authHeader.substring(7); // Remove 'Bearer ' prefix
    
    // For development/demo purposes, we'll use a simple token validation
    // In production, this would validate a proper JWT token
    if (token === 'demo-token' || token === 'development-token') {
      return {
        id: 'demo-user',
        email: 'demo@smartmarket.com',
        role: 'trader'
      };
    }

    // In production, you would:
    // 1. Verify JWT signature
    // 2. Check token expiration
    // 3. Validate user exists in database
    // 4. Check user permissions
    
    return null;
  } catch (error) {
    console.error('Authentication error:', error);
    return null;
  }
}

/**
 * Check if user has required role
 */
export function hasRole(user: AuthUser, requiredRole: string): boolean {
  // Simple role check - in production you might have more complex role hierarchies
  const roleHierarchy = {
    'admin': ['admin', 'trader', 'viewer'],
    'trader': ['trader', 'viewer'],
    'viewer': ['viewer']
  };

  return roleHierarchy[user.role as keyof typeof roleHierarchy]?.includes(requiredRole) || false;
}

/**
 * Create authentication response for unauthorized requests
 */
export function createUnauthorizedResponse(message: string = 'Authentication required') {
  return Response.json(
    {
      success: false,
      error: 'Unauthorized',
      message,
      code: 'AUTH_REQUIRED'
    },
    { status: 401 }
  );
}

/**
 * Create forbidden response for insufficient permissions
 */
export function createForbiddenResponse(message: string = 'Insufficient permissions') {
  return Response.json(
    {
      success: false,
      error: 'Forbidden',
      message,
      code: 'INSUFFICIENT_PERMISSIONS'
    },
    { status: 403 }
  );
}
