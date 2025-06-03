import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { useRouter } from 'next/router';

interface User {
  id: string;
  name: string;
  email: string;
  role: string;
  isVerified: boolean;
  lastLoginAt?: string;
  sessionId?: string;
}

interface AuthState {
  user: User | null;
  token: string | null;
  refreshToken: string | null;
  loading: boolean;
  isAuthenticated: boolean;
  tokenExpiry: number | null;
  sessionId: string | null;
}

interface AuthContextType extends AuthState {
  login: (email: string, password: string, rememberMe?: boolean) => Promise<boolean>;
  register: (name: string, email: string, password: string) => Promise<boolean>;
  logout: () => Promise<void>;
  updateUser: (userData: Partial<User>) => void;
  forgotPassword: (email: string) => Promise<boolean>;
  resetPassword: (token: string, password: string) => Promise<boolean>;
  verifyEmail: (token: string) => Promise<boolean>;
  refreshSession: () => Promise<boolean>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [state, setState] = useState<AuthState>({
    user: null,
    token: null,
    refreshToken: null,
    loading: true,
    isAuthenticated: false,
    tokenExpiry: null,
    sessionId: null,
  });

  const router = useRouter();

  // Initialize auth state from localStorage on mount
  useEffect(() => {
    const loadStoredAuth = () => {
      try {
        const storedUser = localStorage.getItem('user');
        const storedToken = localStorage.getItem('token');
        const storedRefreshToken = localStorage.getItem('refreshToken');
        const storedTokenExpiry = localStorage.getItem('tokenExpiry');
        const storedSessionId = localStorage.getItem('sessionId');

        if (storedUser && storedToken && storedRefreshToken) {
          const tokenExpiry = storedTokenExpiry ? parseInt(storedTokenExpiry) : null;

          // Check if token is expired
          if (tokenExpiry && Date.now() > tokenExpiry) {
            // Token is expired, try to refresh
            refreshSessionInternal(storedRefreshToken);
          } else {
            setState({
              user: JSON.parse(storedUser),
              token: storedToken,
              refreshToken: storedRefreshToken,
              tokenExpiry,
              sessionId: storedSessionId,
              loading: false,
              isAuthenticated: true,
            });
          }
        } else {
          setState(prev => ({ ...prev, loading: false }));
        }
      } catch (error) {
        console.error('Failed to load auth from storage:', error);
        setState(prev => ({ ...prev, loading: false }));
      }
    };

    loadStoredAuth();
  }, []);

  // Auto-refresh token before expiry
  useEffect(() => {
    if (!state.tokenExpiry || !state.refreshToken) return;

    const timeUntilExpiry = state.tokenExpiry - Date.now();
    const refreshTime = Math.max(timeUntilExpiry - 60000, 30000); // Refresh 1 minute before expiry, minimum 30 seconds

    const refreshTimer = setTimeout(() => {
      refreshSession();
    }, refreshTime);

    return () => clearTimeout(refreshTimer);
  }, [state.tokenExpiry, state.refreshToken]);

  // Internal refresh function for initialization
  const refreshSessionInternal = async (refreshToken: string): Promise<void> => {
    try {
      const response = await fetch(`${API_URL}/api/auth/refresh-token`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ refreshToken }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || 'Failed to refresh token');
      }

      // Calculate token expiry (15 minutes from now)
      const tokenExpiry = Date.now() + (data.data.expiresIn * 1000);

      // Update storage
      localStorage.setItem('token', data.data.accessToken);
      localStorage.setItem('refreshToken', data.data.refreshToken);
      localStorage.setItem('tokenExpiry', tokenExpiry.toString());
      localStorage.setItem('sessionId', data.data.sessionId);

      setState(prev => ({
        ...prev,
        token: data.data.accessToken,
        refreshToken: data.data.refreshToken,
        tokenExpiry,
        sessionId: data.data.sessionId,
        loading: false,
        isAuthenticated: true,
      }));
    } catch (error) {
      console.error('Internal token refresh error:', error);
      // Clear invalid auth data
      localStorage.removeItem('user');
      localStorage.removeItem('token');
      localStorage.removeItem('refreshToken');
      localStorage.removeItem('tokenExpiry');
      localStorage.removeItem('sessionId');

      setState(prev => ({
        ...prev,
        user: null,
        token: null,
        refreshToken: null,
        tokenExpiry: null,
        sessionId: null,
        loading: false,
        isAuthenticated: false,
      }));
    }
  };

  // Fetch CSRF token for secure requests
  const getCsrfToken = async (): Promise<string> => {
    try {
      const response = await fetch(`${API_URL}/api/auth/csrf-token`, {
        method: 'GET',
        credentials: 'include',
      });

      if (!response.ok) {
        throw new Error('Failed to get CSRF token');
      }

      const data = await response.json();
      return data.csrfToken;
    } catch (error) {
      console.error('CSRF token error:', error);
      return '';
    }
  };

  // Login function
  const login = async (email: string, password: string, rememberMe: boolean = false): Promise<boolean> => {
    try {
      const csrfToken = await getCsrfToken();

      const response = await fetch(`${API_URL}/api/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRF-Token': csrfToken,
        },
        credentials: 'include',
        body: JSON.stringify({ email, password, rememberMe }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || 'Login failed');
      }

      // Calculate token expiry (15 minutes from now)
      const tokenExpiry = Date.now() + (data.data.expiresIn * 1000);

      // Store auth data with new format
      localStorage.setItem('user', JSON.stringify(data.data));
      localStorage.setItem('token', data.data.accessToken);
      localStorage.setItem('refreshToken', data.data.refreshToken);
      localStorage.setItem('tokenExpiry', tokenExpiry.toString());
      localStorage.setItem('sessionId', data.data.sessionId);

      setState({
        user: data.data,
        token: data.data.accessToken,
        refreshToken: data.data.refreshToken,
        tokenExpiry,
        sessionId: data.data.sessionId,
        loading: false,
        isAuthenticated: true,
      });

      return true;
    } catch (error) {
      console.error('Login error:', error);
      return false;
    }
  };

  // Register function
  const register = async (name: string, email: string, password: string): Promise<boolean> => {
    try {
      const csrfToken = await getCsrfToken();

      const response = await fetch(`${API_URL}/api/auth/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRF-Token': csrfToken,
        },
        credentials: 'include',
        body: JSON.stringify({ name, email, password }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || 'Registration failed');
      }

      // Calculate token expiry (15 minutes from now)
      const tokenExpiry = Date.now() + (data.data.expiresIn * 1000);

      // Store auth data with new format
      localStorage.setItem('user', JSON.stringify(data.data));
      localStorage.setItem('token', data.data.accessToken);
      localStorage.setItem('refreshToken', data.data.refreshToken);
      localStorage.setItem('tokenExpiry', tokenExpiry.toString());
      localStorage.setItem('sessionId', data.data.sessionId);

      setState({
        user: data.data,
        token: data.data.accessToken,
        refreshToken: data.data.refreshToken,
        tokenExpiry,
        sessionId: data.data.sessionId,
        loading: false,
        isAuthenticated: true,
      });

      return true;
    } catch (error) {
      console.error('Registration error:', error);
      return false;
    }
  };

  // Logout function
  const logout = async (): Promise<void> => {
    try {
      if (state.token) {
        const csrfToken = await getCsrfToken();

        await fetch(`${API_URL}/api/auth/logout`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${state.token}`,
            'X-CSRF-Token': csrfToken,
          },
          credentials: 'include',
        });
      }
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      // Clear auth data regardless of API success
      localStorage.removeItem('user');
      localStorage.removeItem('token');
      localStorage.removeItem('refreshToken');
      localStorage.removeItem('tokenExpiry');
      localStorage.removeItem('sessionId');

      setState({
        user: null,
        token: null,
        refreshToken: null,
        tokenExpiry: null,
        sessionId: null,
        loading: false,
        isAuthenticated: false,
      });

      router.push('/login');
    }
  };

  // Update user data
  const updateUser = (userData: Partial<User>): void => {
    if (!state.user) return;

    const updatedUser = { ...state.user, ...userData };
    localStorage.setItem('user', JSON.stringify(updatedUser));
    setState(prev => ({ ...prev, user: updatedUser }));
  };

  // Forgot password
  const forgotPassword = async (email: string): Promise<boolean> => {
    try {
      const csrfToken = await getCsrfToken();

      const response = await fetch(`${API_URL}/api/auth/forgot-password`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRF-Token': csrfToken,
        },
        credentials: 'include',
        body: JSON.stringify({ email }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || 'Failed to request password reset');
      }

      return true;
    } catch (error) {
      console.error('Forgot password error:', error);
      return false;
    }
  };

  // Reset password
  const resetPassword = async (token: string, password: string): Promise<boolean> => {
    try {
      const csrfToken = await getCsrfToken();

      const response = await fetch(`${API_URL}/api/auth/reset-password`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRF-Token': csrfToken,
        },
        credentials: 'include',
        body: JSON.stringify({ token, password }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || 'Failed to reset password');
      }

      return true;
    } catch (error) {
      console.error('Reset password error:', error);
      return false;
    }
  };

  // Verify email
  const verifyEmail = async (token: string): Promise<boolean> => {
    try {
      const response = await fetch(`${API_URL}/api/auth/verify-email/${token}`, {
        method: 'GET',
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || 'Failed to verify email');
      }

      // If user is logged in, update verification status
      if (state.user) {
        updateUser({ isVerified: true });
      }

      return true;
    } catch (error) {
      console.error('Email verification error:', error);
      return false;
    }
  };

  // Refresh access token
  const refreshSession = async (): Promise<boolean> => {
    if (!state.refreshToken) return false;

    try {
      const response = await fetch(`${API_URL}/api/auth/refresh-token`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ refreshToken: state.refreshToken }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || 'Failed to refresh token');
      }

      // Calculate token expiry (15 minutes from now)
      const tokenExpiry = Date.now() + (data.data.expiresIn * 1000);

      // Update storage with new token format
      localStorage.setItem('token', data.data.accessToken);
      localStorage.setItem('refreshToken', data.data.refreshToken);
      localStorage.setItem('tokenExpiry', tokenExpiry.toString());
      localStorage.setItem('sessionId', data.data.sessionId);

      setState(prev => ({
        ...prev,
        token: data.data.accessToken,
        refreshToken: data.data.refreshToken,
        tokenExpiry,
        sessionId: data.data.sessionId,
      }));

      return true;
    } catch (error) {
      console.error('Token refresh error:', error);

      // If refresh fails, logout the user
      logout();
      return false;
    }
  };

  const value = {
    ...state,
    login,
    register,
    logout,
    updateUser,
    forgotPassword,
    resetPassword,
    verifyEmail,
    refreshSession,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export default AuthContext;