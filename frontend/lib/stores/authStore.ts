/**
 * Authentication Store for SmartMarketOOPS
 * Simplified auth store for demo purposes
 */

import { create } from 'zustand';

interface User {
  id: string;
  email: string;
  name: string;
  role: string;
}

interface AuthState {
  // Auth state
  isAuthenticated: boolean;
  user: User | null;
  token: string | null;
  isLoading: boolean;
  error: string | null;

  // Auth actions
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  setUser: (user: User) => void;
  setToken: (token: string) => void;
  clearError: () => void;
  
  // Demo mode
  enableDemoMode: () => void;
}

export const useAuthStore = create<AuthState>((set, get) => ({
  // Initial state
  isAuthenticated: false,
  user: null,
  token: null,
  isLoading: false,
  error: null,

  // Login action (simplified for demo)
  login: async (email: string, password: string) => {
    set({ isLoading: true, error: null });
    
    try {
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // For demo purposes, accept any email/password
      const demoUser: User = {
        id: 'demo-user-1',
        email: email || 'demo@smartmarketoops.com',
        name: 'Demo User',
        role: 'trader'
      };
      
      const demoToken = 'demo-jwt-token-' + Date.now();
      
      set({
        isAuthenticated: true,
        user: demoUser,
        token: demoToken,
        isLoading: false,
        error: null
      });
      
      // Store in localStorage for persistence
      localStorage.setItem('auth_token', demoToken);
      localStorage.setItem('auth_user', JSON.stringify(demoUser));
      
    } catch (error) {
      set({
        isLoading: false,
        error: error instanceof Error ? error.message : 'Login failed'
      });
    }
  },

  // Logout action
  logout: () => {
    set({
      isAuthenticated: false,
      user: null,
      token: null,
      error: null
    });
    
    // Clear localStorage
    localStorage.removeItem('auth_token');
    localStorage.removeItem('auth_user');
  },

  // Set user
  setUser: (user: User) => {
    set({ user, isAuthenticated: true });
  },

  // Set token
  setToken: (token: string) => {
    set({ token, isAuthenticated: true });
  },

  // Clear error
  clearError: () => {
    set({ error: null });
  },

  // Enable demo mode (auto-login)
  enableDemoMode: () => {
    const demoUser: User = {
      id: 'demo-user-1',
      email: 'demo@smartmarketoops.com',
      name: 'Demo Trader',
      role: 'trader'
    };
    
    const demoToken = 'demo-jwt-token-' + Date.now();
    
    set({
      isAuthenticated: true,
      user: demoUser,
      token: demoToken,
      isLoading: false,
      error: null
    });
    
    // Store in localStorage
    localStorage.setItem('auth_token', demoToken);
    localStorage.setItem('auth_user', JSON.stringify(demoUser));
  }
}));

// Initialize auth state from localStorage on app start
if (typeof window !== 'undefined') {
  const storedToken = localStorage.getItem('auth_token');
  const storedUser = localStorage.getItem('auth_user');
  
  if (storedToken && storedUser) {
    try {
      const user = JSON.parse(storedUser);
      useAuthStore.getState().setUser(user);
      useAuthStore.getState().setToken(storedToken);
    } catch (error) {
      console.error('Error parsing stored user data:', error);
      localStorage.removeItem('auth_token');
      localStorage.removeItem('auth_user');
    }
  } else {
    // Auto-enable demo mode if no stored auth
    useAuthStore.getState().enableDemoMode();
  }
}
