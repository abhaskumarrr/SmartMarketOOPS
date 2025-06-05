/**
 * End-to-End Trading Workflow Tests
 * Tests complete user workflows from login to trading
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { BrowserRouter } from 'react-router-dom';
import App from '../../app/page';

// Mock API calls
const mockApiCalls = {
  login: jest.fn(),
  getBots: jest.fn(),
  createBot: jest.fn(),
  startBot: jest.fn(),
  stopBot: jest.fn(),
  getMarketData: jest.fn(),
  getPortfolio: jest.fn(),
};

// Mock the API service
jest.mock('../../lib/services/apiService', () => ({
  apiService: {
    post: jest.fn((endpoint, data) => {
      if (endpoint === '/auth/login') {
        return mockApiCalls.login(data);
      }
      if (endpoint === '/bots') {
        return mockApiCalls.createBot(data);
      }
      return Promise.resolve({ success: true, data: {} });
    }),
    get: jest.fn((endpoint) => {
      if (endpoint === '/bots') {
        return mockApiCalls.getBots();
      }
      if (endpoint === '/market-data') {
        return mockApiCalls.getMarketData();
      }
      if (endpoint === '/portfolio') {
        return mockApiCalls.getPortfolio();
      }
      return Promise.resolve({ success: true, data: {} });
    }),
    put: jest.fn(),
    delete: jest.fn(),
  },
}));

// Mock router
const mockPush = jest.fn();
jest.mock('next/navigation', () => ({
  useRouter: () => ({
    push: mockPush,
    replace: jest.fn(),
    back: jest.fn(),
    forward: jest.fn(),
    refresh: jest.fn(),
    prefetch: jest.fn(),
  }),
  usePathname: () => '/',
  useSearchParams: () => new URLSearchParams(),
}));

const theme = createTheme();

const renderWithProviders = (component: React.ReactElement) => {
  return render(
    <BrowserRouter>
      <ThemeProvider theme={theme}>
        {component}
      </ThemeProvider>
    </BrowserRouter>
  );
};

describe('Trading Workflow E2E Tests', () => {
  const user = userEvent.setup();

  beforeEach(() => {
    jest.clearAllMocks();
    
    // Setup default mock responses
    mockApiCalls.login.mockResolvedValue({
      success: true,
      data: {
        user: { id: '1', email: 'test@example.com', name: 'Test User' },
        token: 'mock-token',
      },
    });

    mockApiCalls.getBots.mockResolvedValue({
      success: true,
      data: [],
    });

    mockApiCalls.getMarketData.mockResolvedValue({
      success: true,
      data: {
        BTCUSD: { price: 50000, change24h: 2.5 },
        ETHUSD: { price: 3000, change24h: -1.2 },
      },
    });

    mockApiCalls.getPortfolio.mockResolvedValue({
      success: true,
      data: {
        totalValue: 10000,
        totalPnL: 500,
        totalPnLPercent: 5.0,
        positions: [],
      },
    });
  });

  describe('Complete Trading Workflow', () => {
    it('should allow user to login, create bot, and start trading', async () => {
      renderWithProviders(<App />);

      // Step 1: User should see login form
      expect(screen.getByText('Login to SmartMarket')).toBeInTheDocument();

      // Step 2: Fill in login credentials
      const emailInput = screen.getByLabelText(/email/i);
      const passwordInput = screen.getByLabelText(/password/i);
      
      await user.type(emailInput, 'test@example.com');
      await user.type(passwordInput, 'password123');

      // Step 3: Submit login
      const loginButton = screen.getByRole('button', { name: /login/i });
      await user.click(loginButton);

      // Step 4: Wait for dashboard to load
      await waitFor(() => {
        expect(screen.getByText('Trading Dashboard')).toBeInTheDocument();
      });

      // Step 5: Navigate to bot management
      const botsNavItem = screen.getByText('Bots');
      await user.click(botsNavItem);

      await waitFor(() => {
        expect(screen.getByText('Trading Bots')).toBeInTheDocument();
      });

      // Step 6: Create a new bot
      const createBotButton = screen.getByText('Create Bot');
      await user.click(createBotButton);

      // Mock successful bot creation
      mockApiCalls.createBot.mockResolvedValue({
        success: true,
        data: {
          id: 'new-bot-id',
          name: 'Test Bot',
          symbol: 'BTCUSD',
          strategy: 'ML_PREDICTION',
          timeframe: '1h',
          isActive: false,
        },
      });

      // Fill in bot configuration
      await waitFor(() => {
        expect(screen.getByText('Bot Configuration')).toBeInTheDocument();
      });

      const botNameInput = screen.getByLabelText(/bot name/i);
      await user.type(botNameInput, 'Test Bot');

      const symbolSelect = screen.getByLabelText(/symbol/i);
      await user.click(symbolSelect);
      await user.click(screen.getByText('BTCUSD'));

      const strategySelect = screen.getByLabelText(/strategy/i);
      await user.click(strategySelect);
      await user.click(screen.getByText('ML Prediction'));

      // Save bot
      const saveBotButton = screen.getByText('Create Bot');
      await user.click(saveBotButton);

      // Step 7: Verify bot was created
      await waitFor(() => {
        expect(mockApiCalls.createBot).toHaveBeenCalledWith({
          name: 'Test Bot',
          symbol: 'BTCUSD',
          strategy: 'ML_PREDICTION',
          timeframe: '1h',
          parameters: expect.any(Object),
        });
      });

      // Step 8: Start the bot
      mockApiCalls.getBots.mockResolvedValue({
        success: true,
        data: [{
          id: 'new-bot-id',
          name: 'Test Bot',
          symbol: 'BTCUSD',
          strategy: 'ML_PREDICTION',
          timeframe: '1h',
          isActive: false,
        }],
      });

      mockApiCalls.startBot.mockResolvedValue({
        success: true,
        message: 'Bot started successfully',
      });

      // Refresh bots list
      await waitFor(() => {
        expect(screen.getByText('Test Bot')).toBeInTheDocument();
      });

      const startButton = screen.getByLabelText('Start Bot');
      await user.click(startButton);

      // Step 9: Verify bot started
      await waitFor(() => {
        expect(mockApiCalls.startBot).toHaveBeenCalledWith('new-bot-id');
      });

      // Step 10: Check portfolio
      const portfolioNavItem = screen.getByText('Portfolio');
      await user.click(portfolioNavItem);

      await waitFor(() => {
        expect(screen.getByText('Portfolio Overview')).toBeInTheDocument();
        expect(screen.getByText('$10,000')).toBeInTheDocument(); // Total value
        expect(screen.getByText('+$500')).toBeInTheDocument(); // P&L
      });
    });

    it('should handle bot configuration with advanced parameters', async () => {
      renderWithProviders(<App />);

      // Login first
      const emailInput = screen.getByLabelText(/email/i);
      const passwordInput = screen.getByLabelText(/password/i);
      
      await user.type(emailInput, 'test@example.com');
      await user.type(passwordInput, 'password123');

      const loginButton = screen.getByRole('button', { name: /login/i });
      await user.click(loginButton);

      await waitFor(() => {
        expect(screen.getByText('Trading Dashboard')).toBeInTheDocument();
      });

      // Navigate to bots
      const botsNavItem = screen.getByText('Bots');
      await user.click(botsNavItem);

      // Create bot with advanced configuration
      const createBotButton = screen.getByText('Create Bot');
      await user.click(createBotButton);

      await waitFor(() => {
        expect(screen.getByText('Bot Configuration')).toBeInTheDocument();
      });

      // Fill basic info
      const botNameInput = screen.getByLabelText(/bot name/i);
      await user.type(botNameInput, 'Advanced Bot');

      // Configure advanced parameters
      const advancedTab = screen.getByText('Advanced');
      await user.click(advancedTab);

      const confidenceThresholdInput = screen.getByLabelText(/confidence threshold/i);
      await user.clear(confidenceThresholdInput);
      await user.type(confidenceThresholdInput, '0.8');

      const riskPerTradeInput = screen.getByLabelText(/risk per trade/i);
      await user.clear(riskPerTradeInput);
      await user.type(riskPerTradeInput, '2');

      const maxPositionsInput = screen.getByLabelText(/max positions/i);
      await user.clear(maxPositionsInput);
      await user.type(maxPositionsInput, '3');

      // Save bot
      const saveBotButton = screen.getByText('Create Bot');
      await user.click(saveBotButton);

      await waitFor(() => {
        expect(mockApiCalls.createBot).toHaveBeenCalledWith(
          expect.objectContaining({
            name: 'Advanced Bot',
            parameters: expect.objectContaining({
              confidenceThreshold: 0.8,
              riskPerTrade: 2,
              maxPositions: 3,
            }),
          })
        );
      });
    });

    it('should handle error scenarios gracefully', async () => {
      renderWithProviders(<App />);

      // Test login failure
      mockApiCalls.login.mockRejectedValue(new Error('Invalid credentials'));

      const emailInput = screen.getByLabelText(/email/i);
      const passwordInput = screen.getByLabelText(/password/i);
      
      await user.type(emailInput, 'wrong@example.com');
      await user.type(passwordInput, 'wrongpassword');

      const loginButton = screen.getByRole('button', { name: /login/i });
      await user.click(loginButton);

      await waitFor(() => {
        expect(screen.getByText(/invalid credentials/i)).toBeInTheDocument();
      });

      // Test successful login after error
      mockApiCalls.login.mockResolvedValue({
        success: true,
        data: {
          user: { id: '1', email: 'test@example.com', name: 'Test User' },
          token: 'mock-token',
        },
      });

      await user.clear(emailInput);
      await user.clear(passwordInput);
      await user.type(emailInput, 'test@example.com');
      await user.type(passwordInput, 'password123');
      await user.click(loginButton);

      await waitFor(() => {
        expect(screen.getByText('Trading Dashboard')).toBeInTheDocument();
      });
    });

    it('should handle network connectivity issues', async () => {
      renderWithProviders(<App />);

      // Simulate network error
      mockApiCalls.login.mockRejectedValue(new Error('Network Error'));

      const emailInput = screen.getByLabelText(/email/i);
      const passwordInput = screen.getByLabelText(/password/i);
      
      await user.type(emailInput, 'test@example.com');
      await user.type(passwordInput, 'password123');

      const loginButton = screen.getByRole('button', { name: /login/i });
      await user.click(loginButton);

      await waitFor(() => {
        expect(screen.getByText(/network error/i)).toBeInTheDocument();
      });

      // Test retry functionality
      const retryButton = screen.getByText('Retry');
      
      // Mock successful retry
      mockApiCalls.login.mockResolvedValue({
        success: true,
        data: {
          user: { id: '1', email: 'test@example.com', name: 'Test User' },
          token: 'mock-token',
        },
      });

      await user.click(retryButton);

      await waitFor(() => {
        expect(screen.getByText('Trading Dashboard')).toBeInTheDocument();
      });
    });

    it('should maintain state across navigation', async () => {
      renderWithProviders(<App />);

      // Login
      const emailInput = screen.getByLabelText(/email/i);
      const passwordInput = screen.getByLabelText(/password/i);
      
      await user.type(emailInput, 'test@example.com');
      await user.type(passwordInput, 'password123');

      const loginButton = screen.getByRole('button', { name: /login/i });
      await user.click(loginButton);

      await waitFor(() => {
        expect(screen.getByText('Trading Dashboard')).toBeInTheDocument();
      });

      // Navigate to different sections and verify state persistence
      const sections = ['Bots', 'Portfolio', 'Paper Trading', 'Dashboard'];

      for (const section of sections) {
        const navItem = screen.getByText(section);
        await user.click(navItem);

        await waitFor(() => {
          // User should remain logged in
          expect(screen.queryByText('Login to SmartMarket')).not.toBeInTheDocument();
        });
      }
    });
  });

  describe('Performance Under Load', () => {
    it('should handle rapid user interactions', async () => {
      renderWithProviders(<App />);

      // Login
      const emailInput = screen.getByLabelText(/email/i);
      const passwordInput = screen.getByLabelText(/password/i);
      
      await user.type(emailInput, 'test@example.com');
      await user.type(passwordInput, 'password123');

      const loginButton = screen.getByRole('button', { name: /login/i });
      await user.click(loginButton);

      await waitFor(() => {
        expect(screen.getByText('Trading Dashboard')).toBeInTheDocument();
      });

      // Rapidly navigate between sections
      const navItems = ['Bots', 'Portfolio', 'Dashboard', 'Paper Trading'];
      
      for (let i = 0; i < 10; i++) {
        const randomSection = navItems[Math.floor(Math.random() * navItems.length)];
        const navItem = screen.getByText(randomSection);
        await user.click(navItem);
        
        // Small delay to simulate real user behavior
        await new Promise(resolve => setTimeout(resolve, 100));
      }

      // Application should remain responsive
      expect(screen.getByText('Trading Dashboard')).toBeInTheDocument();
    });
  });
});
