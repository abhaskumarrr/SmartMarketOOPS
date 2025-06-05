/**
 * Test Helpers
 * Utility functions for testing
 */

import { PrismaClient } from '@prisma/client';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import { getTestPrisma } from './testApp';

interface TestUser {
  id: string;
  email: string;
  name: string;
  password: string;
}

interface TestBot {
  id: string;
  name: string;
  symbol: string;
  strategy: string;
  timeframe: string;
  userId: string;
}

let testPrisma: PrismaClient;
const createdUsers: string[] = [];
const createdBots: string[] = [];

export const initializeTestHelpers = () => {
  testPrisma = getTestPrisma();
};

export const createTestUser = async (userData?: Partial<TestUser>): Promise<{
  user: TestUser;
  token: string;
}> => {
  if (!testPrisma) {
    testPrisma = getTestPrisma();
  }

  const defaultUserData = {
    email: 'test@example.com',
    name: 'Test User',
    password: 'password123',
  };

  const finalUserData = { ...defaultUserData, ...userData };
  const hashedPassword = await bcrypt.hash(finalUserData.password, 10);

  const user = await testPrisma.user.create({
    data: {
      email: finalUserData.email,
      name: finalUserData.name,
      password: hashedPassword,
      role: 'user',
    },
  });

  createdUsers.push(user.id);

  // Generate JWT token
  const token = jwt.sign(
    { userId: user.id, email: user.email },
    process.env.JWT_SECRET || 'test-secret',
    { expiresIn: '1h' }
  );

  return {
    user: {
      id: user.id,
      email: user.email,
      name: user.name,
      password: finalUserData.password,
    },
    token,
  };
};

export const createTestBot = async (
  userId: string,
  botData?: Partial<TestBot>
): Promise<TestBot> => {
  if (!testPrisma) {
    testPrisma = getTestPrisma();
  }

  const defaultBotData = {
    name: 'Test Bot',
    symbol: 'BTCUSD',
    strategy: 'ML_PREDICTION',
    timeframe: '1h',
    parameters: {
      confidence_threshold: 0.7,
    },
  };

  const finalBotData = { ...defaultBotData, ...botData };

  const bot = await testPrisma.bot.create({
    data: {
      name: finalBotData.name,
      symbol: finalBotData.symbol,
      strategy: finalBotData.strategy,
      timeframe: finalBotData.timeframe,
      parameters: finalBotData.parameters,
      userId,
      isActive: false,
    },
  });

  createdBots.push(bot.id);

  return {
    id: bot.id,
    name: bot.name,
    symbol: bot.symbol,
    strategy: bot.strategy,
    timeframe: bot.timeframe,
    userId: bot.userId,
  };
};

export const createTestMarketData = async (symbol: string, count: number = 100) => {
  if (!testPrisma) {
    testPrisma = getTestPrisma();
  }

  const marketData = [];
  const basePrice = 50000; // Starting price for BTCUSD
  let currentPrice = basePrice;

  for (let i = 0; i < count; i++) {
    // Simulate price movement
    const change = (Math.random() - 0.5) * 0.02; // ±1% change
    currentPrice = currentPrice * (1 + change);

    const timestamp = new Date(Date.now() - (count - i) * 60 * 60 * 1000); // Hourly data

    marketData.push({
      symbol,
      timestamp,
      open: currentPrice * 0.999,
      high: currentPrice * 1.001,
      low: currentPrice * 0.998,
      close: currentPrice,
      volume: Math.random() * 1000000,
    });
  }

  // Note: This assumes you have a MarketData model in your Prisma schema
  // If not, you might need to mock this data differently
  try {
    await testPrisma.marketData.createMany({
      data: marketData,
      skipDuplicates: true,
    });
  } catch (error) {
    // If MarketData model doesn't exist, just return the mock data
    console.warn('MarketData model not found, returning mock data');
  }

  return marketData;
};

export const cleanupTestData = async (): Promise<void> => {
  if (!testPrisma) {
    return;
  }

  try {
    // Delete in reverse order of dependencies
    if (createdBots.length > 0) {
      await testPrisma.bot.deleteMany({
        where: {
          id: {
            in: createdBots,
          },
        },
      });
    }

    if (createdUsers.length > 0) {
      await testPrisma.user.deleteMany({
        where: {
          id: {
            in: createdUsers,
          },
        },
      });
    }

    // Clear tracking arrays
    createdUsers.length = 0;
    createdBots.length = 0;
  } catch (error) {
    console.error('Error cleaning up test data:', error);
  }
};

export const mockApiResponse = <T>(data: T, success: boolean = true) => ({
  success,
  data: success ? data : undefined,
  message: success ? 'Success' : 'Error',
  error: success ? undefined : 'Test error',
});

export const waitFor = (ms: number): Promise<void> => {
  return new Promise(resolve => setTimeout(resolve, ms));
};

export const generateRandomString = (length: number = 10): string => {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let result = '';
  for (let i = 0; i < length; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return result;
};

export const generateRandomEmail = (): string => {
  return `test-${generateRandomString(8)}@example.com`;
};

export const mockPerformanceMetrics = () => ({
  pageLoadTime: 1000 + Math.random() * 500,
  firstContentfulPaint: 800 + Math.random() * 400,
  largestContentfulPaint: 1200 + Math.random() * 600,
  cumulativeLayoutShift: Math.random() * 0.1,
  firstInputDelay: Math.random() * 100,
  timeToInteractive: 1500 + Math.random() * 1000,
});

export const mockBotStatus = () => ({
  isRunning: Math.random() > 0.5,
  health: ['excellent', 'good', 'degraded', 'poor'][Math.floor(Math.random() * 4)],
  metrics: {
    profitLoss: (Math.random() - 0.5) * 1000,
    profitLossPercent: (Math.random() - 0.5) * 10,
    successRate: 60 + Math.random() * 30,
    tradesExecuted: Math.floor(Math.random() * 100),
    latency: Math.random() * 100,
  },
  activePositions: Math.floor(Math.random() * 5),
  logs: [],
  errors: [],
});

export const mockMarketData = (symbol: string) => ({
  symbol,
  price: 50000 + (Math.random() - 0.5) * 10000,
  change24h: (Math.random() - 0.5) * 10,
  volume24h: Math.random() * 1000000,
  timestamp: Date.now(),
});

// Database transaction helper for tests
export const withTransaction = async <T>(
  callback: (prisma: PrismaClient) => Promise<T>
): Promise<T> => {
  if (!testPrisma) {
    testPrisma = getTestPrisma();
  }

  return await testPrisma.$transaction(async (tx) => {
    return await callback(tx);
  });
};

// Reset database to clean state
export const resetTestDatabase = async (): Promise<void> => {
  if (!testPrisma) {
    return;
  }

  // Delete all test data in correct order
  await testPrisma.bot.deleteMany();
  await testPrisma.user.deleteMany();
  
  // Reset sequences if using PostgreSQL
  try {
    await testPrisma.$executeRaw`ALTER SEQUENCE "User_id_seq" RESTART WITH 1`;
    await testPrisma.$executeRaw`ALTER SEQUENCE "Bot_id_seq" RESTART WITH 1`;
  } catch (error) {
    // Ignore if not using PostgreSQL or sequences don't exist
  }
};
