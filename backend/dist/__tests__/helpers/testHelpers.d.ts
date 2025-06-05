/**
 * Test Helpers
 * Utility functions for testing
 */
import { PrismaClient } from '@prisma/client';
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
export declare const initializeTestHelpers: () => void;
export declare const createTestUser: (userData?: Partial<TestUser>) => Promise<{
    user: TestUser;
    token: string;
}>;
export declare const createTestBot: (userId: string, botData?: Partial<TestBot>) => Promise<TestBot>;
export declare const createTestMarketData: (symbol: string, count?: number) => Promise<any[]>;
export declare const cleanupTestData: () => Promise<void>;
export declare const mockApiResponse: <T>(data: T, success?: boolean) => {
    success: boolean;
    data: T;
    message: string;
    error: string;
};
export declare const waitFor: (ms: number) => Promise<void>;
export declare const generateRandomString: (length?: number) => string;
export declare const generateRandomEmail: () => string;
export declare const mockPerformanceMetrics: () => {
    pageLoadTime: number;
    firstContentfulPaint: number;
    largestContentfulPaint: number;
    cumulativeLayoutShift: number;
    firstInputDelay: number;
    timeToInteractive: number;
};
export declare const mockBotStatus: () => {
    isRunning: boolean;
    health: string;
    metrics: {
        profitLoss: number;
        profitLossPercent: number;
        successRate: number;
        tradesExecuted: number;
        latency: number;
    };
    activePositions: number;
    logs: any[];
    errors: any[];
};
export declare const mockMarketData: (symbol: string) => {
    symbol: string;
    price: number;
    change24h: number;
    volume24h: number;
    timestamp: number;
};
export declare const withTransaction: <T>(callback: (prisma: PrismaClient) => Promise<T>) => Promise<T>;
export declare const resetTestDatabase: () => Promise<void>;
export {};
