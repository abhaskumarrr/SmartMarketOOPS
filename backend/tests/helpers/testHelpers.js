"use strict";
/**
 * Test Helpers
 * Utility functions for testing
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.generateLoadTestData = exports.waitForCondition = exports.generateTestPerformance = exports.generateTestMarketData = exports.resetTestDatabase = exports.withTransaction = exports.mockMarketData = exports.mockBotStatus = exports.mockPerformanceMetrics = exports.generateRandomEmail = exports.generateRandomString = exports.waitFor = exports.mockApiResponse = exports.cleanupTestData = exports.createTestMarketData = exports.createTestBot = exports.createTestUser = exports.initializeTestHelpers = void 0;
const bcryptjs_1 = __importDefault(require("bcryptjs"));
const jsonwebtoken_1 = __importDefault(require("jsonwebtoken"));
const testApp_1 = require("./testApp");
let testPrisma;
const createdUsers = [];
const createdBots = [];
const initializeTestHelpers = () => {
    testPrisma = (0, testApp_1.getTestPrisma)();
};
exports.initializeTestHelpers = initializeTestHelpers;
const createTestUser = async (userData) => {
    if (!testPrisma) {
        testPrisma = (0, testApp_1.getTestPrisma)();
    }
    const defaultUserData = {
        email: 'test@example.com',
        name: 'Test User',
        password: 'password123',
    };
    const finalUserData = { ...defaultUserData, ...userData };
    const hashedPassword = await bcryptjs_1.default.hash(finalUserData.password, 10);
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
    const token = jsonwebtoken_1.default.sign({ userId: user.id, email: user.email }, process.env.JWT_SECRET || 'test-secret', { expiresIn: '1h' });
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
exports.createTestUser = createTestUser;
const createTestBot = async (userId, botData) => {
    if (!testPrisma) {
        testPrisma = (0, testApp_1.getTestPrisma)();
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
exports.createTestBot = createTestBot;
const createTestMarketData = async (symbol, count = 100) => {
    if (!testPrisma) {
        testPrisma = (0, testApp_1.getTestPrisma)();
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
    }
    catch (error) {
        // If MarketData model doesn't exist, just return the mock data
        console.warn('MarketData model not found, returning mock data');
    }
    return marketData;
};
exports.createTestMarketData = createTestMarketData;
const cleanupTestData = async () => {
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
    }
    catch (error) {
        console.error('Error cleaning up test data:', error);
    }
};
exports.cleanupTestData = cleanupTestData;
const mockApiResponse = (data, success = true) => ({
    success,
    data: success ? data : undefined,
    message: success ? 'Success' : 'Error',
    error: success ? undefined : 'Test error',
});
exports.mockApiResponse = mockApiResponse;
const waitFor = (ms) => {
    return new Promise(resolve => setTimeout(resolve, ms));
};
exports.waitFor = waitFor;
const generateRandomString = (length = 10) => {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let result = '';
    for (let i = 0; i < length; i++) {
        result += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return result;
};
exports.generateRandomString = generateRandomString;
const generateRandomEmail = () => {
    return `test-${(0, exports.generateRandomString)(8)}@example.com`;
};
exports.generateRandomEmail = generateRandomEmail;
const mockPerformanceMetrics = () => ({
    pageLoadTime: 1000 + Math.random() * 500,
    firstContentfulPaint: 800 + Math.random() * 400,
    largestContentfulPaint: 1200 + Math.random() * 600,
    cumulativeLayoutShift: Math.random() * 0.1,
    firstInputDelay: Math.random() * 100,
    timeToInteractive: 1500 + Math.random() * 1000,
});
exports.mockPerformanceMetrics = mockPerformanceMetrics;
const mockBotStatus = () => ({
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
exports.mockBotStatus = mockBotStatus;
const mockMarketData = (symbol) => ({
    symbol,
    price: 50000 + (Math.random() - 0.5) * 10000,
    change24h: (Math.random() - 0.5) * 10,
    volume24h: Math.random() * 1000000,
    timestamp: Date.now(),
});
exports.mockMarketData = mockMarketData;
// Database transaction helper for tests
const withTransaction = async (callback) => {
    if (!testPrisma) {
        testPrisma = (0, testApp_1.getTestPrisma)();
    }
    return await testPrisma.$transaction(async (tx) => {
        return await callback(tx);
    });
};
exports.withTransaction = withTransaction;
// Reset database to clean state
const resetTestDatabase = async () => {
    if (!testPrisma) {
        return;
    }
    // Delete all test data in correct order
    await testPrisma.bot.deleteMany();
    await testPrisma.user.deleteMany();
    // Reset sequences if using PostgreSQL
    try {
        await testPrisma.$executeRaw `ALTER SEQUENCE "User_id_seq" RESTART WITH 1`;
        await testPrisma.$executeRaw `ALTER SEQUENCE "Bot_id_seq" RESTART WITH 1`;
    }
    catch (error) {
        // Ignore if not using PostgreSQL or sequences don't exist
    }
};
exports.resetTestDatabase = resetTestDatabase;
/**
 * Generate test market data for ML models
 */
const generateTestMarketData = (symbol, periods = 100) => {
    const data = [];
    let basePrice = 50000; // Starting price
    // Generate OHLCV data
    for (let i = 0; i < periods; i++) {
        const volatility = 0.02; // 2% volatility
        const change = (Math.random() - 0.5) * volatility;
        const open = basePrice;
        const close = basePrice * (1 + change);
        const high = Math.max(open, close) * (1 + Math.random() * 0.01);
        const low = Math.min(open, close) * (1 - Math.random() * 0.01);
        const volume = 1000 + Math.random() * 5000;
        // Create feature vector [open, high, low, close, volume, ...]
        const features = [
            open,
            high,
            low,
            close,
            volume,
            // Add some technical indicators
            close / open - 1, // Price change
            (high - low) / open, // Range
            volume / 1000, // Normalized volume
            Math.sin(i * 0.1), // Cyclical feature
            Math.cos(i * 0.1), // Cyclical feature
            // Pad to 20 features
            ...Array(10).fill(0).map(() => Math.random() * 0.1 - 0.05)
        ];
        data.push([features]);
        basePrice = close;
    }
    // Return as batch with shape [1, periods, features]
    return [data.map(d => d[0])];
};
exports.generateTestMarketData = generateTestMarketData;
/**
 * Generate test performance metrics
 */
const generateTestPerformance = () => {
    const totalReturn = (Math.random() - 0.3) * 0.5; // -30% to +20%
    const winRate = Math.random() * 0.4 + 0.4; // 40% to 80%
    const totalTrades = Math.floor(Math.random() * 100) + 10; // 10 to 110 trades
    return {
        totalReturn,
        totalReturnPercent: totalReturn * 100,
        annualizedReturn: totalReturn * 12, // Assuming monthly data
        sharpeRatio: Math.random() * 3 - 0.5, // -0.5 to 2.5
        maxDrawdown: -Math.random() * 0.3, // 0% to -30%
        maxDrawdownPercent: -Math.random() * 30,
        winRate: winRate * 100,
        profitFactor: winRate / (1 - winRate) * (1 + Math.random()),
        totalTrades,
        winningTrades: Math.floor(totalTrades * winRate),
        losingTrades: Math.floor(totalTrades * (1 - winRate)),
        averageWin: Math.random() * 200 + 50, // $50 to $250
        averageLoss: -(Math.random() * 150 + 25), // -$25 to -$175
        largestWin: Math.random() * 1000 + 100, // $100 to $1100
        largestLoss: -(Math.random() * 800 + 50) // -$50 to -$850
    };
};
exports.generateTestPerformance = generateTestPerformance;
/**
 * Wait for async operation with timeout
 */
const waitForCondition = async (condition, timeout = 10000, interval = 100) => {
    const startTime = Date.now();
    while (Date.now() - startTime < timeout) {
        if (await condition()) {
            return true;
        }
        await new Promise(resolve => setTimeout(resolve, interval));
    }
    return false;
};
exports.waitForCondition = waitForCondition;
/**
 * Generate load test data
 */
const generateLoadTestData = (userCount = 10, botsPerUser = 2) => {
    const users = [];
    for (let i = 0; i < userCount; i++) {
        const user = {
            email: `loadtest-user-${i}@example.com`,
            username: `loadtestuser${i}`,
            password: 'loadtest123',
            bots: []
        };
        for (let j = 0; j < botsPerUser; j++) {
            user.bots.push({
                name: `Load Test Bot ${i}-${j}`,
                symbol: ['BTCUSD', 'ETHUSD', 'ADAUSD'][j % 3],
                strategy: ['enhanced_ml', 'fibonacci_ml'][j % 2],
                timeframe: ['1h', '4h', '1d'][j % 3]
            });
        }
        users.push(user);
    }
    return users;
};
exports.generateLoadTestData = generateLoadTestData;
