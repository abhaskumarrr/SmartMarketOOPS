"use strict";
/**
 * Test App Setup
 * Creates a test instance of the Express app for testing
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.closeTestApp = exports.getTestPrisma = exports.createTestApp = void 0;
const express_1 = __importDefault(require("express"));
const cors_1 = __importDefault(require("cors"));
const helmet_1 = __importDefault(require("helmet"));
const compression_1 = __importDefault(require("compression"));
const cookie_parser_1 = __importDefault(require("cookie-parser"));
const client_1 = require("@prisma/client");
const optimizationMiddleware_1 = __importDefault(require("../../middleware/optimizationMiddleware"));
const errorHandler_1 = require("../../middleware/errorHandler");
// Import routes
const authRoutes_1 = __importDefault(require("../../routes/authRoutes"));
const botRoutes_1 = __importDefault(require("../../routes/botRoutes"));
const userRoutes_1 = __importDefault(require("../../routes/userRoutes"));
// Test database instance
let testPrisma;
const createTestApp = async () => {
    const app = (0, express_1.default)();
    // Initialize test database
    testPrisma = new client_1.PrismaClient({
        datasources: {
            db: {
                url: process.env.TEST_DATABASE_URL || 'postgresql://test:test@localhost:5432/smartmarket_test',
            },
        },
    });
    // Basic middleware
    app.use((0, helmet_1.default)({
        contentSecurityPolicy: false, // Disable for testing
    }));
    app.use((0, cors_1.default)({
        origin: true,
        credentials: true,
    }));
    app.use((0, compression_1.default)());
    app.use(express_1.default.json({ limit: '10mb' }));
    app.use(express_1.default.urlencoded({ extended: true, limit: '10mb' }));
    app.use((0, cookie_parser_1.default)());
    // Performance monitoring (but don't enforce rate limits in tests)
    app.use(optimizationMiddleware_1.default.performanceMonitor());
    app.use(optimizationMiddleware_1.default.securityHeaders());
    app.use(optimizationMiddleware_1.default.requestValidation());
    // Health check
    app.get('/health', (req, res) => {
        res.json({
            status: 'healthy',
            timestamp: new Date().toISOString(),
            uptime: process.uptime(),
            environment: 'test',
        });
    });
    // API routes
    app.use('/api/auth', authRoutes_1.default);
    app.use('/api/bots', botRoutes_1.default);
    app.use('/api/users', userRoutes_1.default);
    // Error handling
    app.use(errorHandler_1.notFoundHandler);
    app.use(optimizationMiddleware_1.default.errorHandler());
    return app;
};
exports.createTestApp = createTestApp;
const getTestPrisma = () => {
    return testPrisma;
};
exports.getTestPrisma = getTestPrisma;
const closeTestApp = async () => {
    if (testPrisma) {
        await testPrisma.$disconnect();
    }
};
exports.closeTestApp = closeTestApp;
//# sourceMappingURL=testApp.js.map