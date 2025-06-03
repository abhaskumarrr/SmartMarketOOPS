"use strict";
/**
 * Main Server Entry Point
 * Sets up and starts the Express server with API routes
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.prisma = exports.io = exports.server = exports.app = void 0;
const express_1 = __importDefault(require("express"));
const cors_1 = __importDefault(require("cors"));
const helmet_1 = __importDefault(require("helmet"));
const path_1 = __importDefault(require("path"));
const dotenv_1 = __importDefault(require("dotenv"));
const http_1 = __importDefault(require("http"));
const fs_1 = require("fs");
const prismaClient_1 = __importDefault(require("./utils/prismaClient"));
exports.prisma = prismaClient_1.default;
const errorHandler_1 = require("./middleware/errorHandler");
const sessionMiddleware_1 = require("./middleware/sessionMiddleware");
// Load environment variables
dotenv_1.default.config({
    path: path_1.default.resolve(__dirname, '../../.env')
});
// Import routes
const healthRoutes_1 = __importDefault(require("./routes/healthRoutes"));
const authRoutes_1 = __importDefault(require("./routes/authRoutes"));
const userRoutes_1 = __importDefault(require("./routes/userRoutes"));
const apiKeyRoutes_1 = __importDefault(require("./routes/apiKeyRoutes"));
const roleRoutes_1 = __importDefault(require("./routes/roleRoutes"));
const sessionRoutes_1 = __importDefault(require("./routes/sessionRoutes"));
const signalRoutes_1 = __importDefault(require("./routes/signalRoutes"));
const riskRoutes_1 = __importDefault(require("./routes/riskRoutes"));
const strategyRoutes_1 = __importDefault(require("./routes/strategyRoutes"));
const bridgeRoutes_1 = __importDefault(require("./routes/bridge/bridgeRoutes"));
const performanceRoutes_1 = __importDefault(require("./routes/performance/performanceRoutes"));
const orderExecutionRoutes_1 = __importDefault(require("./routes/trading/orderExecutionRoutes"));
const apiKeyRoutes_2 = __importDefault(require("./routes/trading/apiKeyRoutes"));
const botRoutes_1 = __importDefault(require("./routes/trading/botRoutes"));
const auditRoutes_1 = __importDefault(require("./routes/auditRoutes"));
const trades_1 = __importDefault(require("./routes/trading/trades"));
const metricsRoutes_1 = __importDefault(require("./routes/metricsRoutes"));
const mlRoutes_1 = __importDefault(require("./routes/mlRoutes"));
const marketDataRoutes_1 = __importDefault(require("./routes/marketDataRoutes"));
// import tradingRoutes from './routes/tradingRoutes';
const tradingRoutesWorking = require('./routes/tradingRoutesWorking');
// Import other routes as needed
// Load socket initialization
const initializeWebsocketServer = require('./sockets/websocketServer').initializeWebsocketServer;
// Create Express app
const app = (0, express_1.default)();
exports.app = app;
const PORT = process.env.PORT || 3333;
const NODE_ENV = process.env.NODE_ENV || 'development';
// Create HTTP server for Socket.IO
const server = http_1.default.createServer(app);
exports.server = server;
// Initialize WebSocket server
const io = initializeWebsocketServer(server);
exports.io = io;
// Logging middleware
const logStream = (0, fs_1.createWriteStream)(path_1.default.join(__dirname, '../logs/server.log'), { flags: 'a' });
app.use((req, res, next) => {
    const start = Date.now();
    res.on('finish', () => {
        const duration = Date.now() - start;
        const log = `${new Date().toISOString()} | ${req.method} ${req.url} ${res.statusCode} ${duration}ms\n`;
        logStream.write(log);
        if (NODE_ENV === 'development') {
            console.log(log);
        }
    });
    next();
});
// Security middleware
app.use((0, helmet_1.default)());
// CORS middleware
app.use((0, cors_1.default)({
    origin: [process.env.CLIENT_URL || 'http://localhost:3000', 'http://localhost:3002', 'http://localhost:3333'],
    credentials: true
}));
// Body parser middleware
app.use(express_1.default.json());
app.use(express_1.default.urlencoded({ extended: true }));
// Cookie parser middleware (with signed cookies)
app.use(sessionMiddleware_1.secureCookieParser);
// Set device ID cookie for session tracking
app.use(sessionMiddleware_1.setDeviceIdCookie);
// Set trust proxy if behind a proxy
if (process.env.TRUST_PROXY === 'true') {
    app.set('trust proxy', 1);
}
// Track session activity for authenticated routes
app.use(sessionMiddleware_1.sessionActivity);
// Root route handler - API welcome page
app.get('/', (req, res) => {
    res.status(200).json({
        name: 'SmartMarket OOPS API',
        version: '1.0.0',
        status: 'online',
        timestamp: new Date().toISOString(),
        routes: {
            health: '/api/health',
            auth: '/api/auth',
            users: '/api/users',
            apiKeys: '/api/api-keys',
            trading: {
                orders: '/api/orders',
                apiKeys: '/api/trading/api-keys',
                bots: '/api/bots',
                trades: '/api/trades'
            }
        }
    });
});
// Basic health check at root path
app.get('/health', (req, res) => {
    res.status(200).json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        uptime: process.uptime()
    });
});
// Use routes
app.use('/api/health', healthRoutes_1.default);
app.use('/api/auth', authRoutes_1.default);
app.use('/api/users', userRoutes_1.default);
app.use('/api/api-keys', apiKeyRoutes_1.default);
app.use('/api/roles', roleRoutes_1.default);
app.use('/api/sessions', sessionRoutes_1.default);
app.use('/api/signals', signalRoutes_1.default);
app.use('/api/risk', riskRoutes_1.default);
app.use('/api/strategies', strategyRoutes_1.default);
app.use('/api/bridge', bridgeRoutes_1.default);
app.use('/api/performance', performanceRoutes_1.default);
app.use('/api/orders', orderExecutionRoutes_1.default);
app.use('/api/trading/api-keys', apiKeyRoutes_2.default);
app.use('/api/bots', botRoutes_1.default);
app.use('/api/audit', auditRoutes_1.default);
app.use('/api/trades', trades_1.default);
app.use('/api', metricsRoutes_1.default);
app.use('/api/ml', mlRoutes_1.default);
app.use('/api/market-data', marketDataRoutes_1.default);
// app.use('/api/trading', tradingRoutes);
app.use('/api/trading', tradingRoutesWorking);
// Use other routes as needed
// Not found middleware for undefined routes
app.use(errorHandler_1.notFoundHandler);
// Global error handling middleware
app.use(errorHandler_1.errorHandler);
// Start server
server.listen(PORT, () => {
    console.log(`Server running in ${NODE_ENV} mode on port ${PORT}`);
});
// Handle unhandled promise rejections
process.on('unhandledRejection', (err) => {
    console.error('Unhandled Promise Rejection:', err);
});
// Handle uncaught exceptions
process.on('uncaughtException', (err) => {
    console.error('Uncaught Exception:', err);
    process.exit(1);
});
// Graceful shutdown
process.on('SIGTERM', async () => {
    console.log('SIGTERM received, shutting down gracefully');
    // Close Prisma connection
    await prismaClient_1.default.$disconnect();
    // Close server
    server.close(() => {
        console.log('Server closed');
        process.exit(0);
    });
    // Force close after timeout
    setTimeout(() => {
        console.error('Could not close connections in time, forcefully shutting down');
        process.exit(1);
    }, 10000);
});
//# sourceMappingURL=server.js.map