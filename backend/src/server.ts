import express from 'express';
import cors from 'cors';
import http from 'http';
import 'dotenv/config';
import { logger } from './utils/logger';
import healthRoutes from './routes/healthRoutes';
import tradingRoutes from './routes/tradingRoutes';
import dashboardRoutes from './routes/dashboardRoutes';
import { errorHandler, notFoundHandler } from './middleware/errorHandler';

const app = express();
const PORT = process.env.PORT || 3000;

logger.info('🚀 Starting server initialization...');
logger.info(`📊 Environment: ${process.env.NODE_ENV || 'development'}`);
logger.info(`🔌 Target port: ${PORT}`);

// --- Essential Middleware ---
logger.info('⚙️ Setting up middleware...');
// Enable Cross-Origin Resource Sharing
app.use(cors());
// Parse JSON bodies
app.use(express.json());
// Parse URL-encoded bodies
app.use(express.urlencoded({ extended: true }));

logger.info('🛣️ Setting up routes...');
// --- Core Routes ---
// Health check endpoint to verify service is up
app.use('/api/health', healthRoutes);
// Dashboard data endpoints
app.use('/api/dashboard', dashboardRoutes);
// The primary trading signal route
app.use('/api/trading', tradingRoutes);

logger.info('🛡️ Setting up error handling...');
// --- Error Handling ---
// Handle 404 for routes not found
app.use(notFoundHandler);
// Centralized error handler
app.use(errorHandler);

logger.info('🏗️ Creating HTTP server...');
const server = http.createServer(app);

logger.info('👂 Starting to listen...');
server.listen(PORT, () => {
  logger.info(`🚀 Server is running on http://localhost:${PORT}`);
  logger.info('✅ Server startup complete!');
});

server.on('error', (error: any) => {
  logger.error('❌ Server error:', error);
  process.exit(1);
});

const gracefulShutdown = async () => {
  logger.info('🔌 Server is shutting down...');
  server.close(async () => {
    logger.info('✅ HTTP server closed.');
    // TODO: Add prisma.$disconnect() when database is set up
    logger.info('🔚 Server shutdown complete.');
    process.exit(0);
  });
};

process.on('SIGINT', gracefulShutdown);
process.on('SIGTERM', gracefulShutdown);

logger.info('🎯 Server setup complete, waiting for connections...'); 