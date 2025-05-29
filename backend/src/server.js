/**
 * Main Server Entry Point
 * Sets up and starts the Express server with API routes
 */

// Load environment variables first before any imports
const dotenv = require('dotenv');
const path = require('path');

// Load environment variables
dotenv.config({
  path: path.resolve(__dirname, '../../.env')
});

// Then import dependencies
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const http = require('http');
const { createWriteStream } = require('fs');
const cookieParser = require('cookie-parser');

// Import environment and configuration
const { env } = require('./utils/env');

// Import middleware
const { errorHandler, notFoundHandler } = require('./middleware/errorHandler');
const { 
  secureCookieParser, 
  sessionActivity, 
  setDeviceIdCookie 
} = require('./middleware/sessionMiddleware');

// Initialize Prisma client after environment is loaded
const prisma = require('./utils/prismaClient');

// Import routes after Prisma client is initialized
const authRoutes = require('./routes/authRoutes');
const userRoutes = require('./routes/userRoutes');
const sessionRoutes = require('./routes/sessionRoutes');
const roleRoutes = require('./routes/roleRoutes');
const healthRoutes = require('./routes/healthRoutes');
const predictionRoutes = require('./routes/predictionRoutes');
const signalRoutes = require('./routes/signalRoutes');
const marketRoutes = require('./routes/marketRoutes');
const botRoutes = require('./routes/botRoutes');
const deltaApiRoutes = require('./routes/deltaApiRoutes');
const apiKeyRoutes = require('./routes/apiKeyRoutes');

// Initialize Express app
const app = express();
const server = http.createServer(app);

// Print environment variables (with sensitive values masked)
console.log('Environment Configuration:');
console.log(JSON.stringify(Object.entries(env).reduce((acc, [key, value]) => {
  if (key.includes('SECRET') || key.includes('KEY') || key.includes('PASSWORD')) {
    acc[key] = '********';
  } else {
    acc[key] = value;
  }
  return acc;
}, {}), null, 2));

// Check for required environment variables
const requiredEnvVars = [
  'DATABASE_URL',
  'JWT_SECRET',
  'JWT_REFRESH_SECRET',
  'COOKIE_SECRET'
];

const missingEnvVars = requiredEnvVars.filter(variable => {
  return !env[variable] || env[variable] === 'not set';
});

if (missingEnvVars.length > 0) {
  console.error('Environment validation errors:');
  missingEnvVars.forEach(variable => {
    console.error(`- ${variable} is invalid or missing`);
  });
}

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(helmet());
app.use(cors({
  origin: env.CORS_ORIGIN,
  credentials: true
}));

// Cookie and session middleware
app.use(secureCookieParser);
app.use(setDeviceIdCookie);
app.use(sessionActivity);

// API Routes
app.use('/api/auth', authRoutes);
app.use('/api/users', userRoutes);
app.use('/api/sessions', sessionRoutes);
app.use('/api/roles', roleRoutes);
app.use('/api/health', healthRoutes);
app.use('/api/predictions', predictionRoutes);
app.use('/api/signals', signalRoutes);
app.use('/api/market', marketRoutes);
app.use('/api/bots', botRoutes);
app.use('/api/delta', deltaApiRoutes);
app.use('/api/keys', apiKeyRoutes);

// Root route handler to fix 404 error
app.get('/', (req, res) => {
  res.json({
    success: true,
    message: 'SmartMarketOOPS API server is running',
    version: '1.0.0',
    endpoints: {
      auth: '/api/auth',
      users: '/api/users',
      sessions: '/api/sessions',
      roles: '/api/roles',
      health: '/api/health',
      predictions: '/api/predictions',
      signals: '/api/signals',
      market: '/api/market',
      bots: '/api/bots',
      delta: '/api/delta',
      keys: '/api/keys'
    }
  });
});

// Not found handler for undefined routes
app.use(notFoundHandler);

// Global error handler
app.use(errorHandler);

// Start server
const PORT = env.PORT || 3001;
const HOST = env.HOST || '0.0.0.0';

server.listen(PORT, HOST, () => {
  console.log(`Server running on http://${HOST}:${PORT}`);
  console.log(`Environment: ${env.NODE_ENV}`);
});

// Handle unhandled promise rejections
process.on('unhandledRejection', (error) => {
  console.error('Unhandled Promise Rejection:', error);
});

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
  // Log to a file
  const logStream = createWriteStream(path.join(__dirname, '../logs/error.log'), { flags: 'a' });
  logStream.write(`${new Date().toISOString()} - Uncaught Exception: ${error.message}\n${error.stack}\n`);
  
  // Graceful shutdown
  server.close(() => {
    console.log('Server shut down due to uncaught exception');
    process.exit(1);
  });
  
  // If server doesn't close in 5 seconds, force exit
  setTimeout(() => {
    console.error('Forcing server shutdown');
    process.exit(1);
  }, 5000);
});

module.exports = { app, server };

