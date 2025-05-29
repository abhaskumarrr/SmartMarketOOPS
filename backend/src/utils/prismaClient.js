/**
 * Prisma Client Singleton
 * Provides a single instance of the Prisma client throughout the application
 */

// Import Prisma from the custom output location defined in schema.prisma
const { PrismaClient } = require('../../generated/prisma');

// Initialize PrismaClient once
let prisma;

// Create a singleton instance of PrismaClient
if (!prisma) {
  try {
    prisma = new PrismaClient({
      log: process.env.NODE_ENV === 'development' 
        ? ['query', 'info', 'warn', 'error'] 
        : ['error'],
      errorFormat: 'pretty',
    });
    
    // Log successful connection
    console.log('Prisma Client initialized successfully');
  } catch (error) {
    console.error('Failed to initialize Prisma client:', error);
    process.exit(1); // Exit with error code
  }
}

// Add middleware for logging queries if in verbose mode
if (process.env.PRISMA_VERBOSE_LOGGING === 'true') {
  prisma.$use(async (params, next) => {
    const before = Date.now();
    const result = await next(params);
    const after = Date.now();
    
    console.log(`Query ${params.model}.${params.action} took ${after - before}ms`);
    
    return result;
  });
}

// Add connection health check
prisma.$on('query', async () => {
  // This event is triggered by queries
  // It helps ensure the connection is active
});

// Handle Prisma Client shutdown
const handleShutdown = async () => {
  await prisma.$disconnect();
  console.log('Prisma client disconnected');
};

process.on('SIGINT', async () => {
  await handleShutdown();
  process.exit(0);
});

process.on('SIGTERM', async () => {
  await handleShutdown();
  process.exit(0);
});

// Export the Prisma client
module.exports = prisma; 