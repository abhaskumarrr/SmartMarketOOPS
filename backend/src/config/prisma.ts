import { PrismaClient } from '@prisma/client';
import { logger } from '../utils/logger';

// Instantiate a single Prisma client for the entire application
const prisma = new PrismaClient({
  log: [
    { emit: 'stdout', level: 'query' },
    { emit: 'stdout', level: 'info' },
    { emit: 'stdout', level: 'warn' },
    { emit: 'stdout', level: 'error' },
  ],
});

logger.info('Prisma Client initialized successfully');

export default prisma; 