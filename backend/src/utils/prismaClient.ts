/**
 * Central Prisma Client
 * Singleton to avoid multiple Prisma Client instances
 */

import { PrismaClient } from '../../generated/prisma';

// Add prisma to the NodeJS global type
interface CustomNodeJsGlobal {
  prisma: PrismaClient;
}

// Prevent multiple instances of Prisma Client in development
declare const global: CustomNodeJsGlobal & typeof globalThis;

// Initialize Prisma Client
const prisma = global.prisma || new PrismaClient();

// Attach to global in non-production environments
if (process.env.NODE_ENV !== 'production') {
  global.prisma = prisma;
}

// Log Prisma initialization
console.log('Prisma Client initialized successfully');

export default prisma; 