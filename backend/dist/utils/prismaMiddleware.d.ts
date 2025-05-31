/**
 * Prisma Middleware Utilities
 * Middleware for logging and validation of database operations
 */
import { PrismaClient } from '@prisma/client';
/**
 * Apply middleware to a Prisma client instance
 * @param {PrismaClient} prisma - Prisma client instance
 * @returns {PrismaClient} - Prisma client with middleware applied
 */
declare function applyMiddleware(prisma: PrismaClient): PrismaClient;
export { applyMiddleware };
