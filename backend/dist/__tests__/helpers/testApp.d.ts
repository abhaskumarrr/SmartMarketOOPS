/**
 * Test App Setup
 * Creates a test instance of the Express app for testing
 */
import { Express } from 'express';
import { PrismaClient } from '@prisma/client';
export declare const createTestApp: () => Promise<Express>;
export declare const getTestPrisma: () => PrismaClient;
export declare const closeTestApp: () => Promise<void>;
