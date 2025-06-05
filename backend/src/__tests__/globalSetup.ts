/**
 * Global Test Setup
 * Runs once before all tests
 */

import { config } from 'dotenv';
import { execSync } from 'child_process';
import { PrismaClient } from '@prisma/client';

export default async function globalSetup() {
  console.log('🚀 Setting up test environment...');

  // Load test environment variables
  config({ path: '.env.test' });

  // Ensure test database URL is set
  if (!process.env.TEST_DATABASE_URL) {
    throw new Error('TEST_DATABASE_URL environment variable is required for testing');
  }

  try {
    // Initialize test database
    console.log('📊 Initializing test database...');
    
    // Run database migrations
    execSync('npx prisma migrate deploy', {
      env: {
        ...process.env,
        DATABASE_URL: process.env.TEST_DATABASE_URL,
      },
      stdio: 'inherit',
    });

    // Generate Prisma client
    execSync('npx prisma generate', {
      stdio: 'inherit',
    });

    // Verify database connection
    const prisma = new PrismaClient({
      datasources: {
        db: {
          url: process.env.TEST_DATABASE_URL,
        },
      },
    });

    await prisma.$connect();
    console.log('✅ Database connection verified');
    await prisma.$disconnect();

    // Set up Redis test instance (if needed)
    if (process.env.REDIS_URL) {
      console.log('🔴 Redis test instance configured');
    }

    console.log('✅ Test environment setup complete');

  } catch (error) {
    console.error('❌ Failed to setup test environment:', error);
    throw error;
  }
}
