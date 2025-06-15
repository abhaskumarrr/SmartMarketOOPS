/**
 * Global Test Teardown
 * Runs once after all tests
 */

import { PrismaClient } from '@prisma/client';

export default async function globalTeardown() {
  console.log('🧹 Cleaning up test environment...');

  try {
    // Clean up test database
    if (process.env.TEST_DATABASE_URL) {
      console.log('📊 Cleaning up test database...');
      
      const prisma = new PrismaClient({
        datasources: {
          db: {
            url: process.env.TEST_DATABASE_URL,
          },
        },
      });

      // Clean up all test data
      await prisma.bot.deleteMany();
      await prisma.user.deleteMany();
      
      // Add other cleanup operations as needed
      
      await prisma.$disconnect();
      console.log('✅ Test database cleaned up');
    }

    // Clean up any other resources
    console.log('✅ Test environment cleanup complete');

  } catch (error) {
    console.error('❌ Failed to cleanup test environment:', error);
    // Don't throw error in teardown to avoid masking test failures
  }
}
