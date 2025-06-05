"use strict";
/**
 * Global Test Teardown
 * Runs once after all tests
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.default = globalTeardown;
const client_1 = require("@prisma/client");
async function globalTeardown() {
    console.log('🧹 Cleaning up test environment...');
    try {
        // Clean up test database
        if (process.env.TEST_DATABASE_URL) {
            console.log('📊 Cleaning up test database...');
            const prisma = new client_1.PrismaClient({
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
    }
    catch (error) {
        console.error('❌ Failed to cleanup test environment:', error);
        // Don't throw error in teardown to avoid masking test failures
    }
}
//# sourceMappingURL=globalTeardown.js.map