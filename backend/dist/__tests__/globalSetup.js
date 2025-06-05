"use strict";
/**
 * Global Test Setup
 * Runs once before all tests
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.default = globalSetup;
const dotenv_1 = require("dotenv");
const child_process_1 = require("child_process");
const client_1 = require("@prisma/client");
async function globalSetup() {
    console.log('🚀 Setting up test environment...');
    // Load test environment variables
    (0, dotenv_1.config)({ path: '.env.test' });
    // Ensure test database URL is set
    if (!process.env.TEST_DATABASE_URL) {
        throw new Error('TEST_DATABASE_URL environment variable is required for testing');
    }
    try {
        // Initialize test database
        console.log('📊 Initializing test database...');
        // Run database migrations
        (0, child_process_1.execSync)('npx prisma migrate deploy', {
            env: {
                ...process.env,
                DATABASE_URL: process.env.TEST_DATABASE_URL,
            },
            stdio: 'inherit',
        });
        // Generate Prisma client
        (0, child_process_1.execSync)('npx prisma generate', {
            stdio: 'inherit',
        });
        // Verify database connection
        const prisma = new client_1.PrismaClient({
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
    }
    catch (error) {
        console.error('❌ Failed to setup test environment:', error);
        throw error;
    }
}
//# sourceMappingURL=globalSetup.js.map