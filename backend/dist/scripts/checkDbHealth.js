#!/usr/bin/env node
/**
 * Database Health Check Script for CI
 *
 * This script can be run directly in CI to validate database health
 * It will exit with code 0 if healthy, or 1 if any check fails
 */
require('dotenv').config();
const { performHealthCheck } = require('../utils/dbHealthCheck');
async function main() {
    console.log('Starting database health check...');
    try {
        const result = await performHealthCheck();
        // Log detailed results
        console.log('\n--- Database Health Check Results ---');
        console.log(`Overall Health: ${result.isHealthy ? '✅ Healthy' : '❌ Unhealthy'}`);
        console.log(`Connection: ${result.connection.message}`);
        console.log(`Schema: ${result.schema.message}`);
        if (result.schema.missingTables) {
            console.log(`Missing tables: ${result.schema.missingTables.join(', ')}`);
        }
        // Exit with appropriate code
        if (!result.isHealthy) {
            console.error('\n❌ Database health check failed!');
            process.exit(1);
        }
        console.log('\n✅ All database health checks passed!');
        process.exit(0);
    }
    catch (error) {
        console.error('\n❌ Error running health check:', error);
        process.exit(1);
    }
}
// Run the health check
main();
//# sourceMappingURL=checkDbHealth.js.map