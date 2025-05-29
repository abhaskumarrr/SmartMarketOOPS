/**
 * Database Health Check Utility
 * Provides functions to check database connectivity and health
 */

const prisma = require('./prismaClient');

/**
 * Check database connection by performing a simple query
 * @returns {Promise<boolean>} True if connection is successful
 */
const checkDatabaseConnection = async () => {
  try {
    // Execute a simple query to verify the connection
    await prisma.$queryRaw`SELECT 1 as result`;
    return true;
  } catch (error) {
    console.error('Database connection check failed:', error);
    return false;
  }
};

/**
 * Get database health metrics
 * @returns {Promise<object>} Database health metrics
 */
const getDatabaseMetrics = async () => {
  try {
    // Check connection first
    const isConnected = await checkDatabaseConnection();
    
    if (!isConnected) {
      return {
        status: 'error',
        connected: false,
        message: 'Database connection failed',
      };
    }
    
    // Get table statistics (example for PostgreSQL)
    // This might need to be adjusted based on your actual database
    const tables = await prisma.$queryRaw`
      SELECT 
        table_name, 
        pg_size_pretty(pg_relation_size(quote_ident(table_name))) as size,
        pg_size_pretty(pg_total_relation_size(quote_ident(table_name))) as total_size
      FROM information_schema.tables 
      WHERE table_schema = 'public'
      ORDER BY pg_relation_size(quote_ident(table_name)) DESC
      LIMIT 10;
    `;
    
    return {
      status: 'ok',
      connected: true,
      tables,
    };
  } catch (error) {
    console.error('Failed to get database metrics:', error);
    return {
      status: 'error',
      connected: false,
      message: `Failed to get database metrics: ${error.message}`,
    };
  }
};

module.exports = {
  checkDatabaseConnection,
  getDatabaseMetrics,
}; 