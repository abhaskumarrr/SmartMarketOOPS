/**
 * Database Health Check Utility
 * Functions for checking database connectivity and health
 */

import prisma from './prismaClient';
import { createLogger } from './logger';

// Create logger
const logger = createLogger('DBHealthCheck');

interface ConnectionStatus {
  success: boolean;
  message: string;
  responseTime: string;
  error?: string;
}

interface TableStatus {
  success: boolean;
  error?: string;
}

interface TablesStatus {
  users: TableStatus;
  [key: string]: TableStatus;
}

interface HealthCheckResult {
  success: boolean;
  connection: ConnectionStatus;
  tables: TablesStatus | null;
  migrations: TableStatus | null;
  message?: string;
  error?: string;
}

/**
 * Checks the database connection health
 * @returns {Promise<ConnectionStatus>} The health check result
 */
async function checkDbConnection(): Promise<ConnectionStatus> {
  const startTime = Date.now();
  
  try {
    // Execute a simple query to verify database connection
    await prisma.$queryRaw`SELECT 1`;
    
    // Calculate query execution time
    const responseTime = Date.now() - startTime;
    
    return {
      success: true,
      message: 'Database connection is healthy',
      responseTime: `${responseTime}ms`
    };
  } catch (error: any) {
    return {
      success: false,
      message: 'Database connection failed',
      error: error.message,
      responseTime: `${Date.now() - startTime}ms`
    };
  }
}

/**
 * Performs a comprehensive database health check
 * @returns {Promise<HealthCheckResult>} Detailed health check result
 */
async function performDatabaseHealthCheck(): Promise<HealthCheckResult> {
  try {
    // Check connection
    const connectionStatus = await checkDbConnection();
    
    // If connection failed, return early
    if (!connectionStatus.success) {
      return {
        success: false,
        connection: connectionStatus,
        tables: null,
        migrations: null
      };
    }
    
    // Check tables (users)
    let userTableStatus: TableStatus = { success: true };
    try {
      await prisma.user.findFirst();
    } catch (error: any) {
      userTableStatus = {
        success: false,
        error: error.message
      };
    }
    
    // Check migrations table
    let migrationsStatus: TableStatus = { success: true };
    try {
      await prisma.$queryRaw`SELECT * FROM "_prisma_migrations" LIMIT 1`;
    } catch (error: any) {
      migrationsStatus = {
        success: false,
        error: error.message
      };
    }
    
    return {
      success: connectionStatus.success && userTableStatus.success && migrationsStatus.success,
      connection: connectionStatus,
      tables: {
        users: userTableStatus
      },
      migrations: migrationsStatus
    };
  } catch (error: any) {
    return {
      success: false,
      message: 'Comprehensive health check failed',
      error: error.message,
      connection: {
        success: false,
        message: 'Health check error',
        responseTime: '0ms'
      },
      tables: null,
      migrations: null
    };
  }
}

export {
  checkDbConnection,
  performDatabaseHealthCheck,
  ConnectionStatus,
  TableStatus,
  TablesStatus,
  HealthCheckResult
}; 