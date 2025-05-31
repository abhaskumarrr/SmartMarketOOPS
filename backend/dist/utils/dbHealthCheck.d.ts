/**
 * Database Health Check Utility
 * Functions for checking database connectivity and health
 */
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
declare function checkDbConnection(): Promise<ConnectionStatus>;
/**
 * Performs a comprehensive database health check
 * @returns {Promise<HealthCheckResult>} Detailed health check result
 */
declare function performDatabaseHealthCheck(): Promise<HealthCheckResult>;
export { checkDbConnection, performDatabaseHealthCheck, ConnectionStatus, TableStatus, TablesStatus, HealthCheckResult };
