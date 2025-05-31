/**
 * Logger Utility
 * Provides a consistent logging interface across the application
 */
export declare enum LogLevel {
    ERROR = 0,
    WARN = 1,
    INFO = 2,
    DEBUG = 3
}
export interface LogData {
    [key: string]: any;
}
/**
 * Logger class
 */
export declare class Logger {
    private moduleName;
    /**
     * Create a new logger instance
     * @param moduleName - The name of the module using this logger
     */
    constructor(moduleName: string);
    /**
     * Log an error message
     * @param message - The message to log
     * @param error - Optional error object or additional data
     */
    error(message: string, error?: Error | LogData | null): void;
    /**
     * Log a warning message
     * @param message - The message to log
     * @param data - Optional additional data
     */
    warn(message: string, data?: LogData | null): void;
    /**
     * Log an info message
     * @param message - The message to log
     * @param data - Optional additional data
     */
    info(message: string, data?: LogData | null): void;
    /**
     * Log a debug message
     * @param message - The message to log
     * @param data - Optional additional data
     */
    debug(message: string, data?: LogData | null): void;
}
/**
 * Create a new logger instance
 * @param moduleName - The name of the module using this logger
 * @returns A logger instance
 */
export declare function createLogger(moduleName: string): Logger;
declare const _default: Logger;
export default _default;
