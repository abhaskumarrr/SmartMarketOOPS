"use strict";
/**
 * Logger Utility
 * Provides a consistent logging interface across the application
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.logger = exports.Logger = exports.LogLevel = void 0;
exports.createLogger = createLogger;
// Log levels in order of verbosity
var LogLevel;
(function (LogLevel) {
    LogLevel[LogLevel["ERROR"] = 0] = "ERROR";
    LogLevel[LogLevel["WARN"] = 1] = "WARN";
    LogLevel[LogLevel["INFO"] = 2] = "INFO";
    LogLevel[LogLevel["DEBUG"] = 3] = "DEBUG";
})(LogLevel || (exports.LogLevel = LogLevel = {}));
// Current log level from environment variable or default to INFO
const currentLogLevel = process.env.LOG_LEVEL ?
    LogLevel[process.env.LOG_LEVEL] :
    LogLevel.INFO;
/**
 * Logger class
 */
class Logger {
    /**
     * Create a new logger instance
     * @param moduleName - The name of the module using this logger
     */
    constructor(moduleName) {
        this.moduleName = moduleName;
    }
    /**
     * Log an error message
     * @param message - The message to log
     * @param error - Optional error object or additional data
     */
    error(message, error) {
        if (currentLogLevel >= LogLevel.ERROR) {
            const timestamp = new Date().toISOString();
            const prefix = `[${timestamp}] [ERROR] [${this.moduleName}]`;
            console.error(`${prefix} ${message}`);
            if (error) {
                if (error instanceof Error) {
                    console.error(`${prefix} Stack:`, error.stack);
                }
                else {
                    console.error(`${prefix} Data:`, error);
                }
            }
        }
    }
    /**
     * Log a warning message
     * @param message - The message to log
     * @param data - Optional additional data
     */
    warn(message, data) {
        if (currentLogLevel >= LogLevel.WARN) {
            const timestamp = new Date().toISOString();
            const prefix = `[${timestamp}] [WARN] [${this.moduleName}]`;
            console.warn(`${prefix} ${message}`);
            if (data) {
                console.warn(`${prefix} Data:`, data);
            }
        }
    }
    /**
     * Log an info message
     * @param message - The message to log
     * @param data - Optional additional data
     */
    info(message, data) {
        if (currentLogLevel >= LogLevel.INFO) {
            const timestamp = new Date().toISOString();
            const prefix = `[${timestamp}] [INFO] [${this.moduleName}]`;
            console.info(`${prefix} ${message}`);
            if (data) {
                console.info(`${prefix} Data:`, data);
            }
        }
    }
    /**
     * Log a debug message
     * @param message - The message to log
     * @param data - Optional additional data
     */
    debug(message, data) {
        if (currentLogLevel >= LogLevel.DEBUG) {
            const timestamp = new Date().toISOString();
            const prefix = `[${timestamp}] [DEBUG] [${this.moduleName}]`;
            console.debug(`${prefix} ${message}`);
            if (data) {
                console.debug(`${prefix} Data:`, data);
            }
        }
    }
}
exports.Logger = Logger;
/**
 * Create a new logger instance
 * @param moduleName - The name of the module using this logger
 * @returns A logger instance
 */
function createLogger(moduleName) {
    return new Logger(moduleName);
}
// Export a default logger for use in places where a specific module name isn't needed
exports.logger = new Logger('App');
exports.default = exports.logger;
