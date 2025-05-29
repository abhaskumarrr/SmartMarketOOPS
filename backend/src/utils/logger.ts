/**
 * Logger Utility
 * Provides a consistent logging interface across the application
 */

// Log levels in order of verbosity
export enum LogLevel {
  ERROR = 0,
  WARN = 1,
  INFO = 2,
  DEBUG = 3
}

// Type for additional log data
export interface LogData {
  [key: string]: any;
}

// Current log level from environment variable or default to INFO
const currentLogLevel: LogLevel = 
  process.env.LOG_LEVEL ? 
    LogLevel[process.env.LOG_LEVEL as keyof typeof LogLevel] : 
    LogLevel.INFO;

/**
 * Logger class
 */
export class Logger {
  private moduleName: string;
  
  /**
   * Create a new logger instance
   * @param moduleName - The name of the module using this logger
   */
  constructor(moduleName: string) {
    this.moduleName = moduleName;
  }
  
  /**
   * Log an error message
   * @param message - The message to log
   * @param error - Optional error object or additional data
   */
  error(message: string, error?: Error | LogData | null): void {
    if (currentLogLevel >= LogLevel.ERROR) {
      const timestamp = new Date().toISOString();
      const prefix = `[${timestamp}] [ERROR] [${this.moduleName}]`;
      
      console.error(`${prefix} ${message}`);
      
      if (error) {
        if (error instanceof Error) {
          console.error(`${prefix} Stack:`, error.stack);
        } else {
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
  warn(message: string, data?: LogData | null): void {
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
  info(message: string, data?: LogData | null): void {
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
  debug(message: string, data?: LogData | null): void {
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

/**
 * Create a new logger instance
 * @param moduleName - The name of the module using this logger
 * @returns A logger instance
 */
export function createLogger(moduleName: string): Logger {
  return new Logger(moduleName);
}

// Export a default logger for use in places where a specific module name isn't needed
export default new Logger('App'); 