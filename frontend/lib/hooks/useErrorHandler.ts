/**
 * Error Handler Hook
 * Centralized error handling with automatic retry and user feedback
 */

import { useState, useCallback, useRef } from 'react';
import { notificationService } from '../services/notificationService';

export interface ErrorInfo {
  message: string;
  code?: string | number;
  type: 'network' | 'api' | 'validation' | 'auth' | 'unknown';
  retryable: boolean;
  context?: string;
  originalError?: any;
}

export interface RetryConfig {
  maxAttempts: number;
  delay: number;
  backoff: boolean;
  retryCondition?: (error: any) => boolean;
}

export interface UseErrorHandlerOptions {
  showNotifications?: boolean;
  autoRetry?: boolean;
  retryConfig?: Partial<RetryConfig>;
  onError?: (error: ErrorInfo) => void;
  onRetry?: (attempt: number) => void;
  onMaxRetriesReached?: (error: ErrorInfo) => void;
}

const defaultRetryConfig: RetryConfig = {
  maxAttempts: 3,
  delay: 1000,
  backoff: true,
  retryCondition: (error) => {
    // Retry on network errors and 5xx server errors
    return (
      !error.response || 
      error.response.status >= 500 || 
      error.code === 'NETWORK_ERROR'
    );
  },
};

export const useErrorHandler = (options: UseErrorHandlerOptions = {}) => {
  const {
    showNotifications = true,
    autoRetry = false,
    retryConfig = {},
    onError,
    onRetry,
    onMaxRetriesReached,
  } = options;

  const [errors, setErrors] = useState<ErrorInfo[]>([]);
  const [isRetrying, setIsRetrying] = useState(false);
  const retryAttempts = useRef<Map<string, number>>(new Map());
  const finalRetryConfig = { ...defaultRetryConfig, ...retryConfig };

  const parseError = useCallback((error: any, context?: string): ErrorInfo => {
    let errorInfo: ErrorInfo = {
      message: 'An unexpected error occurred',
      type: 'unknown',
      retryable: false,
      context,
      originalError: error,
    };

    // Network errors
    if (!error.response && error.request) {
      errorInfo = {
        ...errorInfo,
        message: 'Network connection failed',
        type: 'network',
        retryable: true,
        code: 'NETWORK_ERROR',
      };
    }
    // API errors
    else if (error.response) {
      const { status, data } = error.response;
      
      errorInfo.code = status;
      errorInfo.type = 'api';
      
      // Extract error message
      if (data?.message) {
        errorInfo.message = data.message;
      } else if (data?.error) {
        errorInfo.message = data.error;
      } else {
        errorInfo.message = `Server error (${status})`;
      }

      // Determine if retryable
      errorInfo.retryable = status >= 500 || status === 429; // Server errors and rate limiting

      // Specific error types
      if (status === 401 || status === 403) {
        errorInfo.type = 'auth';
        errorInfo.message = status === 401 ? 'Authentication required' : 'Access denied';
      } else if (status >= 400 && status < 500) {
        errorInfo.type = 'validation';
        errorInfo.retryable = false;
      }
    }
    // JavaScript errors
    else if (error instanceof Error) {
      errorInfo = {
        ...errorInfo,
        message: error.message,
        type: 'unknown',
        retryable: false,
      };
    }
    // String errors
    else if (typeof error === 'string') {
      errorInfo.message = error;
    }

    return errorInfo;
  }, []);

  const handleError = useCallback(async (
    error: any,
    context?: string,
    operation?: () => Promise<any>
  ): Promise<any> => {
    const errorInfo = parseError(error, context);
    
    // Add to errors list
    setErrors(prev => [errorInfo, ...prev.slice(0, 9)]); // Keep last 10 errors

    // Call custom error handler
    if (onError) {
      onError(errorInfo);
    }

    // Show notification
    if (showNotifications) {
      if (errorInfo.type === 'network') {
        notificationService.networkError();
      } else {
        notificationService.apiError(error, context);
      }
    }

    // Handle retry logic
    if (autoRetry && operation && errorInfo.retryable) {
      const operationKey = context || 'default';
      const currentAttempts = retryAttempts.current.get(operationKey) || 0;

      if (currentAttempts < finalRetryConfig.maxAttempts) {
        const nextAttempt = currentAttempts + 1;
        retryAttempts.current.set(operationKey, nextAttempt);

        // Check retry condition
        if (finalRetryConfig.retryCondition && !finalRetryConfig.retryCondition(error)) {
          throw error;
        }

        setIsRetrying(true);

        // Call retry callback
        if (onRetry) {
          onRetry(nextAttempt);
        }

        // Calculate delay with optional backoff
        const delay = finalRetryConfig.backoff 
          ? finalRetryConfig.delay * Math.pow(2, currentAttempts)
          : finalRetryConfig.delay;

        // Wait before retry
        await new Promise(resolve => setTimeout(resolve, delay));

        try {
          const result = await operation();
          
          // Reset retry count on success
          retryAttempts.current.delete(operationKey);
          setIsRetrying(false);
          
          if (showNotifications) {
            notificationService.success('Operation completed successfully');
          }
          
          return result;
        } catch (retryError) {
          setIsRetrying(false);
          return handleError(retryError, context, operation);
        }
      } else {
        // Max retries reached
        retryAttempts.current.delete(operationKey);
        setIsRetrying(false);
        
        if (onMaxRetriesReached) {
          onMaxRetriesReached(errorInfo);
        }
        
        if (showNotifications) {
          notificationService.error(
            `Failed after ${finalRetryConfig.maxAttempts} attempts: ${errorInfo.message}`
          );
        }
      }
    }

    throw error;
  }, [
    parseError,
    autoRetry,
    finalRetryConfig,
    showNotifications,
    onError,
    onRetry,
    onMaxRetriesReached,
  ]);

  const retry = useCallback(async (operation: () => Promise<any>, context?: string) => {
    const operationKey = context || 'manual';
    retryAttempts.current.delete(operationKey); // Reset retry count
    
    try {
      return await operation();
    } catch (error) {
      return handleError(error, context, operation);
    }
  }, [handleError]);

  const clearErrors = useCallback(() => {
    setErrors([]);
    retryAttempts.current.clear();
  }, []);

  const removeError = useCallback((index: number) => {
    setErrors(prev => prev.filter((_, i) => i !== index));
  }, []);

  // Wrapper for async operations with error handling
  const withErrorHandling = useCallback(<T>(
    operation: () => Promise<T>,
    context?: string
  ) => {
    return async (): Promise<T> => {
      try {
        return await operation();
      } catch (error) {
        return handleError(error, context, operation);
      }
    };
  }, [handleError]);

  // Wrapper for API calls
  const apiCall = useCallback(async <T>(
    apiFunction: () => Promise<T>,
    context?: string
  ): Promise<T> => {
    try {
      return await apiFunction();
    } catch (error) {
      return handleError(error, context, apiFunction);
    }
  }, [handleError]);

  return {
    // State
    errors,
    isRetrying,
    hasErrors: errors.length > 0,
    
    // Methods
    handleError,
    retry,
    clearErrors,
    removeError,
    withErrorHandling,
    apiCall,
    
    // Utilities
    parseError,
  };
};

// Convenience hook for API calls
export const useApiErrorHandler = (options?: UseErrorHandlerOptions) => {
  const errorHandler = useErrorHandler({
    showNotifications: true,
    autoRetry: true,
    retryConfig: {
      maxAttempts: 2,
      delay: 1000,
    },
    ...options,
  });

  return errorHandler;
};

// Convenience hook for form validation
export const useFormErrorHandler = (options?: UseErrorHandlerOptions) => {
  const errorHandler = useErrorHandler({
    showNotifications: true,
    autoRetry: false,
    ...options,
  });

  return errorHandler;
};

export default useErrorHandler;
