/**
 * Common Types
 * Shared type definitions used across the application
 */

/**
 * Timestamp represents a date-time string in ISO format
 */
export type Timestamp = string;

/**
 * UUID represents a unique identifier string
 */
export type UUID = string;

/**
 * JSON represents a JSON-serializable value
 */
export type JSON = 
  | string
  | number
  | boolean
  | null
  | JSON[]
  | { [key: string]: JSON };

/**
 * ApiResponse represents a standardized API response format
 */
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
    details?: any;
  };
  timestamp: Timestamp;
}

/**
 * PaginatedResponse represents a paginated API response
 */
export interface PaginatedResponse<T = any> {
  items: T[];
  page: number;
  limit: number;
  totalItems: number;
  totalPages: number;
  hasMore: boolean;
}

/**
 * QueryParams represents common query parameters for API endpoints
 */
export interface QueryParams {
  page?: number;
  limit?: number;
  sort?: string;
  order?: 'asc' | 'desc';
  search?: string;
  fields?: string[];
  [key: string]: any;
}

/**
 * TimeRange represents a time range with start and end timestamps
 */
export interface TimeRange {
  start: Timestamp;
  end: Timestamp;
} 