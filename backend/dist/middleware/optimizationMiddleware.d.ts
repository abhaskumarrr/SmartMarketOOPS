/**
 * API Optimization Middleware
 * Handles rate limiting, compression, caching, and performance monitoring
 */
import { Request, Response, NextFunction } from 'express';
interface RequestMetrics {
    path: string;
    method: string;
    duration: number;
    statusCode: number;
    timestamp: number;
    userAgent?: string;
    ip?: string;
}
declare class OptimizationMiddleware {
    private requestMetrics;
    private maxMetricsHistory;
    performanceMonitor(): (req: Request, res: Response, next: NextFunction) => void;
    compressionMiddleware(): (req: Request, res: Response, next: NextFunction) => void;
    cacheMiddleware(options?: {
        ttl?: number;
        keyGenerator?: (req: Request) => string;
        skipCache?: (req: Request) => boolean;
    }): (req: Request, res: Response, next: NextFunction) => Promise<void | Response<any, Record<string, any>>>;
    createRateLimiter(options?: {
        windowMs?: number;
        max?: number;
        message?: string;
        keyGenerator?: (req: Request) => string;
        skipSuccessfulRequests?: boolean;
        skipFailedRequests?: boolean;
    }): import("express-rate-limit").RateLimitRequestHandler;
    requestValidation(): (req: Request, res: Response, next: NextFunction) => Response<any, Record<string, any>>;
    securityHeaders(): (req: Request, res: Response, next: NextFunction) => void;
    requestTimeout(timeoutMs?: number): (req: Request, res: Response, next: NextFunction) => void;
    errorHandler(): (error: any, req: Request, res: Response, next: NextFunction) => void;
    healthCheck(): (req: Request, res: Response) => Promise<void>;
    private addRequestMetric;
    private getMetricsSummary;
    getMetrics(): {
        recent: {
            requests: number;
            avgResponseTime: number;
            errorRate: number;
        };
        all: RequestMetrics[];
    };
}
declare const optimizationMiddleware: OptimizationMiddleware;
export default optimizationMiddleware;
export { OptimizationMiddleware, RequestMetrics };
