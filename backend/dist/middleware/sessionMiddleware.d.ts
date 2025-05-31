/**
 * Session Middleware
 * Handles session validation, activity tracking, and timeout management
 */
import { Request, Response, NextFunction } from 'express';
import { AuthenticatedRequest } from '../types/auth';
export declare const cookieOptions: {
    httpOnly: boolean;
    secure: boolean;
    sameSite: "strict";
    maxAge: number;
    path: string;
    domain: string;
};
/**
 * Cookie parser middleware with signing
 */
export declare const secureCookieParser: import("express").RequestHandler<import("express-serve-static-core").ParamsDictionary, any, any, import("qs").ParsedQs, Record<string, any>>;
/**
 * Middleware to track user activity and update session
 */
export declare const sessionActivity: (req: AuthenticatedRequest, res: Response, next: NextFunction) => Promise<void>;
/**
 * Session validation middleware
 * More comprehensive than the basic JWT check
 */
export declare const validateUserSession: (req: AuthenticatedRequest, res: Response, next: NextFunction) => Promise<void>;
/**
 * Middleware to set session tracking cookie
 */
export declare const setDeviceIdCookie: (req: Request, res: Response, next: NextFunction) => void;
/**
 * Remember Me cookie handling
 */
export declare const setRememberMeCookie: (req: Request, res: Response, rememberMe: boolean) => void;
declare const _default: {
    secureCookieParser: import("express").RequestHandler<import("express-serve-static-core").ParamsDictionary, any, any, import("qs").ParsedQs, Record<string, any>>;
    sessionActivity: (req: AuthenticatedRequest, res: Response, next: NextFunction) => Promise<void>;
    validateUserSession: (req: AuthenticatedRequest, res: Response, next: NextFunction) => Promise<void>;
    setDeviceIdCookie: (req: Request, res: Response, next: NextFunction) => void;
    setRememberMeCookie: (req: Request, res: Response, rememberMe: boolean) => void;
    cookieOptions: {
        httpOnly: boolean;
        secure: boolean;
        sameSite: "strict";
        maxAge: number;
        path: string;
        domain: string;
    };
};
export default _default;
