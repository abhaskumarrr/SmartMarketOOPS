/**
 * API Security Service - Phase 4.3 Security Hardening
 * Comprehensive API security implementation with threat detection using built-in modules
 */

import { Request, Response, NextFunction } from 'express';
import validator from 'validator';
import helmet from 'helmet';
import { createHash, randomBytes } from 'crypto';
import { createLogger } from '../../utils/logger';
import { createAuditLog } from '../../utils/auditLog';

const logger = createLogger('ApiSecurityService');

interface SecurityConfig {
  rateLimit: {
    windowMs: number;
    maxRequests: number;
    authMaxRequests: number;
  };
  bruteForce: {
    maxWrongAttemptsByIP: number;
    maxConsecutiveFailsByUser: number;
    blockDuration: number;
  };
  validation: {
    maxPayloadSize: string;
    allowedFileTypes: string[];
    maxFileSize: number;
  };
}

interface RateLimitData {
  count: number;
  resetTime: number;
  blocked?: boolean;
  blockUntil?: number;
}

interface AuthFailureData {
  count: number;
  lastAttempt: number;
  blocked?: boolean;
  blockUntil?: number;
}

export class ApiSecurityService {
  private config: SecurityConfig;
  private suspiciousActivity = new Map<string, number>();
  private blockedIPs = new Set<string>();
  private rateLimitStore = new Map<string, RateLimitData>();
  private authFailures = new Map<string, AuthFailureData>();

  constructor(config: SecurityConfig) {
    this.config = config;
    this.initializeCleanupTasks();
    logger.info('API Security Service initialized');
  }

  /**
   * Initialize cleanup tasks for in-memory stores
   */
  private initializeCleanupTasks(): void {
    // Clean up expired rate limit entries every 5 minutes
    setInterval(() => {
      const now = Date.now();
      for (const [key, data] of this.rateLimitStore.entries()) {
        if (now > data.resetTime && (!data.blockUntil || now > data.blockUntil)) {
          this.rateLimitStore.delete(key);
        }
      }
    }, 5 * 60 * 1000);

    // Clean up old auth failures every hour
    setInterval(() => {
      const now = Date.now();
      const oneDayAgo = now - (24 * 60 * 60 * 1000);
      for (const [key, data] of this.authFailures.entries()) {
        if (data.lastAttempt < oneDayAgo && (!data.blockUntil || now > data.blockUntil)) {
          this.authFailures.delete(key);
        }
      }
    }, 60 * 60 * 1000);
  }

  /**
   * Advanced rate limiting middleware with intelligent threat detection
   */
  public createAdvancedRateLimit() {
    return async (req: Request, res: Response, next: NextFunction) => {
      try {
        const clientIP = this.getClientIP(req);
        const userAgent = req.get('User-Agent') || 'unknown';
        
        // Check if IP is in blocklist
        if (this.blockedIPs.has(clientIP)) {
          await this.logSecurityEvent('blocked_ip_attempt', { 
            ip: clientIP, 
            userAgent,
            path: req.path 
          });
          return res.status(429).json({ 
            error: 'IP blocked due to suspicious activity',
            code: 'IP_BLOCKED'
          });
        }

        // Check rate limiting
        const rateLimited = await this.checkRateLimit(clientIP);
        if (rateLimited) {
          this.markSuspiciousActivity(clientIP);
          
          await this.logSecurityEvent('rate_limit_exceeded', {
            ip: clientIP,
            userAgent,
            path: req.path
          });

          return res.status(429).json({
            error: 'Rate limit exceeded',
            retryAfter: Math.round(this.config.rateLimit.windowMs / 1000),
            code: 'RATE_LIMIT_EXCEEDED'
          });
        }

        // Advanced threat detection
        await this.detectThreats(req, clientIP);
        
        next();
      } catch (error) {
        logger.error('Rate limiting error:', error);
        next(); // Don't block on rate limiter errors
      }
    };
  }

  /**
   * Authentication-specific rate limiting
   */
  public createAuthRateLimit() {
    return async (req: Request, res: Response, next: NextFunction) => {
      try {
        const clientIP = this.getClientIP(req);
        const { email, username } = req.body;
        const userKey = email || username || 'anonymous';

        // Check auth rate limiting
        const authLimited = await this.checkAuthRateLimit(clientIP, userKey);
        if (authLimited) {
          await this.logSecurityEvent('auth_rate_limit_exceeded', { 
            ip: clientIP, 
            userKey
          });
          
          return res.status(429).json({
            error: 'Too many authentication attempts',
            retryAfter: Math.round(this.config.bruteForce.blockDuration / 1000),
            code: 'AUTH_RATE_LIMITED'
          });
        }

        next();
      } catch (error) {
        logger.error('Auth rate limiting error:', error);
        next();
      }
    };
  }

  /**
   * Check general rate limiting
   */
  private async checkRateLimit(clientIP: string): Promise<boolean> {
    const now = Date.now();
    const key = `rate_${clientIP}`;
    const data = this.rateLimitStore.get(key);

    if (!data) {
      this.rateLimitStore.set(key, {
        count: 1,
        resetTime: now + this.config.rateLimit.windowMs
      });
      return false;
    }

    // Check if blocked
    if (data.blocked && data.blockUntil && now < data.blockUntil) {
      return true;
    }

    // Reset if window expired
    if (now > data.resetTime) {
      this.rateLimitStore.set(key, {
        count: 1,
        resetTime: now + this.config.rateLimit.windowMs
      });
      return false;
    }

    // Increment count
    data.count++;
    
    // Check if limit exceeded
    if (data.count > this.config.rateLimit.maxRequests) {
      data.blocked = true;
      data.blockUntil = now + (15 * 60 * 1000); // 15 minutes block
      return true;
    }

    return false;
  }

  /**
   * Check authentication rate limiting
   */
  private async checkAuthRateLimit(clientIP: string, userKey: string): Promise<boolean> {
    const now = Date.now();
    const ipKey = `auth_ip_${clientIP}`;
    const userIPKey = `auth_user_${userKey}_${clientIP}`;

    // Check IP-based auth failures
    const ipData = this.authFailures.get(ipKey);
    if (ipData?.blocked && ipData.blockUntil && now < ipData.blockUntil) {
      return true;
    }

    // Check user+IP based auth failures
    const userIPData = this.authFailures.get(userIPKey);
    if (userIPData?.blocked && userIPData.blockUntil && now < userIPData.blockUntil) {
      return true;
    }

    return false;
  }

  /**
   * Record authentication failure for rate limiting
   */
  public async recordAuthFailure(req: Request, userKey?: string): Promise<void> {
    try {
      const clientIP = this.getClientIP(req);
      const now = Date.now();
      const ipKey = `auth_ip_${clientIP}`;
      const userIPKey = userKey ? `auth_user_${userKey}_${clientIP}` : null;

      // Record IP failure
      const ipData = this.authFailures.get(ipKey) || { count: 0, lastAttempt: 0 };
      ipData.count++;
      ipData.lastAttempt = now;

      if (ipData.count >= this.config.bruteForce.maxWrongAttemptsByIP) {
        ipData.blocked = true;
        ipData.blockUntil = now + (24 * 60 * 60 * 1000); // 24 hours
      }

      this.authFailures.set(ipKey, ipData);

      // Record user+IP failure if user provided
      if (userIPKey) {
        const userIPData = this.authFailures.get(userIPKey) || { count: 0, lastAttempt: 0 };
        userIPData.count++;
        userIPData.lastAttempt = now;

        if (userIPData.count >= this.config.bruteForce.maxConsecutiveFailsByUser) {
          userIPData.blocked = true;
          userIPData.blockUntil = now + this.config.bruteForce.blockDuration;
        }

        this.authFailures.set(userIPKey, userIPData);
      }

      await this.logSecurityEvent('auth_failure_recorded', { 
        ip: clientIP, 
        userKey: userKey || 'unknown' 
      });
    } catch (error) {
      logger.error('Error recording auth failure:', error);
    }
  }

  /**
   * Reset rate limiting on successful authentication
   */
  public async resetAuthRateLimit(req: Request, userKey: string): Promise<void> {
    try {
      const clientIP = this.getClientIP(req);
      const userIPKey = `auth_user_${userKey}_${clientIP}`;

      // Reset consecutive failures for successful login
      this.authFailures.delete(userIPKey);
      
      await this.logSecurityEvent('auth_success_rate_limit_reset', { 
        ip: clientIP, 
        userKey 
      });
    } catch (error) {
      logger.error('Error resetting auth rate limit:', error);
    }
  }

  /**
   * Comprehensive input validation and sanitization middleware
   */
  public createInputValidation() {
    return (req: Request, res: Response, next: NextFunction) => {
      try {
        // Validate and sanitize query parameters
        if (req.query) {
          req.query = this.sanitizeObject(req.query);
        }

        // Validate and sanitize body
        if (req.body) {
          req.body = this.sanitizeObject(req.body);
          
          // Additional validation for critical fields
          if (req.body.email && !validator.isEmail(req.body.email)) {
            return res.status(400).json({ 
              error: 'Invalid email format',
              code: 'INVALID_EMAIL'
            });
          }

          if (req.body.password && !this.isValidPassword(req.body.password)) {
            return res.status(400).json({ 
              error: 'Password does not meet security requirements',
              code: 'WEAK_PASSWORD'
            });
          }

          if (req.body.amount && (!validator.isNumeric(req.body.amount.toString()) || parseFloat(req.body.amount) < 0)) {
            return res.status(400).json({ 
              error: 'Invalid amount value',
              code: 'INVALID_AMOUNT'
            });
          }
        }

        // Validate headers
        this.validateHeaders(req);

        next();
      } catch (error) {
        logger.error('Input validation error:', error);
        res.status(400).json({ 
          error: 'Invalid input data',
          code: 'VALIDATION_ERROR'
        });
      }
    };
  }

  /**
   * Security headers middleware using Helmet with custom configuration
   */
  public createSecurityHeaders() {
    return helmet({
      // Strict Transport Security
      hsts: {
        maxAge: 31536000, // 1 year
        includeSubDomains: true,
        preload: true
      },
      
      // Content Security Policy
      contentSecurityPolicy: {
        directives: {
          defaultSrc: ["'self'"],
          styleSrc: ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com"],
          scriptSrc: ["'self'", "'unsafe-inline'"],
          fontSrc: ["'self'", "https://fonts.gstatic.com"],
          imgSrc: ["'self'", "data:", "https:"],
          connectSrc: ["'self'", "wss:", "ws:"],
          frameSrc: ["'none'"],
          objectSrc: ["'none'"],
          upgradeInsecureRequests: []
        }
      },
      
      // Cross-origin policies
      crossOriginEmbedderPolicy: { policy: "require-corp" },
      crossOriginOpenerPolicy: { policy: "same-origin" },
      crossOriginResourcePolicy: { policy: "cross-origin" },
      
      // Additional security headers
      referrerPolicy: { policy: "no-referrer" },
      
      // Basic security headers
      noSniff: true,
      frameguard: { action: 'deny' },
      xssFilter: true
    });
  }

  /**
   * CSRF protection middleware
   */
  public createCSRFProtection() {
    return (req: Request, res: Response, next: NextFunction) => {
      // Skip CSRF for GET, HEAD, OPTIONS
      if (['GET', 'HEAD', 'OPTIONS'].includes(req.method)) {
        return next();
      }

      const token = req.headers['x-csrf-token'] as string || req.body._csrf;
      const sessionToken = (req as any).session?.csrfToken;

      if (!token || !sessionToken || token !== sessionToken) {
        this.logSecurityEvent('csrf_token_mismatch', {
          ip: this.getClientIP(req),
          path: req.path,
          hasToken: !!token,
          hasSessionToken: !!sessionToken
        });

        return res.status(403).json({
          error: 'Invalid CSRF token',
          code: 'CSRF_TOKEN_INVALID'
        });
      }

      next();
    };
  }

  /**
   * Generate CSRF token for session
   */
  public generateCSRFToken(): string {
    return randomBytes(32).toString('hex');
  }

  /**
   * Advanced threat detection
   */
  private async detectThreats(req: Request, clientIP: string): Promise<void> {
    const threats = [];
    
    // SQL injection patterns
    const sqlPatterns = [
      /(\bUNION\b.*\bSELECT\b)/i,
      /(\bSELECT\b.*\bFROM\b)/i,
      /(\bDROP\b.*\bTABLE\b)/i,
      /(\bINSERT\b.*\bINTO\b)/i,
      /'.*OR.*'.*='.*'/i
    ];

    // XSS patterns
    const xssPatterns = [
      /<script[^>]*>.*?<\/script>/gi,
      /javascript:/gi,
      /on\w+\s*=/gi,
      /<iframe[^>]*>.*?<\/iframe>/gi
    ];

    const requestData = JSON.stringify(req.body) + JSON.stringify(req.query);
    
    // Check for SQL injection
    for (const pattern of sqlPatterns) {
      if (pattern.test(requestData)) {
        threats.push('sql_injection');
        break;
      }
    }

    // Check for XSS
    for (const pattern of xssPatterns) {
      if (pattern.test(requestData)) {
        threats.push('xss_attempt');
        break;
      }
    }

    // Check for suspicious user agents
    const userAgent = req.get('User-Agent') || '';
    const suspiciousUAPatterns = [
      /sqlmap/i,
      /nikto/i,
      /nmap/i,
      /masscan/i,
      /nessus/i
    ];

    for (const pattern of suspiciousUAPatterns) {
      if (pattern.test(userAgent)) {
        threats.push('suspicious_user_agent');
        break;
      }
    }

    // Check for path traversal
    if (/\.\.[\/\\]/.test(req.path) || /\.\.[\/\\]/.test(requestData)) {
      threats.push('path_traversal');
    }

    // Check for command injection
    const cmdPatterns = [
      /[\s;|&`$()]/,
      /\b(cat|ls|pwd|id|whoami|uname)\b/i
    ];

    for (const pattern of cmdPatterns) {
      if (pattern.test(requestData)) {
        threats.push('command_injection');
        break;
      }
    }

    // Process detected threats
    if (threats.length > 0) {
      await this.handleThreatDetection(clientIP, threats, req);
    }
  }

  /**
   * Handle detected threats
   */
  private async handleThreatDetection(clientIP: string, threats: string[], req: Request): Promise<void> {
    this.markSuspiciousActivity(clientIP);
    
    // Log all threats
    await this.logSecurityEvent('threat_detected', {
      ip: clientIP,
      threats,
      path: req.path,
      userAgent: req.get('User-Agent'),
      body: req.body,
      query: req.query
    });

    // Auto-block for severe threats
    const severeThreats = ['sql_injection', 'command_injection', 'path_traversal'];
    if (threats.some(t => severeThreats.includes(t))) {
      this.blockedIPs.add(clientIP);
      
      // Set TTL for blocked IP (auto-unblock after 24 hours)
      setTimeout(() => {
        this.blockedIPs.delete(clientIP);
        logger.info(`Auto-unblocked IP: ${clientIP}`);
      }, 24 * 60 * 60 * 1000);

      await this.logSecurityEvent('ip_auto_blocked', {
        ip: clientIP,
        threats,
        reason: 'severe_threat_detected'
      });
    }
  }

  /**
   * Mark suspicious activity
   */
  private markSuspiciousActivity(clientIP: string): void {
    const currentCount = this.suspiciousActivity.get(clientIP) || 0;
    this.suspiciousActivity.set(clientIP, currentCount + 1);
    
    // Remove from suspicious list after 1 hour
    setTimeout(() => {
      this.suspiciousActivity.delete(clientIP);
    }, 60 * 60 * 1000);
  }

  /**
   * Sanitize object recursively
   */
  private sanitizeObject(obj: any): any {
    if (typeof obj === 'string') {
      return this.sanitizeString(obj);
    }
    
    if (Array.isArray(obj)) {
      return obj.map(item => this.sanitizeObject(item));
    }
    
    if (obj && typeof obj === 'object') {
      const sanitized: any = {};
      for (const [key, value] of Object.entries(obj)) {
        sanitized[this.sanitizeString(key)] = this.sanitizeObject(value);
      }
      return sanitized;
    }
    
    return obj;
  }

  /**
   * Sanitize string input
   */
  private sanitizeString(input: string): string {
    if (typeof input !== 'string') return input;
    
    // Basic XSS protection - strip HTML tags and special characters
    let sanitized = input
      .replace(/<script[^>]*>.*?<\/script>/gi, '')
      .replace(/<[^>]*>/g, '')
      .replace(/javascript:/gi, '')
      .replace(/on\w+\s*=/gi, '');
    
    // Remove null bytes and control characters
    sanitized = sanitized.replace(/\0/g, '').replace(/[\x00-\x1F\x7F]/g, '');
    
    return sanitized;
  }

  /**
   * Validate password strength
   */
  private isValidPassword(password: string): boolean {
    if (typeof password !== 'string' || password.length < 8) return false;
    
    const hasUpperCase = /[A-Z]/.test(password);
    const hasLowerCase = /[a-z]/.test(password);
    const hasNumbers = /\d/.test(password);
    const hasSpecialChar = /[!@#$%^&*(),.?":{}|<>]/.test(password);
    
    return hasUpperCase && hasLowerCase && hasNumbers && hasSpecialChar;
  }

  /**
   * Validate request headers
   */
  private validateHeaders(req: Request): void {
    const userAgent = req.get('User-Agent');
    if (!userAgent || userAgent.length > 500) {
      throw new Error('Invalid User-Agent header');
    }

    const contentType = req.get('Content-Type');
    if (contentType && !['application/json', 'application/x-www-form-urlencoded', 'multipart/form-data'].some(type => contentType.includes(type))) {
      throw new Error('Invalid Content-Type header');
    }
  }

  /**
   * Get client IP address with proxy support
   */
  private getClientIP(req: Request): string {
    return (
      req.ip ||
      req.connection?.remoteAddress ||
      req.socket?.remoteAddress ||
      (req.connection as any)?.socket?.remoteAddress ||
      req.headers['x-forwarded-for']?.toString().split(',')[0] ||
      req.headers['x-real-ip']?.toString() ||
      req.headers['cf-connecting-ip']?.toString() ||
      'unknown'
    ).replace(/^.*:/, ''); // Remove IPv6 prefix
  }

  /**
   * Log security events
   */
  private async logSecurityEvent(event: string, details: any): Promise<void> {
    try {
      await createAuditLog({
        userId: 'system',
        action: `security.${event}`,
        details: {
          ...details,
          timestamp: new Date().toISOString(),
          severity: this.getEventSeverity(event)
        },
        ipAddress: details.ip
      });

      logger.warn(`Security event: ${event}`, details);
    } catch (error) {
      logger.error('Failed to log security event:', error);
    }
  }

  /**
   * Get event severity level
   */
  private getEventSeverity(event: string): string {
    const highSeverity = ['sql_injection', 'command_injection', 'path_traversal', 'ip_auto_blocked'];
    const mediumSeverity = ['xss_attempt', 'suspicious_user_agent', 'csrf_token_mismatch'];
    
    if (highSeverity.some(e => event.includes(e))) return 'high';
    if (mediumSeverity.some(e => event.includes(e))) return 'medium';
    return 'low';
  }

  /**
   * Get security metrics
   */
  public async getSecurityMetrics(): Promise<any> {
    try {
      const metrics = {
        blockedIPs: this.blockedIPs.size,
        suspiciousIPs: this.suspiciousActivity.size,
        rateLimitedIPs: this.rateLimitStore.size,
        authFailures: this.authFailures.size,
        timestamp: new Date().toISOString()
      };

      return metrics;
    } catch (error) {
      logger.error('Error getting security metrics:', error);
      return {};
    }
  }

  /**
   * Manually block/unblock IP
   */
  public blockIP(ip: string, duration?: number): void {
    this.blockedIPs.add(ip);
    
    if (duration) {
      setTimeout(() => {
        this.blockedIPs.delete(ip);
        logger.info(`Auto-unblocked IP: ${ip}`);
      }, duration);
    }
    
    this.logSecurityEvent('ip_manually_blocked', { ip, duration });
  }

  public unblockIP(ip: string): void {
    this.blockedIPs.delete(ip);
    this.logSecurityEvent('ip_manually_unblocked', { ip });
  }

  /**
   * Cleanup resources
   */
  public async cleanup(): Promise<void> {
    try {
      this.rateLimitStore.clear();
      this.authFailures.clear();
      this.suspiciousActivity.clear();
      this.blockedIPs.clear();
      logger.info('API Security Service cleaned up');
    } catch (error) {
      logger.error('Error cleaning up API Security Service:', error);
    }
  }
}

// Export singleton instance
export const createApiSecurityService = (config: SecurityConfig) => {
  return new ApiSecurityService(config);
};

// Default configuration
export const defaultSecurityConfig: SecurityConfig = {
  rateLimit: {
    windowMs: 15 * 60 * 1000, // 15 minutes
    maxRequests: 1000,
    authMaxRequests: 50
  },
  bruteForce: {
    maxWrongAttemptsByIP: 100,
    maxConsecutiveFailsByUser: 10,
    blockDuration: 60 * 60 * 1000 // 1 hour
  },
  validation: {
    maxPayloadSize: '10mb',
    allowedFileTypes: ['jpg', 'jpeg', 'png', 'pdf', 'doc', 'docx'],
    maxFileSize: 10 * 1024 * 1024 // 10MB
  }
}; 