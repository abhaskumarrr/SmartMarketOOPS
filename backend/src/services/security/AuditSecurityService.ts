/**
 * Enhanced Audit Security Service - Phase 4.3 Security Hardening
 * Comprehensive security event monitoring, compliance logging, and real-time alerts
 */

import { Request, Response, NextFunction } from 'express';
import { createLogger } from '../../utils/logger';
import { createAuditLog, getUserAuditLogs } from '../../utils/auditLog';
import prisma from '../../utils/prismaClient';
import { randomBytes, createHash } from 'crypto';

const logger = createLogger('AuditSecurityService');

export enum SecurityEventType {
  AUTHENTICATION = 'authentication',
  AUTHORIZATION = 'authorization',
  DATA_ACCESS = 'data_access',
  DATA_MODIFICATION = 'data_modification',
  SECURITY_VIOLATION = 'security_violation',
  SYSTEM_ACCESS = 'system_access',
  CONFIGURATION_CHANGE = 'configuration_change',
  API_ACCESS = 'api_access',
  FINANCIAL_TRANSACTION = 'financial_transaction',
  ADMIN_ACTION = 'admin_action'
}

export enum SecurityEventSeverity {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}

export interface SecurityEvent {
  id?: string;
  userId: string;
  eventType: SecurityEventType;
  action: string;
  severity: SecurityEventSeverity;
  ipAddress: string;
  userAgent?: string;
  details: any;
  timestamp: Date;
  sessionId?: string;
  resource?: string;
  outcome: 'success' | 'failure' | 'blocked';
  riskScore?: number;
}

export interface AuditConfig {
  enableRealTimeAlerts: boolean;
  retentionDays: number;
  criticalAlertThreshold: number;
  highRiskThreshold: number;
  complianceMode: boolean;
  encryptSensitiveData: boolean;
}

export interface SecurityAlert {
  id: string;
  eventId: string;
  severity: SecurityEventSeverity;
  type: string;
  message: string;
  timestamp: Date;
  acknowledged: boolean;
  resolvedAt?: Date;
}

export class AuditSecurityService {
  private config: AuditConfig;
  private activeAlerts = new Map<string, SecurityAlert>();
  private eventBuffer: SecurityEvent[] = [];
  private riskPatterns = new Map<string, number>();
  private suspiciousActivities = new Map<string, any[]>();

  constructor(config: AuditConfig) {
    this.config = config;
    this.initializeService();
    logger.info('Enhanced Audit Security Service initialized');
  }

  /**
   * Initialize service with background tasks
   */
  private initializeService(): void {
    // Flush event buffer every 30 seconds
    setInterval(() => {
      this.flushEventBuffer();
    }, 30000);

    // Clean up old data based on retention policy
    setInterval(() => {
      this.cleanupOldData();
    }, 24 * 60 * 60 * 1000); // Daily

    // Process risk patterns every 5 minutes
    setInterval(() => {
      this.processRiskPatterns();
    }, 5 * 60 * 1000);
  }

  /**
   * Log a security event with comprehensive details
   */
  public async logSecurityEvent(event: Partial<SecurityEvent>, req?: Request): Promise<SecurityEvent> {
    try {
      const securityEvent: SecurityEvent = {
        id: randomBytes(16).toString('hex'),
        userId: event.userId || 'anonymous',
        eventType: event.eventType || SecurityEventType.SYSTEM_ACCESS,
        action: event.action || 'unknown_action',
        severity: event.severity || SecurityEventSeverity.LOW,
        ipAddress: event.ipAddress || this.getClientIP(req),
        userAgent: event.userAgent || req?.get('User-Agent'),
        details: this.sanitizeDetails(event.details || {}),
        timestamp: new Date(),
        sessionId: event.sessionId || this.getSessionId(req),
        resource: event.resource,
        outcome: event.outcome || 'success',
        riskScore: event.riskScore || this.calculateRiskScore(event)
      };

      // Add to buffer for batch processing
      this.eventBuffer.push(securityEvent);

      // Immediate processing for critical events
      if (securityEvent.severity === SecurityEventSeverity.CRITICAL) {
        await this.processImmediately(securityEvent);
      }

      // Check for real-time alerts
      if (this.config.enableRealTimeAlerts) {
        await this.checkForAlerts(securityEvent);
      }

      // Update risk patterns
      this.updateRiskPatterns(securityEvent);

      // Log to application logger
      logger.info(`Security event logged: ${securityEvent.action}`, {
        eventId: securityEvent.id,
        userId: securityEvent.userId,
        severity: securityEvent.severity,
        outcome: securityEvent.outcome
      });

      return securityEvent;
    } catch (error) {
      logger.error('Failed to log security event:', error);
      throw error;
    }
  }

  /**
   * Enhanced authentication event logging
   */
  public async logAuthenticationEvent(
    action: 'login_attempt' | 'login_success' | 'login_failure' | 'logout' | 'session_expired' | 'mfa_challenge' | 'password_change',
    userId: string,
    outcome: 'success' | 'failure' | 'blocked',
    req: Request,
    details: any = {}
  ): Promise<SecurityEvent> {
    const severity = this.getAuthSeverity(action, outcome);
    
    return await this.logSecurityEvent({
      userId,
      eventType: SecurityEventType.AUTHENTICATION,
      action: `auth.${action}`,
      severity,
      outcome,
      details: {
        ...details,
        method: details.method || 'password',
        twoFactorUsed: details.twoFactorUsed || false,
        deviceFingerprint: this.generateDeviceFingerprint(req)
      }
    }, req);
  }

  /**
   * Data access event logging
   */
  public async logDataAccessEvent(
    userId: string,
    resource: string,
    action: 'read' | 'write' | 'delete' | 'export' | 'import',
    outcome: 'success' | 'failure' | 'blocked',
    req: Request,
    details: any = {}
  ): Promise<SecurityEvent> {
    const severity = this.getDataAccessSeverity(action, resource, details);
    
    return await this.logSecurityEvent({
      userId,
      eventType: SecurityEventType.DATA_ACCESS,
      action: `data.${action}`,
      severity,
      outcome,
      resource,
      details: {
        ...details,
        resourceType: this.classifyResource(resource),
        dataVolume: details.recordCount || details.dataSize || 1,
        sensitivityLevel: this.assessDataSensitivity(resource, details)
      }
    }, req);
  }

  /**
   * Financial transaction event logging
   */
  public async logFinancialEvent(
    userId: string,
    action: string,
    amount: number,
    currency: string,
    outcome: 'success' | 'failure' | 'blocked',
    req: Request,
    details: any = {}
  ): Promise<SecurityEvent> {
    return await this.logSecurityEvent({
      userId,
      eventType: SecurityEventType.FINANCIAL_TRANSACTION,
      action: `financial.${action}`,
      severity: this.getFinancialSeverity(action, amount, outcome),
      outcome,
      resource: `transaction_${details.transactionId || 'unknown'}`,
      details: {
        ...details,
        amount,
        currency,
        riskIndicators: this.analyzeFinancialRisk(amount, details),
        complianceFlags: this.checkComplianceRequirements(amount, details)
      }
    }, req);
  }

  /**
   * Administrative action logging
   */
  public async logAdminEvent(
    userId: string,
    action: string,
    targetResource: string,
    outcome: 'success' | 'failure' | 'blocked',
    req: Request,
    details: any = {}
  ): Promise<SecurityEvent> {
    return await this.logSecurityEvent({
      userId,
      eventType: SecurityEventType.ADMIN_ACTION,
      action: `admin.${action}`,
      severity: SecurityEventSeverity.HIGH, // Admin actions are always high severity
      outcome,
      resource: targetResource,
      details: {
        ...details,
        targetUser: details.targetUserId,
        permissionsChanged: details.permissionsChanged,
        configurationChanges: details.configurationChanges
      }
    }, req);
  }

  /**
   * Security violation logging
   */
  public async logSecurityViolation(
    userId: string,
    violationType: string,
    severity: SecurityEventSeverity,
    req: Request,
    details: any = {}
  ): Promise<SecurityEvent> {
    return await this.logSecurityEvent({
      userId,
      eventType: SecurityEventType.SECURITY_VIOLATION,
      action: `violation.${violationType}`,
      severity,
      outcome: 'blocked',
      details: {
        ...details,
        threatLevel: this.assessThreatLevel(violationType, details),
        mitigationActions: details.mitigationActions || [],
        automaticResponse: details.automaticResponse || false
      }
    }, req);
  }

  /**
   * Create audit trail middleware
   */
  public createAuditMiddleware() {
    return async (req: Request, res: Response, next: NextFunction) => {
      const startTime = Date.now();
      const originalSend = res.send;
      
      // Capture response
      res.send = function(data) {
        const responseTime = Date.now() - startTime;
        
        // Log API access
        setImmediate(async () => {
          try {
            const userId = (req as any).user?.id || 'anonymous';
            const action = `${req.method.toLowerCase()}_${req.path.replace(/[^a-zA-Z0-9]/g, '_')}`;
            const outcome = res.statusCode < 400 ? 'success' : 'failure';
            const severity = res.statusCode >= 500 ? SecurityEventSeverity.HIGH : 
                           res.statusCode >= 400 ? SecurityEventSeverity.MEDIUM : 
                           SecurityEventSeverity.LOW;

            await this.logSecurityEvent({
              userId,
              eventType: SecurityEventType.API_ACCESS,
              action,
              severity,
              outcome,
              resource: req.path,
              details: {
                method: req.method,
                statusCode: res.statusCode,
                responseTime,
                bodySize: data ? Buffer.byteLength(data) : 0,
                queryParams: Object.keys(req.query || {}).length,
                hasBody: !!req.body && Object.keys(req.body).length > 0
              }
            }, req);
          } catch (error) {
            logger.error('Failed to log API access event:', error);
          }
        });

        return originalSend.call(this, data);
      };

      next();
    };
  }

  /**
   * Get security events with filtering and pagination
   */
  public async getSecurityEvents(filters: {
    userId?: string;
    eventType?: SecurityEventType;
    severity?: SecurityEventSeverity;
    startDate?: Date;
    endDate?: Date;
    outcome?: string;
    limit?: number;
    offset?: number;
  } = {}): Promise<{ events: SecurityEvent[]; total: number }> {
    try {
      const whereClause: any = {};
      
      if (filters.userId) whereClause.userId = filters.userId;
      if (filters.eventType) whereClause.action = { startsWith: filters.eventType };
      if (filters.startDate || filters.endDate) {
        whereClause.timestamp = {};
        if (filters.startDate) whereClause.timestamp.gte = filters.startDate;
        if (filters.endDate) whereClause.timestamp.lte = filters.endDate;
      }

      const auditLogs = await prisma.auditLog.findMany({
        where: whereClause,
        orderBy: { timestamp: 'desc' },
        skip: filters.offset || 0,
        take: filters.limit || 100
      });

      const total = await prisma.auditLog.count({ where: whereClause });

      const events = auditLogs.map(log => this.mapToSecurityEvent(log));

      return { events, total };
    } catch (error) {
      logger.error('Failed to get security events:', error);
      return { events: [], total: 0 };
    }
  }

  /**
   * Generate compliance report
   */
  public async generateComplianceReport(
    startDate: Date,
    endDate: Date,
    options: {
      includeFailedAttempts?: boolean;
      includeDataAccess?: boolean;
      includeFinancialTransactions?: boolean;
      format?: 'json' | 'csv';
    } = {}
  ): Promise<any> {
    try {
      const { events } = await this.getSecurityEvents({
        startDate,
        endDate,
        limit: 10000
      });

      const report = {
        reportId: randomBytes(16).toString('hex'),
        generatedAt: new Date(),
        period: { startDate, endDate },
        summary: {
          totalEvents: events.length,
          authenticationEvents: events.filter(e => e.eventType === SecurityEventType.AUTHENTICATION).length,
          dataAccessEvents: events.filter(e => e.eventType === SecurityEventType.DATA_ACCESS).length,
          securityViolations: events.filter(e => e.eventType === SecurityEventType.SECURITY_VIOLATION).length,
          financialTransactions: events.filter(e => e.eventType === SecurityEventType.FINANCIAL_TRANSACTION).length,
          failedAttempts: events.filter(e => e.outcome === 'failure').length,
          criticalEvents: events.filter(e => e.severity === SecurityEventSeverity.CRITICAL).length
        },
        details: {
          eventsByType: this.groupEventsByType(events),
          eventsBySeverity: this.groupEventsBySeverity(events),
          topUsers: this.getTopUsersByActivity(events),
          riskPatterns: Array.from(this.riskPatterns.entries()),
          securityAlerts: Array.from(this.activeAlerts.values())
        },
        compliance: {
          dataRetentionCompliance: this.checkDataRetentionCompliance(),
          accessControlCompliance: this.checkAccessControlCompliance(events),
          auditTrailCompleteness: this.checkAuditTrailCompleteness(events)
        }
      };

      if (options.format === 'csv') {
        return this.convertToCSV(report, events);
      }

      return report;
    } catch (error) {
      logger.error('Failed to generate compliance report:', error);
      throw error;
    }
  }

  /**
   * Get security alerts
   */
  public getActiveAlerts(): SecurityAlert[] {
    return Array.from(this.activeAlerts.values());
  }

  /**
   * Acknowledge security alert
   */
  public acknowledgeAlert(alertId: string, userId: string): boolean {
    const alert = this.activeAlerts.get(alertId);
    if (alert) {
      alert.acknowledged = true;
      alert.resolvedAt = new Date();
      
      this.logSecurityEvent({
        userId,
        eventType: SecurityEventType.ADMIN_ACTION,
        action: 'alert.acknowledged',
        severity: SecurityEventSeverity.MEDIUM,
        outcome: 'success',
        details: { alertId, alertType: alert.type }
      });
      
      return true;
    }
    return false;
  }

  /**
   * Get security metrics for monitoring
   */
  public getSecurityMetrics(): any {
    return {
      activeAlerts: this.activeAlerts.size,
      eventBufferSize: this.eventBuffer.length,
      riskPatterns: this.riskPatterns.size,
      suspiciousActivities: this.suspiciousActivities.size,
      lastProcessed: new Date(),
      systemHealth: {
        auditingEnabled: true,
        complianceMode: this.config.complianceMode,
        retentionDays: this.config.retentionDays
      }
    };
  }

  // Private helper methods

  private async flushEventBuffer(): Promise<void> {
    if (this.eventBuffer.length === 0) return;

    try {
      const events = [...this.eventBuffer];
      this.eventBuffer = [];

      // Batch insert into database
      await Promise.all(events.map(event => this.persistEvent(event)));
      
      logger.debug(`Flushed ${events.length} security events to database`);
    } catch (error) {
      logger.error('Failed to flush event buffer:', error);
    }
  }

  private async persistEvent(event: SecurityEvent): Promise<void> {
    try {
      await createAuditLog({
        userId: event.userId,
        action: event.action,
        details: {
          ...event.details,
          eventType: event.eventType,
          severity: event.severity,
          outcome: event.outcome,
          riskScore: event.riskScore,
          sessionId: event.sessionId,
          resource: event.resource
        },
        ipAddress: event.ipAddress,
        userAgent: event.userAgent
      });
    } catch (error) {
      logger.error('Failed to persist security event:', error);
    }
  }

  private calculateRiskScore(event: Partial<SecurityEvent>): number {
    let score = 0;

    // Base score by event type
    switch (event.eventType) {
      case SecurityEventType.AUTHENTICATION:
        score = event.outcome === 'failure' ? 30 : 10;
        break;
      case SecurityEventType.DATA_ACCESS:
        score = 20;
        break;
      case SecurityEventType.FINANCIAL_TRANSACTION:
        score = 40;
        break;
      case SecurityEventType.SECURITY_VIOLATION:
        score = 80;
        break;
      default:
        score = 10;
    }

    // Severity multiplier
    switch (event.severity) {
      case SecurityEventSeverity.CRITICAL:
        score *= 3;
        break;
      case SecurityEventSeverity.HIGH:
        score *= 2;
        break;
      case SecurityEventSeverity.MEDIUM:
        score *= 1.5;
        break;
    }

    return Math.min(score, 100);
  }

  private getClientIP(req?: Request): string {
    if (!req) return 'unknown';
    
    return (
      req.ip ||
      req.headers['x-forwarded-for']?.toString().split(',')[0] ||
      req.headers['x-real-ip']?.toString() ||
      req.connection?.remoteAddress ||
      'unknown'
    );
  }

  private getSessionId(req?: Request): string | undefined {
    return (req as any)?.sessionID || (req as any)?.session?.id;
  }

  private sanitizeDetails(details: any): any {
    const sanitized = { ...details };
    
    // Remove sensitive fields
    const sensitiveFields = ['password', 'token', 'secret', 'key', 'apiKey', 'privateKey'];
    sensitiveFields.forEach(field => {
      if (sanitized[field]) {
        sanitized[field] = '[REDACTED]';
      }
    });

    return sanitized;
  }

  private generateDeviceFingerprint(req: Request): string {
    const userAgent = req.get('User-Agent') || '';
    const acceptLanguage = req.get('Accept-Language') || '';
    const acceptEncoding = req.get('Accept-Encoding') || '';
    
    const fingerprint = createHash('sha256')
      .update(userAgent + acceptLanguage + acceptEncoding)
      .digest('hex')
      .substring(0, 16);
    
    return fingerprint;
  }

  private getAuthSeverity(action: string, outcome: string): SecurityEventSeverity {
    if (outcome === 'failure' || outcome === 'blocked') {
      return action.includes('login') ? SecurityEventSeverity.MEDIUM : SecurityEventSeverity.HIGH;
    }
    return SecurityEventSeverity.LOW;
  }

  private getDataAccessSeverity(action: string, resource: string, details: any): SecurityEventSeverity {
    if (action === 'delete' || action === 'export') return SecurityEventSeverity.HIGH;
    if (this.assessDataSensitivity(resource, details) === 'high') return SecurityEventSeverity.MEDIUM;
    return SecurityEventSeverity.LOW;
  }

  private getFinancialSeverity(action: string, amount: number, outcome: string): SecurityEventSeverity {
    if (outcome === 'blocked') return SecurityEventSeverity.HIGH;
    if (amount > 10000) return SecurityEventSeverity.HIGH;
    if (amount > 1000) return SecurityEventSeverity.MEDIUM;
    return SecurityEventSeverity.LOW;
  }

  private classifyResource(resource: string): string {
    if (resource.includes('user')) return 'user_data';
    if (resource.includes('financial') || resource.includes('transaction')) return 'financial_data';
    if (resource.includes('trading') || resource.includes('order')) return 'trading_data';
    if (resource.includes('api')) return 'api_endpoint';
    return 'system_resource';
  }

  private assessDataSensitivity(resource: string, details: any): string {
    const highSensitivity = ['password', 'apikey', 'secret', 'financial', 'trading', 'personal'];
    if (highSensitivity.some(term => resource.toLowerCase().includes(term))) return 'high';
    if (details.recordCount > 100) return 'medium';
    return 'low';
  }

  private analyzeFinancialRisk(amount: number, details: any): string[] {
    const risks = [];
    if (amount > 50000) risks.push('large_amount');
    if (details.fromDifferentCountry) risks.push('cross_border');
    if (details.unusualTime) risks.push('unusual_timing');
    if (details.newPaymentMethod) risks.push('new_payment_method');
    return risks;
  }

  private checkComplianceRequirements(amount: number, details: any): string[] {
    const flags = [];
    if (amount > 10000) flags.push('aml_reporting_required');
    if (details.internationalTransfer) flags.push('international_compliance');
    return flags;
  }

  private assessThreatLevel(violationType: string, details: any): string {
    const highThreatTypes = ['sql_injection', 'command_injection', 'path_traversal'];
    if (highThreatTypes.includes(violationType)) return 'high';
    if (details.repeatedAttempts > 5) return 'medium';
    return 'low';
  }

  private async processImmediately(event: SecurityEvent): Promise<void> {
    await this.persistEvent(event);
    
    if (event.severity === SecurityEventSeverity.CRITICAL) {
      await this.createAlert(event, 'Critical security event detected');
    }
  }

  private async checkForAlerts(event: SecurityEvent): Promise<void> {
    // Pattern-based alert detection
    if (event.outcome === 'failure' && event.eventType === SecurityEventType.AUTHENTICATION) {
      const recentFailures = this.countRecentFailures(event.userId, event.ipAddress);
      if (recentFailures >= 5) {
        await this.createAlert(event, `Multiple authentication failures: ${recentFailures} attempts`);
      }
    }

    if (event.riskScore && event.riskScore > this.config.highRiskThreshold) {
      await this.createAlert(event, `High risk activity detected: score ${event.riskScore}`);
    }
  }

  private async createAlert(event: SecurityEvent, message: string): Promise<void> {
    const alert: SecurityAlert = {
      id: randomBytes(16).toString('hex'),
      eventId: event.id!,
      severity: event.severity,
      type: event.eventType,
      message,
      timestamp: new Date(),
      acknowledged: false
    };

    this.activeAlerts.set(alert.id, alert);
    
    logger.warn(`Security alert created: ${message}`, {
      alertId: alert.id,
      eventId: event.id,
      severity: event.severity
    });
  }

  private updateRiskPatterns(event: SecurityEvent): void {
    const patternKey = `${event.eventType}_${event.outcome}_${event.ipAddress}`;
    const currentCount = this.riskPatterns.get(patternKey) || 0;
    this.riskPatterns.set(patternKey, currentCount + 1);
  }

  private async processRiskPatterns(): Promise<void> {
    // Analyze patterns for anomalies
    for (const [pattern, count] of this.riskPatterns.entries()) {
      if (count > 10) { // Threshold for suspicious pattern
        logger.warn(`Suspicious pattern detected: ${pattern} (${count} occurrences)`);
      }
    }
  }

  private countRecentFailures(userId: string, ipAddress: string): number {
    // Implementation would check recent events in buffer/database
    return 0; // Simplified for now
  }

  private async cleanupOldData(): Promise<void> {
    try {
      const cutoffDate = new Date();
      cutoffDate.setDate(cutoffDate.getDate() - this.config.retentionDays);

      await prisma.auditLog.deleteMany({
        where: {
          timestamp: {
            lt: cutoffDate
          }
        }
      });

      logger.info(`Cleaned up audit logs older than ${this.config.retentionDays} days`);
    } catch (error) {
      logger.error('Failed to cleanup old audit data:', error);
    }
  }

  private mapToSecurityEvent(auditLog: any): SecurityEvent {
    return {
      id: auditLog.id,
      userId: auditLog.userId,
      eventType: auditLog.details?.eventType || SecurityEventType.SYSTEM_ACCESS,
      action: auditLog.action,
      severity: auditLog.details?.severity || SecurityEventSeverity.LOW,
      ipAddress: auditLog.ipAddress || 'unknown',
      userAgent: auditLog.userAgent,
      details: auditLog.details || {},
      timestamp: auditLog.timestamp,
      sessionId: auditLog.details?.sessionId,
      resource: auditLog.details?.resource,
      outcome: auditLog.details?.outcome || 'success',
      riskScore: auditLog.details?.riskScore
    };
  }

  private groupEventsByType(events: SecurityEvent[]): any {
    return events.reduce((acc, event) => {
      acc[event.eventType] = (acc[event.eventType] || 0) + 1;
      return acc;
    }, {} as any);
  }

  private groupEventsBySeverity(events: SecurityEvent[]): any {
    return events.reduce((acc, event) => {
      acc[event.severity] = (acc[event.severity] || 0) + 1;
      return acc;
    }, {} as any);
  }

  private getTopUsersByActivity(events: SecurityEvent[]): any[] {
    const userActivity = events.reduce((acc, event) => {
      acc[event.userId] = (acc[event.userId] || 0) + 1;
      return acc;
    }, {} as any);

    return Object.entries(userActivity)
      .sort(([,a], [,b]) => (b as number) - (a as number))
      .slice(0, 10)
      .map(([userId, count]) => ({ userId, eventCount: count }));
  }

  private checkDataRetentionCompliance(): boolean {
    return this.config.retentionDays > 0;
  }

  private checkAccessControlCompliance(events: SecurityEvent[]): any {
    const authEvents = events.filter(e => e.eventType === SecurityEventType.AUTHENTICATION);
    const totalAuth = authEvents.length;
    const failedAuth = authEvents.filter(e => e.outcome === 'failure').length;
    
    return {
      totalAuthenticationEvents: totalAuth,
      failedAuthenticationEvents: failedAuth,
      failureRate: totalAuth > 0 ? (failedAuth / totalAuth) * 100 : 0,
      compliant: totalAuth > 0 && (failedAuth / totalAuth) < 0.1 // Less than 10% failure rate
    };
  }

  private checkAuditTrailCompleteness(events: SecurityEvent[]): any {
    const eventTypes = new Set(events.map(e => e.eventType));
    const requiredTypes = [
      SecurityEventType.AUTHENTICATION,
      SecurityEventType.DATA_ACCESS,
      SecurityEventType.API_ACCESS
    ];
    
    const missingTypes = requiredTypes.filter(type => !eventTypes.has(type));
    
    return {
      hasAllRequiredTypes: missingTypes.length === 0,
      missingEventTypes: missingTypes,
      completeness: ((requiredTypes.length - missingTypes.length) / requiredTypes.length) * 100
    };
  }

  private convertToCSV(report: any, events: SecurityEvent[]): string {
    const headers = [
      'Timestamp', 'User ID', 'Event Type', 'Action', 'Severity', 
      'Outcome', 'IP Address', 'Resource', 'Risk Score'
    ];
    
    const rows = events.map(event => [
      event.timestamp.toISOString(),
      event.userId,
      event.eventType,
      event.action,
      event.severity,
      event.outcome,
      event.ipAddress,
      event.resource || '',
      event.riskScore || ''
    ]);
    
    return [headers.join(','), ...rows.map(row => row.join(','))].join('\n');
  }
}

// Export factory function
export const createAuditSecurityService = (config: AuditConfig) => {
  return new AuditSecurityService(config);
};

// Default configuration
export const defaultAuditConfig: AuditConfig = {
  enableRealTimeAlerts: true,
  retentionDays: 90,
  criticalAlertThreshold: 5,
  highRiskThreshold: 70,
  complianceMode: process.env.NODE_ENV === 'production',
  encryptSensitiveData: true
}; 