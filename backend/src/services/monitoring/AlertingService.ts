/**
 * Phase 4: Production Readiness - Comprehensive Alerting and Error Tracking Service
 * Real-time monitoring, alerting, and error tracking for production trading platform
 */

import { EventEmitter } from 'events';
import prisma from '../../config/prisma-readonly';
import Redis from 'ioredis';
import { logger } from '../../utils/logger';
import nodemailer from 'nodemailer';

export interface AlertRule {
  id: string;
  name: string;
  condition: string;
  threshold: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  enabled: boolean;
  cooldownMinutes: number;
  channels: AlertChannel[];
  description: string;
}

export interface AlertChannel {
  type: 'email' | 'webhook' | 'sms' | 'slack' | 'discord';
  config: Record<string, any>;
  enabled: boolean;
}

export interface Alert {
  id: string;
  ruleId: string;
  message: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  timestamp: Date;
  resolved: boolean;
  resolvedAt?: Date;
  metadata: Record<string, any>;
  affectedUsers?: string[];
}

export interface ErrorEvent {
  id: string;
  type: 'api_error' | 'database_error' | 'trading_error' | 'system_error' | 'security_error';
  message: string;
  stack?: string;
  context: Record<string, any>;
  severity: 'low' | 'medium' | 'high' | 'critical';
  timestamp: Date;
  userId?: string;
  endpoint?: string;
  resolved: boolean;
}

export interface SystemHealthCheck {
  name: string;
  status: 'healthy' | 'degraded' | 'unhealthy';
  responseTime: number;
  lastCheck: Date;
  details?: Record<string, any>;
}

export class AlertingService extends EventEmitter {
  private redis: Redis;
  private emailTransporter: nodemailer.Transporter;
  private alertRules: Map<string, AlertRule> = new Map();
  private activeAlerts: Map<string, Alert> = new Map();
  private errorHistory: ErrorEvent[] = [];
  private healthCheckInterval: NodeJS.Timeout | null = null;

  constructor() {
    super();
    
    this.redis = new Redis({
      host: process.env.REDIS_HOST || 'localhost',
      port: parseInt(process.env.REDIS_PORT || '6379'),
      retryDelayOnFailover: 100,
      enableReadyCheck: false,
      maxRetriesPerRequest: null,
    });

    // Setup email transporter
    this.emailTransporter = nodemailer.createTransporter({
      host: process.env.SMTP_HOST,
      port: parseInt(process.env.SMTP_PORT || '587'),
      secure: false,
      auth: {
        user: process.env.SMTP_USER,
        pass: process.env.SMTP_PASS,
      },
    });

    this.initializeDefaultAlertRules();
    this.startHealthChecks();

    logger.info('Alerting service initialized');
  }

  /**
   * Initialize default alert rules for production monitoring
   */
  private initializeDefaultAlertRules(): void {
    const defaultRules: AlertRule[] = [
      {
        id: 'high_error_rate',
        name: 'High Error Rate',
        condition: 'error_rate > threshold',
        threshold: 0.05, // 5%
        severity: 'high',
        enabled: true,
        cooldownMinutes: 15,
        channels: [
          { type: 'email', config: { recipients: ['ops@trading.com'] }, enabled: true },
          { type: 'slack', config: { webhook: process.env.SLACK_WEBHOOK }, enabled: true },
        ],
        description: 'API error rate exceeds acceptable threshold',
      },
      {
        id: 'database_connection_failure',
        name: 'Database Connection Failure',
        condition: 'db_connections < threshold',
        threshold: 1,
        severity: 'critical',
        enabled: true,
        cooldownMinutes: 5,
        channels: [
          { type: 'email', config: { recipients: ['ops@trading.com', 'dev@trading.com'] }, enabled: true },
          { type: 'webhook', config: { url: process.env.PAGER_DUTY_WEBHOOK }, enabled: true },
        ],
        description: 'Database connection pool exhausted or unavailable',
      },
      {
        id: 'excessive_drawdown',
        name: 'Excessive Trading Drawdown',
        condition: 'max_drawdown > threshold',
        threshold: 0.2, // 20%
        severity: 'high',
        enabled: true,
        cooldownMinutes: 30,
        channels: [
          { type: 'email', config: { recipients: ['risk@trading.com'] }, enabled: true },
        ],
        description: 'User portfolio drawdown exceeds risk limits',
      },
      {
        id: 'api_response_time',
        name: 'High API Response Time',
        condition: 'avg_response_time > threshold',
        threshold: 2000, // 2 seconds
        severity: 'medium',
        enabled: true,
        cooldownMinutes: 10,
        channels: [
          { type: 'email', config: { recipients: ['ops@trading.com'] }, enabled: true },
        ],
        description: 'API response times are degraded',
      },
      {
        id: 'failed_trades',
        name: 'High Trade Failure Rate',
        condition: 'trade_failure_rate > threshold',
        threshold: 0.1, // 10%
        severity: 'high',
        enabled: true,
        cooldownMinutes: 20,
        channels: [
          { type: 'email', config: { recipients: ['trading@trading.com'] }, enabled: true },
        ],
        description: 'High percentage of trade executions are failing',
      },
      {
        id: 'memory_usage',
        name: 'High Memory Usage',
        condition: 'memory_usage > threshold',
        threshold: 0.85, // 85%
        severity: 'medium',
        enabled: true,
        cooldownMinutes: 15,
        channels: [
          { type: 'email', config: { recipients: ['ops@trading.com'] }, enabled: true },
        ],
        description: 'System memory usage is critically high',
      },
      {
        id: 'websocket_connections',
        name: 'WebSocket Connection Issues',
        condition: 'websocket_errors > threshold',
        threshold: 100,
        severity: 'medium',
        enabled: true,
        cooldownMinutes: 10,
        channels: [
          { type: 'email', config: { recipients: ['ops@trading.com'] }, enabled: true },
        ],
        description: 'High number of WebSocket connection errors',
      },
    ];

    defaultRules.forEach(rule => {
      this.alertRules.set(rule.id, rule);
    });
  }

  /**
   * Check metrics against alert rules and trigger alerts if needed
   */
  public async checkAlerts(metrics: Record<string, number>): Promise<void> {
    try {
      for (const [ruleId, rule] of this.alertRules.entries()) {
        if (!rule.enabled) continue;

        const shouldAlert = await this.evaluateRule(rule, metrics);
        
        if (shouldAlert && !this.isInCooldown(ruleId)) {
          await this.triggerAlert(rule, metrics);
        }
      }
    } catch (error) {
      logger.error('Error checking alerts:', error);
    }
  }

  /**
   * Track and categorize errors with severity levels
   */
  public async trackError(error: Partial<ErrorEvent>): Promise<void> {
    try {
      const errorEvent: ErrorEvent = {
        id: this.generateId(),
        type: error.type || 'system_error',
        message: error.message || 'Unknown error',
        stack: error.stack,
        context: error.context || {},
        severity: error.severity || 'medium',
        timestamp: new Date(),
        userId: error.userId,
        endpoint: error.endpoint,
        resolved: false,
      };

      // Store in memory (could be replaced with database storage)
      this.errorHistory.push(errorEvent);
      
      // Keep only last 10000 errors
      if (this.errorHistory.length > 10000) {
        this.errorHistory = this.errorHistory.slice(-10000);
      }

      // Store in Redis for real-time access
      await this.redis.lpush('errors:history', JSON.stringify(errorEvent));
      await this.redis.ltrim('errors:history', 0, 9999);

      // Update error rate metrics
      await this.updateErrorMetrics(errorEvent);

      // Check if error should trigger an alert
      if (errorEvent.severity === 'critical' || errorEvent.severity === 'high') {
        await this.checkErrorAlerts(errorEvent);
      }

      this.emit('error:tracked', errorEvent);
      logger.error(`Error tracked: ${errorEvent.type} - ${errorEvent.message}`, errorEvent.context);
    } catch (trackingError) {
      logger.error('Failed to track error:', trackingError);
    }
  }

  /**
   * Perform comprehensive system health checks
   */
  public async performHealthChecks(): Promise<SystemHealthCheck[]> {
    const checks: SystemHealthCheck[] = [];

    try {
      // Database health check
      const dbCheck = await this.checkDatabaseHealth();
      checks.push(dbCheck);

      // Redis health check
      const redisCheck = await this.checkRedisHealth();
      checks.push(redisCheck);

      // External API health checks
      const externalChecks = await this.checkExternalAPIs();
      checks.push(...externalChecks);

      // System resource checks
      const resourceChecks = await this.checkSystemResources();
      checks.push(...resourceChecks);

      // Trading system checks
      const tradingChecks = await this.checkTradingSystem();
      checks.push(...tradingChecks);

      // Store health check results
      await this.redis.setex('health:checks', 300, JSON.stringify(checks));

      // Check for unhealthy services and alert
      const unhealthyServices = checks.filter(check => check.status === 'unhealthy');
      if (unhealthyServices.length > 0) {
        await this.alertUnhealthyServices(unhealthyServices);
      }

      return checks;
    } catch (error) {
      logger.error('Error performing health checks:', error);
      return [];
    }
  }

  /**
   * Get current system status dashboard
   */
  public async getSystemStatus(): Promise<{
    overall: 'healthy' | 'degraded' | 'unhealthy';
    services: SystemHealthCheck[];
    activeAlerts: Alert[];
    recentErrors: ErrorEvent[];
    uptime: number;
  }> {
    try {
      const services = await this.performHealthChecks();
      const activeAlerts = Array.from(this.activeAlerts.values());
      const recentErrors = this.errorHistory.slice(-50);

      // Determine overall status
      const unhealthyCount = services.filter(s => s.status === 'unhealthy').length;
      const degradedCount = services.filter(s => s.status === 'degraded').length;
      
      let overall: 'healthy' | 'degraded' | 'unhealthy';
      if (unhealthyCount > 0) {
        overall = 'unhealthy';
      } else if (degradedCount > 0 || activeAlerts.length > 0) {
        overall = 'degraded';
      } else {
        overall = 'healthy';
      }

      return {
        overall,
        services,
        activeAlerts,
        recentErrors,
        uptime: process.uptime(),
      };
    } catch (error) {
      logger.error('Error getting system status:', error);
      throw error;
    }
  }

  /**
   * Resolve an active alert
   */
  public async resolveAlert(alertId: string, resolvedBy: string): Promise<void> {
    try {
      const alert = this.activeAlerts.get(alertId);
      if (!alert) {
        throw new Error(`Alert ${alertId} not found`);
      }

      alert.resolved = true;
      alert.resolvedAt = new Date();
      alert.metadata.resolvedBy = resolvedBy;

      // Remove from active alerts
      this.activeAlerts.delete(alertId);

      // Store in resolved alerts history
      await this.redis.lpush('alerts:resolved', JSON.stringify(alert));
      await this.redis.ltrim('alerts:resolved', 0, 999);

      // Notify about resolution
      await this.sendAlertResolutionNotification(alert);

      this.emit('alert:resolved', alert);
      logger.info(`Alert resolved: ${alert.id} by ${resolvedBy}`);
    } catch (error) {
      logger.error('Error resolving alert:', error);
      throw error;
    }
  }

  /**
   * Configure alert rules
   */
  public async configureAlertRule(rule: AlertRule): Promise<void> {
    try {
      this.alertRules.set(rule.id, rule);
      
      // Persist to database
      await prisma.alertRule.upsert({
        where: { id: rule.id },
        update: {
          name: rule.name,
          condition: rule.condition,
          threshold: rule.threshold,
          severity: rule.severity,
          enabled: rule.enabled,
          cooldownMinutes: rule.cooldownMinutes,
          channels: JSON.stringify(rule.channels),
          description: rule.description,
        },
        create: {
          id: rule.id,
          name: rule.name,
          condition: rule.condition,
          threshold: rule.threshold,
          severity: rule.severity,
          enabled: rule.enabled,
          cooldownMinutes: rule.cooldownMinutes,
          channels: JSON.stringify(rule.channels),
          description: rule.description,
        },
      });

      logger.info(`Alert rule configured: ${rule.id}`);
    } catch (error) {
      logger.error('Error configuring alert rule:', error);
      throw error;
    }
  }

  /**
   * Get error analytics and trends
   */
  public async getErrorAnalytics(period: string = '24h'): Promise<{
    totalErrors: number;
    errorsByType: Record<string, number>;
    errorsBySeverity: Record<string, number>;
    topErrors: Array<{ message: string; count: number }>;
    trend: Array<{ timestamp: string; count: number }>;
    resolutionTime: number;
  }> {
    try {
      const startTime = this.getPeriodStartTime(period);
      const recentErrors = this.errorHistory.filter(error => 
        error.timestamp >= startTime
      );

      // Error statistics
      const errorsByType: Record<string, number> = {};
      const errorsBySeverity: Record<string, number> = {};
      const errorCounts: Record<string, number> = {};

      recentErrors.forEach(error => {
        errorsByType[error.type] = (errorsByType[error.type] || 0) + 1;
        errorsBySeverity[error.severity] = (errorsBySeverity[error.severity] || 0) + 1;
        errorCounts[error.message] = (errorCounts[error.message] || 0) + 1;
      });

      // Top errors
      const topErrors = Object.entries(errorCounts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10)
        .map(([message, count]) => ({ message, count }));

      // Error trend (hourly buckets)
      const trend = this.generateErrorTrend(recentErrors, period);

      // Average resolution time
      const resolvedErrors = recentErrors.filter(error => error.resolved);
      const resolutionTime = resolvedErrors.length > 0 
        ? resolvedErrors.reduce((sum, error) => {
            const resolvedAt = new Date(); // Would track actual resolution time
            return sum + (resolvedAt.getTime() - error.timestamp.getTime());
          }, 0) / resolvedErrors.length
        : 0;

      return {
        totalErrors: recentErrors.length,
        errorsByType,
        errorsBySeverity,
        topErrors,
        trend,
        resolutionTime: resolutionTime / (1000 * 60), // Convert to minutes
      };
    } catch (error) {
      logger.error('Error getting error analytics:', error);
      throw error;
    }
  }

  /**
   * Private helper methods
   */
  private async evaluateRule(rule: AlertRule, metrics: Record<string, number>): Promise<boolean> {
    try {
      const { condition, threshold } = rule;
      
      // Simple condition evaluation (could be enhanced with expression parser)
      if (condition.includes('error_rate > threshold')) {
        return (metrics.errorRate || 0) > threshold;
      }
      
      if (condition.includes('db_connections < threshold')) {
        return (metrics.databaseConnections || 0) < threshold;
      }
      
      if (condition.includes('max_drawdown > threshold')) {
        return (metrics.maxDrawdown || 0) > threshold;
      }
      
      if (condition.includes('avg_response_time > threshold')) {
        return (metrics.avgResponseTime || 0) > threshold;
      }
      
      if (condition.includes('trade_failure_rate > threshold')) {
        return (metrics.tradeFailureRate || 0) > threshold;
      }
      
      if (condition.includes('memory_usage > threshold')) {
        return (metrics.memoryUsage || 0) > threshold;
      }
      
      if (condition.includes('websocket_errors > threshold')) {
        return (metrics.websocketErrors || 0) > threshold;
      }

      return false;
    } catch (error) {
      logger.error('Error evaluating rule:', error);
      return false;
    }
  }

  private isInCooldown(ruleId: string): boolean {
    const lastAlert = this.redis.get(`alert:cooldown:${ruleId}`);
    return lastAlert !== null;
  }

  private async triggerAlert(rule: AlertRule, metrics: Record<string, any>): Promise<void> {
    try {
      const alert: Alert = {
        id: this.generateId(),
        ruleId: rule.id,
        message: `${rule.name}: ${rule.description}`,
        severity: rule.severity,
        timestamp: new Date(),
        resolved: false,
        metadata: { metrics, rule },
      };

      // Store active alert
      this.activeAlerts.set(alert.id, alert);

      // Set cooldown
      await this.redis.setex(
        `alert:cooldown:${rule.id}`,
        rule.cooldownMinutes * 60,
        alert.id
      );

      // Send notifications
      await this.sendAlertNotifications(alert, rule.channels);

      // Store alert history
      await this.redis.lpush('alerts:history', JSON.stringify(alert));
      await this.redis.ltrim('alerts:history', 0, 999);

      this.emit('alert:triggered', alert);
      logger.warn(`Alert triggered: ${alert.id} - ${alert.message}`);
    } catch (error) {
      logger.error('Error triggering alert:', error);
    }
  }

  private async sendAlertNotifications(alert: Alert, channels: AlertChannel[]): Promise<void> {
    for (const channel of channels) {
      if (!channel.enabled) continue;

      try {
        switch (channel.type) {
          case 'email':
            await this.sendEmailAlert(alert, channel.config);
            break;
          case 'webhook':
            await this.sendWebhookAlert(alert, channel.config);
            break;
          case 'slack':
            await this.sendSlackAlert(alert, channel.config);
            break;
          // Add more channel types as needed
        }
      } catch (error) {
        logger.error(`Error sending alert via ${channel.type}:`, error);
      }
    }
  }

  private async sendEmailAlert(alert: Alert, config: any): Promise<void> {
    const emailOptions = {
      from: process.env.SMTP_FROM || 'alerts@trading.com',
      to: config.recipients.join(', '),
      subject: `[${alert.severity.toUpperCase()}] Trading Platform Alert`,
      html: `
        <h2>Trading Platform Alert</h2>
        <p><strong>Severity:</strong> ${alert.severity}</p>
        <p><strong>Message:</strong> ${alert.message}</p>
        <p><strong>Time:</strong> ${alert.timestamp.toISOString()}</p>
        <p><strong>Alert ID:</strong> ${alert.id}</p>
        <hr>
        <p>Please investigate and resolve this issue promptly.</p>
      `,
    };

    await this.emailTransporter.sendMail(emailOptions);
  }

  private async sendWebhookAlert(alert: Alert, config: any): Promise<void> {
    const payload = {
      alert_id: alert.id,
      severity: alert.severity,
      message: alert.message,
      timestamp: alert.timestamp.toISOString(),
      metadata: alert.metadata,
    };

    // Send webhook (implementation depends on webhook service)
    // await fetch(config.url, {
    //   method: 'POST',
    //   headers: { 'Content-Type': 'application/json' },
    //   body: JSON.stringify(payload),
    // });
  }

  private async sendSlackAlert(alert: Alert, config: any): Promise<void> {
    const payload = {
      text: `🚨 ${alert.severity.toUpperCase()} Alert`,
      attachments: [
        {
          color: this.getSeverityColor(alert.severity),
          fields: [
            { title: 'Message', value: alert.message, short: false },
            { title: 'Severity', value: alert.severity, short: true },
            { title: 'Time', value: alert.timestamp.toISOString(), short: true },
            { title: 'Alert ID', value: alert.id, short: true },
          ],
        },
      ],
    };

    // Send to Slack webhook
    // await fetch(config.webhook, {
    //   method: 'POST',
    //   headers: { 'Content-Type': 'application/json' },
    //   body: JSON.stringify(payload),
    // });
  }

  private getSeverityColor(severity: string): string {
    switch (severity) {
      case 'critical': return 'danger';
      case 'high': return 'warning';
      case 'medium': return '#ff9500';
      case 'low': return 'good';
      default: return '#000000';
    }
  }

  private async updateErrorMetrics(error: ErrorEvent): Promise<void> {
    const hour = new Date().toISOString().substring(0, 13); // YYYY-MM-DDTHH
    
    await this.redis.incr(`errors:count:${hour}`);
    await this.redis.incr(`errors:total`);
    await this.redis.incr(`errors:type:${error.type}`);
    await this.redis.incr(`errors:severity:${error.severity}`);
    
    // Set TTL for hourly counters
    await this.redis.expire(`errors:count:${hour}`, 7 * 24 * 60 * 60); // 7 days
  }

  private async checkErrorAlerts(error: ErrorEvent): Promise<void> {
    // Check if error rate is too high
    const currentHour = new Date().toISOString().substring(0, 13);
    const errorCount = await this.redis.get(`errors:count:${currentHour}`) || '0';
    const requestCount = await this.redis.get(`requests:count:${currentHour}`) || '1';
    
    const errorRate = parseInt(errorCount) / parseInt(requestCount);
    
    await this.checkAlerts({
      errorRate,
      errorCount: parseInt(errorCount),
    });
  }

  private async checkDatabaseHealth(): Promise<SystemHealthCheck> {
    const startTime = Date.now();
    
    try {
      await prisma.$queryRaw`SELECT 1`;
      const responseTime = Date.now() - startTime;
      
      return {
        name: 'Database',
        status: responseTime < 1000 ? 'healthy' : 'degraded',
        responseTime,
        lastCheck: new Date(),
        details: { connectionPool: 'available' },
      };
    } catch (error) {
      return {
        name: 'Database',
        status: 'unhealthy',
        responseTime: Date.now() - startTime,
        lastCheck: new Date(),
        details: { error: error instanceof Error ? error.message : 'Unknown error' },
      };
    }
  }

  private async checkRedisHealth(): Promise<SystemHealthCheck> {
    const startTime = Date.now();
    
    try {
      await this.redis.ping();
      const responseTime = Date.now() - startTime;
      
      return {
        name: 'Redis',
        status: responseTime < 100 ? 'healthy' : 'degraded',
        responseTime,
        lastCheck: new Date(),
      };
    } catch (error) {
      return {
        name: 'Redis',
        status: 'unhealthy',
        responseTime: Date.now() - startTime,
        lastCheck: new Date(),
        details: { error: error instanceof Error ? error.message : 'Unknown error' },
      };
    }
  }

  private async checkExternalAPIs(): Promise<SystemHealthCheck[]> {
    const checks: SystemHealthCheck[] = [];
    
    // Add external API health checks (exchange APIs, data providers, etc.)
    // Example implementation would ping each external service
    
    return checks;
  }

  private async checkSystemResources(): Promise<SystemHealthCheck[]> {
    const checks: SystemHealthCheck[] = [];
    
    // Memory check
    const memUsage = process.memoryUsage();
    const memUsagePercent = memUsage.heapUsed / memUsage.heapTotal;
    
    checks.push({
      name: 'Memory',
      status: memUsagePercent < 0.8 ? 'healthy' : memUsagePercent < 0.9 ? 'degraded' : 'unhealthy',
      responseTime: 0,
      lastCheck: new Date(),
      details: {
        heapUsed: memUsage.heapUsed,
        heapTotal: memUsage.heapTotal,
        usagePercent: memUsagePercent,
      },
    });
    
    return checks;
  }

  private async checkTradingSystem(): Promise<SystemHealthCheck[]> {
    const checks: SystemHealthCheck[] = [];
    
    // Check trading-specific components
    // - Order execution system
    // - Market data feeds
    // - Risk management system
    // - ML model endpoints
    
    return checks;
  }

  private async alertUnhealthyServices(services: SystemHealthCheck[]): Promise<void> {
    for (const service of services) {
      await this.trackError({
        type: 'system_error',
        message: `Service unhealthy: ${service.name}`,
        severity: 'high',
        context: { service },
      });
    }
  }

  private async sendAlertResolutionNotification(alert: Alert): Promise<void> {
    // Send resolution notification to same channels as original alert
    logger.info(`Alert resolved notification would be sent for: ${alert.id}`);
  }

  private startHealthChecks(): void {
    // Perform health checks every 60 seconds
    this.healthCheckInterval = setInterval(async () => {
      try {
        await this.performHealthChecks();
      } catch (error) {
        logger.error('Error in health check interval:', error);
      }
    }, 60000);
  }

  private getPeriodStartTime(period: string): Date {
    const now = new Date();
    
    switch (period) {
      case '1h':
        return new Date(now.getTime() - 60 * 60 * 1000);
      case '24h':
        return new Date(now.getTime() - 24 * 60 * 60 * 1000);
      case '7d':
        return new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
      default:
        return new Date(now.getTime() - 24 * 60 * 60 * 1000);
    }
  }

  private generateErrorTrend(errors: ErrorEvent[], period: string): Array<{ timestamp: string; count: number }> {
    const buckets = new Map<string, number>();
    const bucketSize = period === '1h' ? 5 * 60 * 1000 : 60 * 60 * 1000; // 5 min or 1 hour buckets
    
    errors.forEach(error => {
      const bucketTime = new Date(Math.floor(error.timestamp.getTime() / bucketSize) * bucketSize);
      const bucketKey = bucketTime.toISOString();
      buckets.set(bucketKey, (buckets.get(bucketKey) || 0) + 1);
    });
    
    return Array.from(buckets.entries())
      .map(([timestamp, count]) => ({ timestamp, count }))
      .sort((a, b) => a.timestamp.localeCompare(b.timestamp));
  }

  private generateId(): string {
    return `alert_${Date.now()}_${Math.random().toString(36).substring(2, 15)}`;
  }

  public async shutdown(): Promise<void> {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
    }
    
    await this.redis.quit();
    logger.info('Alerting service shut down');
  }
}

export default AlertingService;