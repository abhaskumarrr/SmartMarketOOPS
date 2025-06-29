/**
 * ML SERVICE BRIDGE - 2025 ENHANCED VERSION
 * 
 * Advanced ML service integration with 2025 patterns:
 * - Circuit breaker pattern for ML service calls
 * - Request/response caching with TTL
 * - Load balancing across multiple ML service instances
 * - Health checking and fallback strategies
 * - Performance monitoring and metrics collection
 * - Intelligent retry logic with exponential backoff
 * 
 * Provides enterprise-grade reliability for ML predictions in trading systems
 */

import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';
import { EventEmitter } from 'events';
import { logger } from '../../utils/logger';

// ML Service Configuration
export interface MLServiceConfig {
  baseURL: string;
  timeout: number;
  retries: number;
  healthCheckInterval: number;
  circuitBreaker: {
    failureThreshold: number;
    resetTimeout: number;
    monitoringWindow: number;
  };
  cache: {
    enabled: boolean;
    ttl: number;
    maxSize: number;
  };
  loadBalancer: {
    strategy: 'round-robin' | 'least-connections' | 'health-based';
    healthyOnly: boolean;
  };
}

// ML Service Instance
export interface MLServiceInstance {
  id: string;
  url: string;
  isHealthy: boolean;
  lastHealthCheck: number;
  responseTime: number;
  activeConnections: number;
  totalRequests: number;
  successfulRequests: number;
  failedRequests: number;
}

// Prediction Request/Response Types
export interface PredictionRequest {
  model: string;
  features: Record<string, any>;
  symbol: string;
  timeframe: string;
  timestamp: number;
  metadata?: Record<string, any>;
}

export interface PredictionResponse {
  prediction: number;
  confidence: number;
  model: string;
  processingTime: number;
  timestamp: number;
  metadata?: Record<string, any>;
}

// Cache Entry
interface CacheEntry {
  data: PredictionResponse;
  timestamp: number;
  ttl: number;
}

// Circuit Breaker States
enum CircuitState {
  CLOSED = 'CLOSED',
  OPEN = 'OPEN',
  HALF_OPEN = 'HALF_OPEN'
}

// Performance Metrics
export interface MLServiceMetrics {
  totalRequests: number;
  successfulRequests: number;
  failedRequests: number;
  averageResponseTime: number;
  cacheHitRate: number;
  circuitBreakerState: CircuitState;
  healthyInstances: number;
  totalInstances: number;
  requestsPerSecond: number;
}

export class MLServiceBridge extends EventEmitter {
  private instances: Map<string, MLServiceInstance> = new Map();
  private cache: Map<string, CacheEntry> = new Map();
  private circuitBreaker: CircuitBreaker;
  private currentInstanceIndex: number = 0;
  private metrics: MLServiceMetrics;
  private healthCheckInterval?: NodeJS.Timeout;
  private isRunning: boolean = false;

  // Default configuration
  private readonly DEFAULT_CONFIG: MLServiceConfig = {
    baseURL: process.env.ML_SERVICE_URL || 'http://localhost:8001',
    timeout: 30000, // 30 seconds
    retries: 3,
    healthCheckInterval: 30000, // 30 seconds
    circuitBreaker: {
      failureThreshold: 5,
      resetTimeout: 60000, // 1 minute
      monitoringWindow: 300000, // 5 minutes
    },
    cache: {
      enabled: true,
      ttl: 60000, // 1 minute
      maxSize: 1000,
    },
    loadBalancer: {
      strategy: 'health-based',
      healthyOnly: true,
    },
  };

  constructor(private config: Partial<MLServiceConfig> = {}) {
    super();
    
    this.config = { ...this.DEFAULT_CONFIG, ...config };
    this.circuitBreaker = new CircuitBreaker(this.config.circuitBreaker!);
    this.initializeMetrics();
    this.setupCircuitBreakerEvents();
    
    logger.info('🤖 MLServiceBridge initialized with 2025 patterns');
  }

  /**
   * Initialize the ML Service Bridge
   */
  async initialize(): Promise<void> {
    try {
      // Parse service URLs and create instances
      await this.createServiceInstances();
      
      // Start health monitoring
      this.startHealthMonitoring();
      
      this.isRunning = true;
      
      logger.info('✅ MLServiceBridge fully initialized');
      this.emit('initialized');
    } catch (error) {
      logger.error('❌ Failed to initialize MLServiceBridge:', error);
      throw error;
    }
  }

  /**
   * Create service instances from configuration
   */
  private async createServiceInstances(): Promise<void> {
    // Support multiple ML service instances for load balancing
    const serviceUrls = [
      this.config.baseURL,
      process.env.ML_SERVICE_URL_2,
      process.env.ML_SERVICE_URL_3,
    ].filter(Boolean) as string[];

    for (let i = 0; i < serviceUrls.length; i++) {
      const url = serviceUrls[i];
      const instance: MLServiceInstance = {
        id: `ml-service-${i + 1}`,
        url,
        isHealthy: false,
        lastHealthCheck: 0,
        responseTime: 0,
        activeConnections: 0,
        totalRequests: 0,
        successfulRequests: 0,
        failedRequests: 0,
      };
      
      this.instances.set(instance.id, instance);
      logger.info(`🔧 Added ML service instance: ${instance.id} at ${url}`);
    }

    // Perform initial health check
    await this.performHealthChecks();
  }

  /**
   * Make a prediction request with full resilience patterns
   */
  async predict(request: PredictionRequest): Promise<PredictionResponse | null> {
    const startTime = Date.now();
    
    try {
      // Check circuit breaker
      if (this.circuitBreaker.getState() === CircuitState.OPEN) {
        throw new Error('Circuit breaker is OPEN - ML service unavailable');
      }

      // Check cache first
      if (this.config.cache?.enabled) {
        const cached = this.getCachedPrediction(request);
        if (cached) {
          this.updateMetrics('cache_hit', Date.now() - startTime);
          return cached;
        }
      }

      // Select healthy instance
      const instance = this.selectInstance();
      if (!instance) {
        throw new Error('No healthy ML service instances available');
      }

      // Make prediction request
      const response = await this.makeRequest(instance, request);
      
      // Cache successful response
      if (this.config.cache?.enabled && response) {
        this.cachePrediction(request, response);
      }

      // Update metrics and circuit breaker
      const responseTime = Date.now() - startTime;
      this.updateMetrics('success', responseTime);
      this.circuitBreaker.recordSuccess();
      this.updateInstanceMetrics(instance, true, responseTime);

      logger.debug(`✅ ML prediction successful in ${responseTime}ms via ${instance.id}`);
      this.emit('predictionSuccess', { request, response, instance: instance.id, responseTime });
      
      return response;

    } catch (error) {
      const responseTime = Date.now() - startTime;
      this.updateMetrics('failure', responseTime);
      this.circuitBreaker.recordFailure();
      
      logger.error(`❌ ML prediction failed after ${responseTime}ms:`, error);
      this.emit('predictionError', { request, error, responseTime });
      
      // Try fallback strategies
      return await this.handlePredictionFailure(request, error as Error);
    }
  }

  /**
   * Make HTTP request to ML service with retry logic
   */
  private async makeRequest(
    instance: MLServiceInstance,
    request: PredictionRequest,
    attempt: number = 1
  ): Promise<PredictionResponse> {
    instance.activeConnections++;
    instance.totalRequests++;
    
    try {
      const client = this.createAxiosClient(instance);
      
      const response = await client.post('/predict', {
        model: request.model,
        features: request.features,
        symbol: request.symbol,
        timeframe: request.timeframe,
        timestamp: request.timestamp,
        metadata: request.metadata,
      });

      instance.activeConnections--;
      instance.successfulRequests++;
      
      return response.data;

    } catch (error: any) {
      instance.activeConnections--;
      instance.failedRequests++;
      
      // Retry logic with exponential backoff
      if (attempt < this.config.retries!) {
        const backoffDelay = Math.pow(2, attempt) * 1000; // Exponential backoff
        logger.warn(`🔄 Retrying ML request (attempt ${attempt + 1}/${this.config.retries}) after ${backoffDelay}ms`);
        
        await new Promise(resolve => setTimeout(resolve, backoffDelay));
        return this.makeRequest(instance, request, attempt + 1);
      }
      
      throw error;
    }
  }

  /**
   * Create Axios client with proper configuration
   */
  private createAxiosClient(instance: MLServiceInstance): AxiosInstance {
    return axios.create({
      baseURL: instance.url,
      timeout: this.config.timeout,
      headers: {
        'Content-Type': 'application/json',
        'X-Request-ID': `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      },
    });
  }

  /**
   * Select best instance based on load balancing strategy
   */
  private selectInstance(): MLServiceInstance | null {
    const healthyInstances = Array.from(this.instances.values()).filter(
      instance => !this.config.loadBalancer?.healthyOnly || instance.isHealthy
    );

    if (healthyInstances.length === 0) {
      return null;
    }

    switch (this.config.loadBalancer?.strategy) {
      case 'round-robin':
        return this.selectRoundRobin(healthyInstances);
      
      case 'least-connections':
        return this.selectLeastConnections(healthyInstances);
      
      case 'health-based':
      default:
        return this.selectHealthBased(healthyInstances);
    }
  }

  /**
   * Round-robin instance selection
   */
  private selectRoundRobin(instances: MLServiceInstance[]): MLServiceInstance {
    const selected = instances[this.currentInstanceIndex % instances.length];
    this.currentInstanceIndex++;
    return selected;
  }

  /**
   * Least connections instance selection
   */
  private selectLeastConnections(instances: MLServiceInstance[]): MLServiceInstance {
    return instances.reduce((best, current) => 
      current.activeConnections < best.activeConnections ? current : best
    );
  }

  /**
   * Health-based instance selection (best response time + healthy)
   */
  private selectHealthBased(instances: MLServiceInstance[]): MLServiceInstance {
    const healthyInstances = instances.filter(instance => instance.isHealthy);
    
    if (healthyInstances.length === 0) {
      return instances[0]; // Fallback to any instance
    }

    return healthyInstances.reduce((best, current) => 
      current.responseTime < best.responseTime ? current : best
    );
  }

  /**
   * Cache prediction response
   */
  private cachePrediction(request: PredictionRequest, response: PredictionResponse): void {
    const cacheKey = this.generateCacheKey(request);
    const entry: CacheEntry = {
      data: response,
      timestamp: Date.now(),
      ttl: this.config.cache!.ttl!,
    };

    // Manage cache size
    if (this.cache.size >= this.config.cache!.maxSize!) {
      this.evictOldestCacheEntry();
    }

    this.cache.set(cacheKey, entry);
  }

  /**
   * Get cached prediction if available and valid
   */
  private getCachedPrediction(request: PredictionRequest): PredictionResponse | null {
    const cacheKey = this.generateCacheKey(request);
    const entry = this.cache.get(cacheKey);

    if (!entry) return null;

    const now = Date.now();
    if (now - entry.timestamp > entry.ttl) {
      this.cache.delete(cacheKey);
      return null;
    }

    return entry.data;
  }

  /**
   * Generate cache key from request
   */
  private generateCacheKey(request: PredictionRequest): string {
    return `${request.model}:${request.symbol}:${request.timeframe}:${JSON.stringify(request.features)}`;
  }

  /**
   * Evict oldest cache entry
   */
  private evictOldestCacheEntry(): void {
    let oldestKey = '';
    let oldestTime = Date.now();

    for (const [key, entry] of this.cache.entries()) {
      if (entry.timestamp < oldestTime) {
        oldestTime = entry.timestamp;
        oldestKey = key;
      }
    }

    if (oldestKey) {
      this.cache.delete(oldestKey);
    }
  }

  /**
   * Handle prediction failure with fallback strategies
   */
  private async handlePredictionFailure(
    request: PredictionRequest,
    error: Error
  ): Promise<PredictionResponse | null> {
    // Try cached result even if expired
    const cacheKey = this.generateCacheKey(request);
    const entry = this.cache.get(cacheKey);
    if (entry) {
      logger.warn('🔄 Using expired cache entry as fallback');
      return entry.data;
    }

    // Use default/historical prediction as last resort
    logger.warn('🤖 Using default prediction as fallback');
    return {
      prediction: 0, // Neutral/hold signal
      confidence: 0.1, // Very low confidence
      model: 'fallback',
      processingTime: 0,
      timestamp: Date.now(),
      metadata: {
        fallback: true,
        originalError: error.message,
      },
    };
  }

  /**
   * Perform health checks on all instances
   */
  private async performHealthChecks(): Promise<void> {
    const healthPromises = Array.from(this.instances.values()).map(
      instance => this.checkInstanceHealth(instance)
    );

    await Promise.allSettled(healthPromises);
  }

  /**
   * Check health of a specific instance
   */
  private async checkInstanceHealth(instance: MLServiceInstance): Promise<void> {
    const startTime = Date.now();
    
    try {
      const client = this.createAxiosClient(instance);
      
      // Simple health check endpoint
      const response = await client.get('/health', { timeout: 5000 });
      
      const responseTime = Date.now() - startTime;
      const wasHealthy = instance.isHealthy;
      
      instance.isHealthy = response.status === 200;
      instance.lastHealthCheck = Date.now();
      instance.responseTime = responseTime;

      if (!wasHealthy && instance.isHealthy) {
        logger.info(`✅ ML service instance ${instance.id} is now healthy`);
        this.emit('instanceHealthy', instance.id);
      }

    } catch (error) {
      const wasHealthy = instance.isHealthy;
      
      instance.isHealthy = false;
      instance.lastHealthCheck = Date.now();
      
      if (wasHealthy) {
        logger.warn(`❌ ML service instance ${instance.id} is now unhealthy:`, error);
        this.emit('instanceUnhealthy', { instanceId: instance.id, error });
      }
    }
  }

  /**
   * Get current metrics
   */
  getMetrics(): MLServiceMetrics {
    return { ...this.metrics };
  }

  /**
   * Get instance health status
   */
  getInstanceHealth(): MLServiceInstance[] {
    return Array.from(this.instances.values()).map(instance => ({ ...instance }));
  }

  /**
   * Shutdown the ML service bridge
   */
  async shutdown(): Promise<void> {
    logger.info('🛑 Shutting down MLServiceBridge...');
    
    this.isRunning = false;
    
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
    }
    
    this.cache.clear();
    
    logger.info('✅ MLServiceBridge shutdown complete');
    this.emit('shutdown');
  }

  // Private helper methods...
  
  private initializeMetrics(): void {
    this.metrics = {
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0,
      averageResponseTime: 0,
      cacheHitRate: 0,
      circuitBreakerState: CircuitState.CLOSED,
      healthyInstances: 0,
      totalInstances: 0,
      requestsPerSecond: 0,
    };
  }

  private setupCircuitBreakerEvents(): void {
    this.circuitBreaker.on('open', () => {
      logger.warn('🔴 ML Service Circuit Breaker OPEN');
      this.metrics.circuitBreakerState = CircuitState.OPEN;
      this.emit('circuitBreakerOpen');
    });

    this.circuitBreaker.on('halfOpen', () => {
      logger.info('🟡 ML Service Circuit Breaker HALF-OPEN');
      this.metrics.circuitBreakerState = CircuitState.HALF_OPEN;
      this.emit('circuitBreakerHalfOpen');
    });

    this.circuitBreaker.on('closed', () => {
      logger.info('🟢 ML Service Circuit Breaker CLOSED');
      this.metrics.circuitBreakerState = CircuitState.CLOSED;
      this.emit('circuitBreakerClosed');
    });
  }

  private startHealthMonitoring(): void {
    this.healthCheckInterval = setInterval(async () => {
      if (!this.isRunning) return;
      
      try {
        await this.performHealthChecks();
        this.updateHealthMetrics();
        this.cleanupExpiredCache();
      } catch (error) {
        logger.error('❌ Health monitoring error:', error);
      }
    }, this.config.healthCheckInterval);
  }

  private updateHealthMetrics(): void {
    const instances = Array.from(this.instances.values());
    this.metrics.totalInstances = instances.length;
    this.metrics.healthyInstances = instances.filter(i => i.isHealthy).length;
  }

  private updateMetrics(type: 'success' | 'failure' | 'cache_hit', responseTime: number): void {
    this.metrics.totalRequests++;
    
    if (type === 'success') {
      this.metrics.successfulRequests++;
    } else if (type === 'failure') {
      this.metrics.failedRequests++;
    }
    
    // Update average response time
    this.metrics.averageResponseTime = 
      (this.metrics.averageResponseTime + responseTime) / 2;
    
    // Update cache hit rate
    const totalRequests = this.metrics.totalRequests;
    const cacheHits = totalRequests - this.metrics.successfulRequests - this.metrics.failedRequests;
    this.metrics.cacheHitRate = totalRequests > 0 ? cacheHits / totalRequests : 0;
  }

  private updateInstanceMetrics(
    instance: MLServiceInstance,
    success: boolean,
    responseTime: number
  ): void {
    if (success) {
      instance.successfulRequests++;
    } else {
      instance.failedRequests++;
    }
    
    instance.responseTime = (instance.responseTime + responseTime) / 2;
  }

  private cleanupExpiredCache(): void {
    const now = Date.now();
    
    for (const [key, entry] of this.cache.entries()) {
      if (now - entry.timestamp > entry.ttl) {
        this.cache.delete(key);
      }
    }
  }
}

/**
 * Circuit Breaker Implementation for ML Service
 */
class CircuitBreaker extends EventEmitter {
  private state: CircuitState = CircuitState.CLOSED;
  private failures: number[] = [];
  private lastFailureTime: number = 0;
  private halfOpenAttempts: number = 0;

  constructor(private config: MLServiceConfig['circuitBreaker']) {
    super();
  }

  recordSuccess(): void {
    if (this.state === CircuitState.HALF_OPEN) {
      this.setState(CircuitState.CLOSED);
      this.failures = [];
      this.halfOpenAttempts = 0;
    }
  }

  recordFailure(): void {
    const now = Date.now();
    this.failures.push(now);
    this.lastFailureTime = now;

    // Clean old failures outside monitoring window
    this.failures = this.failures.filter(
      time => now - time < this.config!.monitoringWindow!
    );

    if (this.failures.length >= this.config!.failureThreshold!) {
      this.setState(CircuitState.OPEN);
    }
  }

  getState(): CircuitState {
    if (this.state === CircuitState.OPEN) {
      const now = Date.now();
      if (now - this.lastFailureTime >= this.config!.resetTimeout!) {
        this.setState(CircuitState.HALF_OPEN);
        this.halfOpenAttempts = 0;
      }
    }
    return this.state;
  }

  private setState(newState: CircuitState): void {
    if (this.state !== newState) {
      this.state = newState;
      this.emit(newState.toLowerCase());
    }
  }
}

export default MLServiceBridge;