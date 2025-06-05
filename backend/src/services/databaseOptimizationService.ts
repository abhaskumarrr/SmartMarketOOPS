/**
 * Database Optimization Service
 * Handles connection pooling, query optimization, and performance monitoring
 */

import { PrismaClient } from '../../generated/prisma';
import { performance } from 'perf_hooks';

interface QueryMetrics {
  query: string;
  duration: number;
  timestamp: number;
  success: boolean;
  error?: string;
}

interface DatabaseStats {
  totalQueries: number;
  avgQueryTime: number;
  slowQueries: number;
  failedQueries: number;
  connectionPoolSize: number;
  activeConnections: number;
}

class DatabaseOptimizationService {
  private prisma: PrismaClient;
  private queryMetrics: QueryMetrics[] = [];
  private slowQueryThreshold: number = 1000; // 1 second
  private maxMetricsHistory: number = 1000;
  private stats: DatabaseStats;

  constructor() {
    this.prisma = new PrismaClient({
      log: [
        { emit: 'event', level: 'query' },
        { emit: 'event', level: 'error' },
        { emit: 'event', level: 'info' },
        { emit: 'event', level: 'warn' },
      ],
      datasources: {
        db: {
          url: process.env.DATABASE_URL,
        },
      },
    });

    this.stats = {
      totalQueries: 0,
      avgQueryTime: 0,
      slowQueries: 0,
      failedQueries: 0,
      connectionPoolSize: 0,
      activeConnections: 0,
    };

    this.setupQueryLogging();
  }

  private setupQueryLogging(): void {
    this.prisma.$on('query', (e) => {
      const duration = parseFloat(e.duration);
      const metric: QueryMetrics = {
        query: e.query,
        duration,
        timestamp: Date.now(),
        success: true,
      };

      this.addQueryMetric(metric);
      
      if (duration > this.slowQueryThreshold) {
        console.warn(`Slow query detected (${duration}ms):`, e.query);
        this.stats.slowQueries++;
      }
    });

    this.prisma.$on('error', (e) => {
      const metric: QueryMetrics = {
        query: e.target || 'Unknown',
        duration: 0,
        timestamp: Date.now(),
        success: false,
        error: e.message,
      };

      this.addQueryMetric(metric);
      this.stats.failedQueries++;
      console.error('Database error:', e);
    });
  }

  private addQueryMetric(metric: QueryMetrics): void {
    this.queryMetrics.push(metric);
    
    // Keep only recent metrics
    if (this.queryMetrics.length > this.maxMetricsHistory) {
      this.queryMetrics = this.queryMetrics.slice(-this.maxMetricsHistory);
    }

    // Update stats
    this.stats.totalQueries++;
    if (metric.success) {
      this.stats.avgQueryTime = 
        (this.stats.avgQueryTime * (this.stats.totalQueries - 1) + metric.duration) / 
        this.stats.totalQueries;
    }
  }

  // Optimized query wrapper with caching and metrics
  async executeQuery<T>(
    queryFn: () => Promise<T>,
    cacheKey?: string,
    cacheTtl: number = 300
  ): Promise<T> {
    const start = performance.now();
    
    try {
      // Check cache first if cacheKey provided
      if (cacheKey) {
        const { getCacheService } = await import('./cacheService');
        const cache = getCacheService();
        
        if (cache) {
          const cached = await cache.get<T>(cacheKey);
          if (cached !== null) {
            return cached;
          }
        }
      }

      // Execute query
      const result = await queryFn();
      
      // Cache result if cacheKey provided
      if (cacheKey && result) {
        const { getCacheService } = await import('./cacheService');
        const cache = getCacheService();
        
        if (cache) {
          await cache.set(cacheKey, result, { ttl: cacheTtl });
        }
      }

      return result;
    } catch (error) {
      const duration = performance.now() - start;
      console.error(`Query failed after ${duration}ms:`, error);
      throw error;
    }
  }

  // Batch operations for better performance
  async batchInsert<T>(
    model: any,
    data: T[],
    batchSize: number = 100
  ): Promise<void> {
    const batches = [];
    for (let i = 0; i < data.length; i += batchSize) {
      batches.push(data.slice(i, i + batchSize));
    }

    for (const batch of batches) {
      await this.executeQuery(() => model.createMany({
        data: batch,
        skipDuplicates: true,
      }));
    }
  }

  // Optimized pagination
  async paginatedQuery<T>(
    queryFn: (skip: number, take: number) => Promise<T[]>,
    countFn: () => Promise<number>,
    page: number = 1,
    limit: number = 20,
    cacheKey?: string
  ): Promise<{
    data: T[];
    total: number;
    page: number;
    limit: number;
    totalPages: number;
    hasNext: boolean;
    hasPrev: boolean;
  }> {
    const skip = (page - 1) * limit;
    
    const [data, total] = await Promise.all([
      this.executeQuery(
        () => queryFn(skip, limit),
        cacheKey ? `${cacheKey}:page:${page}:limit:${limit}` : undefined
      ),
      this.executeQuery(
        countFn,
        cacheKey ? `${cacheKey}:count` : undefined,
        600 // Cache count for 10 minutes
      ),
    ]);

    const totalPages = Math.ceil(total / limit);

    return {
      data,
      total,
      page,
      limit,
      totalPages,
      hasNext: page < totalPages,
      hasPrev: page > 1,
    };
  }

  // Connection pool monitoring
  async getConnectionInfo(): Promise<any> {
    try {
      const result = await this.prisma.$queryRaw`
        SELECT 
          count(*) as total_connections,
          count(*) FILTER (WHERE state = 'active') as active_connections,
          count(*) FILTER (WHERE state = 'idle') as idle_connections
        FROM pg_stat_activity 
        WHERE datname = current_database()
      `;
      return result;
    } catch (error) {
      console.error('Error getting connection info:', error);
      return null;
    }
  }

  // Query performance analysis
  getSlowQueries(limit: number = 10): QueryMetrics[] {
    return this.queryMetrics
      .filter(m => m.success && m.duration > this.slowQueryThreshold)
      .sort((a, b) => b.duration - a.duration)
      .slice(0, limit);
  }

  getQueryStats(): DatabaseStats {
    return { ...this.stats };
  }

  // Database health check
  async healthCheck(): Promise<{
    status: 'healthy' | 'degraded' | 'unhealthy';
    details: any;
  }> {
    try {
      const start = performance.now();
      await this.prisma.$queryRaw`SELECT 1`;
      const queryTime = performance.now() - start;

      const connectionInfo = await this.getConnectionInfo();
      
      let status: 'healthy' | 'degraded' | 'unhealthy' = 'healthy';
      
      if (queryTime > 1000) {
        status = 'degraded';
      }
      
      if (queryTime > 5000 || this.stats.failedQueries > this.stats.totalQueries * 0.1) {
        status = 'unhealthy';
      }

      return {
        status,
        details: {
          queryTime,
          connectionInfo,
          stats: this.getQueryStats(),
          recentErrors: this.queryMetrics
            .filter(m => !m.success)
            .slice(-5),
        },
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        details: {
          error: error instanceof Error ? error.message : 'Unknown error',
        },
      };
    }
  }

  // Index analysis and recommendations
  async analyzeIndexUsage(): Promise<any> {
    try {
      const indexUsage = await this.prisma.$queryRaw`
        SELECT 
          schemaname,
          tablename,
          indexname,
          idx_tup_read,
          idx_tup_fetch,
          idx_scan
        FROM pg_stat_user_indexes
        ORDER BY idx_scan DESC
      `;

      const unusedIndexes = await this.prisma.$queryRaw`
        SELECT 
          schemaname,
          tablename,
          indexname
        FROM pg_stat_user_indexes
        WHERE idx_scan = 0
      `;

      return {
        mostUsedIndexes: indexUsage,
        unusedIndexes,
      };
    } catch (error) {
      console.error('Error analyzing index usage:', error);
      return null;
    }
  }

  // Table statistics
  async getTableStats(): Promise<any> {
    try {
      const tableStats = await this.prisma.$queryRaw`
        SELECT 
          schemaname,
          tablename,
          n_tup_ins as inserts,
          n_tup_upd as updates,
          n_tup_del as deletes,
          n_live_tup as live_tuples,
          n_dead_tup as dead_tuples,
          last_vacuum,
          last_autovacuum,
          last_analyze,
          last_autoanalyze
        FROM pg_stat_user_tables
        ORDER BY n_live_tup DESC
      `;

      return tableStats;
    } catch (error) {
      console.error('Error getting table stats:', error);
      return null;
    }
  }

  // Cleanup and maintenance
  async cleanup(): Promise<void> {
    try {
      // Clear old metrics
      const oneHourAgo = Date.now() - 60 * 60 * 1000;
      this.queryMetrics = this.queryMetrics.filter(m => m.timestamp > oneHourAgo);
      
      // Reset stats if needed
      if (this.stats.totalQueries > 10000) {
        this.stats = {
          totalQueries: 0,
          avgQueryTime: 0,
          slowQueries: 0,
          failedQueries: 0,
          connectionPoolSize: 0,
          activeConnections: 0,
        };
      }
    } catch (error) {
      console.error('Error during cleanup:', error);
    }
  }

  async disconnect(): Promise<void> {
    await this.prisma.$disconnect();
  }
}

// Singleton instance
let dbOptimizationInstance: DatabaseOptimizationService | null = null;

export const createDatabaseOptimizationService = (): DatabaseOptimizationService => {
  if (!dbOptimizationInstance) {
    dbOptimizationInstance = new DatabaseOptimizationService();
  }
  return dbOptimizationInstance;
};

export const getDatabaseOptimizationService = (): DatabaseOptimizationService | null => {
  return dbOptimizationInstance;
};

export { DatabaseOptimizationService, QueryMetrics, DatabaseStats };
