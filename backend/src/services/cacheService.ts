/**
 * Advanced Caching Service
 * Redis-based caching with multiple strategies and performance optimization
 */

import Redis from 'ioredis';
import { performance } from 'perf_hooks';

interface CacheConfig {
  host: string;
  port: number;
  password?: string;
  db?: number;
  keyPrefix?: string;
  retryDelayOnFailover?: number;
  maxRetriesPerRequest?: number;
  lazyConnect?: boolean;
}

interface CacheOptions {
  ttl?: number; // Time to live in seconds
  compress?: boolean;
  tags?: string[];
  namespace?: string;
}

interface CacheStats {
  hits: number;
  misses: number;
  sets: number;
  deletes: number;
  errors: number;
  totalRequests: number;
  avgResponseTime: number;
}

class CacheService {
  private redis: Redis;
  private stats: CacheStats;
  private isConnected: boolean = false;
  private compressionThreshold: number = 1024; // Compress data larger than 1KB

  constructor(config: CacheConfig) {
    this.redis = new Redis({
      host: config.host,
      port: config.port,
      password: config.password,
      db: config.db || 0,
      keyPrefix: config.keyPrefix || 'smartmarket:',
      retryDelayOnFailover: config.retryDelayOnFailover || 100,
      maxRetriesPerRequest: config.maxRetriesPerRequest || 3,
      lazyConnect: config.lazyConnect || true,
      // Connection pool settings
      family: 4,
      keepAlive: true,
      // Performance optimizations
      enableReadyCheck: false,
      maxLoadingTimeout: 5000,
    });

    this.stats = {
      hits: 0,
      misses: 0,
      sets: 0,
      deletes: 0,
      errors: 0,
      totalRequests: 0,
      avgResponseTime: 0,
    };

    this.setupEventHandlers();
  }

  private setupEventHandlers(): void {
    this.redis.on('connect', () => {
      console.log('Redis connected');
      this.isConnected = true;
    });

    this.redis.on('error', (error) => {
      console.error('Redis error:', error);
      this.stats.errors++;
      this.isConnected = false;
    });

    this.redis.on('close', () => {
      console.log('Redis connection closed');
      this.isConnected = false;
    });

    this.redis.on('reconnecting', () => {
      console.log('Redis reconnecting...');
    });
  }

  private async measureOperation<T>(operation: () => Promise<T>): Promise<T> {
    const start = performance.now();
    try {
      const result = await operation();
      const duration = performance.now() - start;
      this.updateStats(duration);
      return result;
    } catch (error) {
      this.stats.errors++;
      throw error;
    }
  }

  private updateStats(duration: number): void {
    this.stats.totalRequests++;
    this.stats.avgResponseTime = 
      (this.stats.avgResponseTime * (this.stats.totalRequests - 1) + duration) / 
      this.stats.totalRequests;
  }

  private compressData(data: string): string {
    if (data.length < this.compressionThreshold) {
      return data;
    }
    // Simple compression using JSON.stringify optimization
    return JSON.stringify(JSON.parse(data));
  }

  private decompressData(data: string): string {
    return data;
  }

  private buildKey(key: string, namespace?: string): string {
    return namespace ? `${namespace}:${key}` : key;
  }

  async get<T>(key: string, options: CacheOptions = {}): Promise<T | null> {
    if (!this.isConnected) {
      return null;
    }

    return this.measureOperation(async () => {
      try {
        const fullKey = this.buildKey(key, options.namespace);
        const data = await this.redis.get(fullKey);
        
        if (data === null) {
          this.stats.misses++;
          return null;
        }

        this.stats.hits++;
        const decompressed = this.decompressData(data);
        return JSON.parse(decompressed);
      } catch (error) {
        console.error('Cache get error:', error);
        this.stats.misses++;
        return null;
      }
    });
  }

  async set<T>(
    key: string, 
    value: T, 
    options: CacheOptions = {}
  ): Promise<boolean> {
    if (!this.isConnected) {
      return false;
    }

    return this.measureOperation(async () => {
      try {
        const fullKey = this.buildKey(key, options.namespace);
        const serialized = JSON.stringify(value);
        const compressed = options.compress ? this.compressData(serialized) : serialized;
        
        if (options.ttl) {
          await this.redis.setex(fullKey, options.ttl, compressed);
        } else {
          await this.redis.set(fullKey, compressed);
        }

        // Handle tags for cache invalidation
        if (options.tags && options.tags.length > 0) {
          const tagPromises = options.tags.map(tag => 
            this.redis.sadd(`tag:${tag}`, fullKey)
          );
          await Promise.all(tagPromises);
        }

        this.stats.sets++;
        return true;
      } catch (error) {
        console.error('Cache set error:', error);
        return false;
      }
    });
  }

  async del(key: string, namespace?: string): Promise<boolean> {
    if (!this.isConnected) {
      return false;
    }

    return this.measureOperation(async () => {
      try {
        const fullKey = this.buildKey(key, namespace);
        const result = await this.redis.del(fullKey);
        this.stats.deletes++;
        return result > 0;
      } catch (error) {
        console.error('Cache delete error:', error);
        return false;
      }
    });
  }

  async invalidateByTag(tag: string): Promise<number> {
    if (!this.isConnected) {
      return 0;
    }

    return this.measureOperation(async () => {
      try {
        const keys = await this.redis.smembers(`tag:${tag}`);
        if (keys.length === 0) {
          return 0;
        }

        const pipeline = this.redis.pipeline();
        keys.forEach(key => pipeline.del(key));
        pipeline.del(`tag:${tag}`);
        
        await pipeline.exec();
        return keys.length;
      } catch (error) {
        console.error('Cache invalidate by tag error:', error);
        return 0;
      }
    });
  }

  async mget<T>(keys: string[], namespace?: string): Promise<(T | null)[]> {
    if (!this.isConnected || keys.length === 0) {
      return keys.map(() => null);
    }

    return this.measureOperation(async () => {
      try {
        const fullKeys = keys.map(key => this.buildKey(key, namespace));
        const results = await this.redis.mget(...fullKeys);
        
        return results.map((data, index) => {
          if (data === null) {
            this.stats.misses++;
            return null;
          }
          
          this.stats.hits++;
          try {
            const decompressed = this.decompressData(data);
            return JSON.parse(decompressed);
          } catch (error) {
            console.error(`Error parsing cached data for key ${keys[index]}:`, error);
            return null;
          }
        });
      } catch (error) {
        console.error('Cache mget error:', error);
        return keys.map(() => null);
      }
    });
  }

  async mset<T>(
    keyValuePairs: Array<{ key: string; value: T; ttl?: number }>,
    options: CacheOptions = {}
  ): Promise<boolean> {
    if (!this.isConnected || keyValuePairs.length === 0) {
      return false;
    }

    return this.measureOperation(async () => {
      try {
        const pipeline = this.redis.pipeline();
        
        keyValuePairs.forEach(({ key, value, ttl }) => {
          const fullKey = this.buildKey(key, options.namespace);
          const serialized = JSON.stringify(value);
          const compressed = options.compress ? this.compressData(serialized) : serialized;
          
          if (ttl || options.ttl) {
            pipeline.setex(fullKey, ttl || options.ttl!, compressed);
          } else {
            pipeline.set(fullKey, compressed);
          }
        });

        await pipeline.exec();
        this.stats.sets += keyValuePairs.length;
        return true;
      } catch (error) {
        console.error('Cache mset error:', error);
        return false;
      }
    });
  }

  async exists(key: string, namespace?: string): Promise<boolean> {
    if (!this.isConnected) {
      return false;
    }

    return this.measureOperation(async () => {
      try {
        const fullKey = this.buildKey(key, namespace);
        const result = await this.redis.exists(fullKey);
        return result === 1;
      } catch (error) {
        console.error('Cache exists error:', error);
        return false;
      }
    });
  }

  async ttl(key: string, namespace?: string): Promise<number> {
    if (!this.isConnected) {
      return -1;
    }

    return this.measureOperation(async () => {
      try {
        const fullKey = this.buildKey(key, namespace);
        return await this.redis.ttl(fullKey);
      } catch (error) {
        console.error('Cache TTL error:', error);
        return -1;
      }
    });
  }

  async flush(namespace?: string): Promise<boolean> {
    if (!this.isConnected) {
      return false;
    }

    return this.measureOperation(async () => {
      try {
        if (namespace) {
          const pattern = this.buildKey('*', namespace);
          const keys = await this.redis.keys(pattern);
          if (keys.length > 0) {
            await this.redis.del(...keys);
          }
        } else {
          await this.redis.flushdb();
        }
        return true;
      } catch (error) {
        console.error('Cache flush error:', error);
        return false;
      }
    });
  }

  getStats(): CacheStats {
    const hitRate = this.stats.totalRequests > 0 
      ? (this.stats.hits / (this.stats.hits + this.stats.misses)) * 100 
      : 0;
    
    return {
      ...this.stats,
      hitRate: parseFloat(hitRate.toFixed(2)),
    } as CacheStats & { hitRate: number };
  }

  async getInfo(): Promise<any> {
    if (!this.isConnected) {
      return null;
    }

    try {
      const info = await this.redis.info();
      return info;
    } catch (error) {
      console.error('Cache info error:', error);
      return null;
    }
  }

  async disconnect(): Promise<void> {
    try {
      await this.redis.quit();
      this.isConnected = false;
    } catch (error) {
      console.error('Cache disconnect error:', error);
    }
  }
}

// Singleton instance
let cacheInstance: CacheService | null = null;

export const createCacheService = (config: CacheConfig): CacheService => {
  if (!cacheInstance) {
    cacheInstance = new CacheService(config);
  }
  return cacheInstance;
};

export const getCacheService = (): CacheService | null => {
  return cacheInstance;
};

export { CacheService, CacheConfig, CacheOptions, CacheStats };
