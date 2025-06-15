/**
 * Cache Service Tests
 * Tests for the Redis-based caching service
 */

import { CacheService } from '../../services/cacheService';

// Mock ioredis
jest.mock('ioredis', () => {
  return jest.fn().mockImplementation(() => ({
    on: jest.fn(),
    get: jest.fn(),
    set: jest.fn(),
    setex: jest.fn(),
    del: jest.fn(),
    exists: jest.fn(),
    ttl: jest.fn(),
    mget: jest.fn(),
    mset: jest.fn(),
    sadd: jest.fn(),
    smembers: jest.fn(),
    keys: jest.fn(),
    flushdb: jest.fn(),
    info: jest.fn(),
    quit: jest.fn(),
    pipeline: jest.fn(() => ({
      del: jest.fn(),
      set: jest.fn(),
      setex: jest.fn(),
      exec: jest.fn().mockResolvedValue([]),
    })),
  }));
});

describe('CacheService', () => {
  let cacheService: CacheService;
  let mockRedis: any;

  beforeEach(() => {
    const Redis = require('ioredis');
    mockRedis = new Redis();
    cacheService = new CacheService({
      host: 'localhost',
      port: 6379,
      keyPrefix: 'test:',
    });
    
    // Simulate connected state
    (cacheService as any).isConnected = true;
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('get', () => {
    it('should retrieve data from cache', async () => {
      const testData = { id: 1, name: 'test' };
      mockRedis.get.mockResolvedValue(JSON.stringify(testData));

      const result = await cacheService.get('test-key');

      expect(mockRedis.get).toHaveBeenCalledWith('test-key');
      expect(result).toEqual(testData);
    });

    it('should return null for non-existent keys', async () => {
      mockRedis.get.mockResolvedValue(null);

      const result = await cacheService.get('non-existent-key');

      expect(result).toBeNull();
    });

    it('should handle JSON parse errors', async () => {
      mockRedis.get.mockResolvedValue('invalid-json');
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation();

      const result = await cacheService.get('invalid-key');

      expect(result).toBeNull();
      expect(consoleSpy).toHaveBeenCalledWith('Cache get error:', expect.any(Error));
      
      consoleSpy.mockRestore();
    });

    it('should return null when not connected', async () => {
      (cacheService as any).isConnected = false;

      const result = await cacheService.get('test-key');

      expect(result).toBeNull();
      expect(mockRedis.get).not.toHaveBeenCalled();
    });

    it('should use namespace in key', async () => {
      mockRedis.get.mockResolvedValue(JSON.stringify({ data: 'test' }));

      await cacheService.get('test-key', { namespace: 'api' });

      expect(mockRedis.get).toHaveBeenCalledWith('api:test-key');
    });
  });

  describe('set', () => {
    it('should store data in cache', async () => {
      const testData = { id: 1, name: 'test' };
      mockRedis.set.mockResolvedValue('OK');

      const result = await cacheService.set('test-key', testData);

      expect(mockRedis.set).toHaveBeenCalledWith('test-key', JSON.stringify(testData));
      expect(result).toBe(true);
    });

    it('should store data with TTL', async () => {
      const testData = { id: 1, name: 'test' };
      mockRedis.setex.mockResolvedValue('OK');

      const result = await cacheService.set('test-key', testData, { ttl: 300 });

      expect(mockRedis.setex).toHaveBeenCalledWith('test-key', 300, JSON.stringify(testData));
      expect(result).toBe(true);
    });

    it('should handle tags for cache invalidation', async () => {
      const testData = { id: 1, name: 'test' };
      mockRedis.set.mockResolvedValue('OK');
      mockRedis.sadd.mockResolvedValue(1);

      const result = await cacheService.set('test-key', testData, { 
        tags: ['user', 'profile'] 
      });

      expect(mockRedis.set).toHaveBeenCalledWith('test-key', JSON.stringify(testData));
      expect(mockRedis.sadd).toHaveBeenCalledWith('tag:user', 'test-key');
      expect(mockRedis.sadd).toHaveBeenCalledWith('tag:profile', 'test-key');
      expect(result).toBe(true);
    });

    it('should return false when not connected', async () => {
      (cacheService as any).isConnected = false;

      const result = await cacheService.set('test-key', { data: 'test' });

      expect(result).toBe(false);
      expect(mockRedis.set).not.toHaveBeenCalled();
    });

    it('should handle Redis errors', async () => {
      mockRedis.set.mockRejectedValue(new Error('Redis error'));
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation();

      const result = await cacheService.set('test-key', { data: 'test' });

      expect(result).toBe(false);
      expect(consoleSpy).toHaveBeenCalledWith('Cache set error:', expect.any(Error));
      
      consoleSpy.mockRestore();
    });
  });

  describe('del', () => {
    it('should delete key from cache', async () => {
      mockRedis.del.mockResolvedValue(1);

      const result = await cacheService.del('test-key');

      expect(mockRedis.del).toHaveBeenCalledWith('test-key');
      expect(result).toBe(true);
    });

    it('should return false for non-existent keys', async () => {
      mockRedis.del.mockResolvedValue(0);

      const result = await cacheService.del('non-existent-key');

      expect(result).toBe(false);
    });

    it('should use namespace in key', async () => {
      mockRedis.del.mockResolvedValue(1);

      await cacheService.del('test-key', 'api');

      expect(mockRedis.del).toHaveBeenCalledWith('api:test-key');
    });
  });

  describe('mget', () => {
    it('should retrieve multiple keys', async () => {
      const testData1 = { id: 1, name: 'test1' };
      const testData2 = { id: 2, name: 'test2' };
      
      mockRedis.mget.mockResolvedValue([
        JSON.stringify(testData1),
        JSON.stringify(testData2),
        null,
      ]);

      const result = await cacheService.mget(['key1', 'key2', 'key3']);

      expect(mockRedis.mget).toHaveBeenCalledWith('key1', 'key2', 'key3');
      expect(result).toEqual([testData1, testData2, null]);
    });

    it('should handle empty key array', async () => {
      const result = await cacheService.mget([]);

      expect(result).toEqual([]);
      expect(mockRedis.mget).not.toHaveBeenCalled();
    });
  });

  describe('invalidateByTag', () => {
    it('should invalidate all keys with tag', async () => {
      const keys = ['key1', 'key2', 'key3'];
      mockRedis.smembers.mockResolvedValue(keys);
      
      const mockPipeline = {
        del: jest.fn(),
        exec: jest.fn().mockResolvedValue([]),
      };
      mockRedis.pipeline.mockReturnValue(mockPipeline);

      const result = await cacheService.invalidateByTag('user');

      expect(mockRedis.smembers).toHaveBeenCalledWith('tag:user');
      expect(mockPipeline.del).toHaveBeenCalledTimes(4); // 3 keys + tag
      expect(result).toBe(3);
    });

    it('should return 0 for non-existent tags', async () => {
      mockRedis.smembers.mockResolvedValue([]);

      const result = await cacheService.invalidateByTag('non-existent');

      expect(result).toBe(0);
    });
  });

  describe('exists', () => {
    it('should check if key exists', async () => {
      mockRedis.exists.mockResolvedValue(1);

      const result = await cacheService.exists('test-key');

      expect(mockRedis.exists).toHaveBeenCalledWith('test-key');
      expect(result).toBe(true);
    });

    it('should return false for non-existent keys', async () => {
      mockRedis.exists.mockResolvedValue(0);

      const result = await cacheService.exists('non-existent-key');

      expect(result).toBe(false);
    });
  });

  describe('ttl', () => {
    it('should return TTL for key', async () => {
      mockRedis.ttl.mockResolvedValue(300);

      const result = await cacheService.ttl('test-key');

      expect(mockRedis.ttl).toHaveBeenCalledWith('test-key');
      expect(result).toBe(300);
    });

    it('should return -1 for keys without TTL', async () => {
      mockRedis.ttl.mockResolvedValue(-1);

      const result = await cacheService.ttl('persistent-key');

      expect(result).toBe(-1);
    });
  });

  describe('flush', () => {
    it('should flush all cache', async () => {
      mockRedis.flushdb.mockResolvedValue('OK');

      const result = await cacheService.flush();

      expect(mockRedis.flushdb).toHaveBeenCalled();
      expect(result).toBe(true);
    });

    it('should flush namespace keys', async () => {
      const keys = ['api:key1', 'api:key2'];
      mockRedis.keys.mockResolvedValue(keys);
      mockRedis.del.mockResolvedValue(2);

      const result = await cacheService.flush('api');

      expect(mockRedis.keys).toHaveBeenCalledWith('api:*');
      expect(mockRedis.del).toHaveBeenCalledWith(...keys);
      expect(result).toBe(true);
    });
  });

  describe('getStats', () => {
    it('should return cache statistics', () => {
      // Simulate some cache operations
      (cacheService as any).stats = {
        hits: 10,
        misses: 5,
        sets: 8,
        deletes: 2,
        errors: 1,
        totalRequests: 15,
        avgResponseTime: 25.5,
      };

      const stats = cacheService.getStats();

      expect(stats).toEqual({
        hits: 10,
        misses: 5,
        sets: 8,
        deletes: 2,
        errors: 1,
        totalRequests: 15,
        avgResponseTime: 25.5,
        hitRate: 66.67, // 10/(10+5) * 100
      });
    });
  });

  describe('cleanup', () => {
    it('should clean expired entries', () => {
      const now = Date.now();
      const cache = new Map();
      
      // Add expired and valid entries
      cache.set('expired', { data: 'test', expiry: now - 1000 });
      cache.set('valid', { data: 'test', expiry: now + 1000 });
      
      (cacheService as any).cache = cache;
      cacheService.cleanup();

      expect(cache.has('expired')).toBe(false);
      expect(cache.has('valid')).toBe(true);
    });
  });
});
