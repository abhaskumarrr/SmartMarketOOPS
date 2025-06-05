/**
 * Advanced Caching Service
 * Redis-based caching with multiple strategies and performance optimization
 */
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
    ttl?: number;
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
declare class CacheService {
    private redis;
    private stats;
    private isConnected;
    private compressionThreshold;
    constructor(config: CacheConfig);
    private setupEventHandlers;
    private measureOperation;
    private updateStats;
    private compressData;
    private decompressData;
    private buildKey;
    get<T>(key: string, options?: CacheOptions): Promise<T | null>;
    set<T>(key: string, value: T, options?: CacheOptions): Promise<boolean>;
    del(key: string, namespace?: string): Promise<boolean>;
    invalidateByTag(tag: string): Promise<number>;
    mget<T>(keys: string[], namespace?: string): Promise<(T | null)[]>;
    mset<T>(keyValuePairs: Array<{
        key: string;
        value: T;
        ttl?: number;
    }>, options?: CacheOptions): Promise<boolean>;
    exists(key: string, namespace?: string): Promise<boolean>;
    ttl(key: string, namespace?: string): Promise<number>;
    flush(namespace?: string): Promise<boolean>;
    getStats(): CacheStats;
    getInfo(): Promise<any>;
    disconnect(): Promise<void>;
}
export declare const createCacheService: (config: CacheConfig) => CacheService;
export declare const getCacheService: () => CacheService | null;
export { CacheService, CacheConfig, CacheOptions, CacheStats };
