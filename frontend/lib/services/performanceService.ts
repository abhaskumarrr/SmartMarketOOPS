/**
 * Performance Optimization Service
 * Handles caching, debouncing, throttling, and other performance optimizations
 */

// Cache implementation with TTL
class MemoryCache {
  private cache = new Map<string, { data: any; expiry: number }>();
  private maxSize: number;

  constructor(maxSize = 100) {
    this.maxSize = maxSize;
  }

  set(key: string, data: any, ttlMs = 300000): void { // 5 minutes default
    // Remove oldest entries if cache is full
    if (this.cache.size >= this.maxSize) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }

    this.cache.set(key, {
      data,
      expiry: Date.now() + ttlMs,
    });
  }

  get(key: string): any | null {
    const item = this.cache.get(key);
    
    if (!item) return null;
    
    if (Date.now() > item.expiry) {
      this.cache.delete(key);
      return null;
    }
    
    return item.data;
  }

  has(key: string): boolean {
    const item = this.cache.get(key);
    if (!item) return false;
    
    if (Date.now() > item.expiry) {
      this.cache.delete(key);
      return false;
    }
    
    return true;
  }

  delete(key: string): boolean {
    return this.cache.delete(key);
  }

  clear(): void {
    this.cache.clear();
  }

  size(): number {
    return this.cache.size;
  }

  // Clean expired entries
  cleanup(): void {
    const now = Date.now();
    for (const [key, item] of this.cache.entries()) {
      if (now > item.expiry) {
        this.cache.delete(key);
      }
    }
  }
}

// Global cache instance
export const cache = new MemoryCache(200);

// Cleanup expired cache entries every 5 minutes
if (typeof window !== 'undefined') {
  setInterval(() => cache.cleanup(), 300000);
}

// Debounce utility
export const debounce = <T extends (...args: any[]) => any>(
  func: T,
  delay: number
): ((...args: Parameters<T>) => void) => {
  let timeoutId: NodeJS.Timeout;
  
  return (...args: Parameters<T>) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => func(...args), delay);
  };
};

// Throttle utility
export const throttle = <T extends (...args: any[]) => any>(
  func: T,
  limit: number
): ((...args: Parameters<T>) => void) => {
  let inThrottle: boolean;
  
  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args);
      inThrottle = true;
      setTimeout(() => (inThrottle = false), limit);
    }
  };
};

// Memoization with cache
export const memoize = <T extends (...args: any[]) => any>(
  func: T,
  keyGenerator?: (...args: Parameters<T>) => string,
  ttlMs = 300000
): T => {
  const memoCache = new Map<string, { result: ReturnType<T>; expiry: number }>();

  return ((...args: Parameters<T>): ReturnType<T> => {
    const key = keyGenerator ? keyGenerator(...args) : JSON.stringify(args);
    const cached = memoCache.get(key);
    
    if (cached && Date.now() < cached.expiry) {
      return cached.result;
    }
    
    const result = func(...args);
    memoCache.set(key, {
      result,
      expiry: Date.now() + ttlMs,
    });
    
    return result;
  }) as T;
};

// API response caching
export const createCachedApiCall = <T>(
  apiCall: (...args: any[]) => Promise<T>,
  cacheKey: string,
  ttlMs = 300000
) => {
  return async (...args: any[]): Promise<T> => {
    const key = `${cacheKey}_${JSON.stringify(args)}`;
    
    // Check cache first
    const cached = cache.get(key);
    if (cached) {
      return cached;
    }
    
    // Make API call and cache result
    try {
      const result = await apiCall(...args);
      cache.set(key, result, ttlMs);
      return result;
    } catch (error) {
      // Don't cache errors
      throw error;
    }
  };
};

// Batch API calls
export class BatchProcessor<T, R> {
  private batch: T[] = [];
  private batchTimeout: NodeJS.Timeout | null = null;
  private readonly batchSize: number;
  private readonly batchDelay: number;
  private readonly processor: (batch: T[]) => Promise<R[]>;

  constructor(
    processor: (batch: T[]) => Promise<R[]>,
    batchSize = 10,
    batchDelay = 100
  ) {
    this.processor = processor;
    this.batchSize = batchSize;
    this.batchDelay = batchDelay;
  }

  add(item: T): Promise<R> {
    return new Promise((resolve, reject) => {
      this.batch.push({ ...item, resolve, reject } as any);
      
      if (this.batch.length >= this.batchSize) {
        this.processBatch();
      } else if (!this.batchTimeout) {
        this.batchTimeout = setTimeout(() => this.processBatch(), this.batchDelay);
      }
    });
  }

  private async processBatch(): Promise<void> {
    if (this.batch.length === 0) return;
    
    const currentBatch = this.batch.splice(0, this.batchSize);
    
    if (this.batchTimeout) {
      clearTimeout(this.batchTimeout);
      this.batchTimeout = null;
    }
    
    try {
      const results = await this.processor(currentBatch.map(item => {
        const { resolve, reject, ...data } = item as any;
        return data;
      }));
      
      currentBatch.forEach((item: any, index) => {
        item.resolve(results[index]);
      });
    } catch (error) {
      currentBatch.forEach((item: any) => {
        item.reject(error);
      });
    }
    
    // Process remaining items if any
    if (this.batch.length > 0) {
      this.batchTimeout = setTimeout(() => this.processBatch(), this.batchDelay);
    }
  }
}

// Image optimization
export const optimizeImageUrl = (
  url: string,
  width?: number,
  height?: number,
  quality = 80
): string => {
  if (!url) return '';
  
  // For Next.js Image optimization
  if (url.startsWith('/') || url.includes(window.location.hostname)) {
    const params = new URLSearchParams();
    if (width) params.set('w', width.toString());
    if (height) params.set('h', height.toString());
    params.set('q', quality.toString());
    
    return `/_next/image?url=${encodeURIComponent(url)}&${params.toString()}`;
  }
  
  return url;
};

// Virtual scrolling helper
export const calculateVisibleRange = (
  scrollTop: number,
  containerHeight: number,
  itemHeight: number,
  totalItems: number,
  overscan = 5
): { start: number; end: number } => {
  const start = Math.max(0, Math.floor(scrollTop / itemHeight) - overscan);
  const visibleCount = Math.ceil(containerHeight / itemHeight);
  const end = Math.min(totalItems - 1, start + visibleCount + overscan * 2);
  
  return { start, end };
};

// Resource preloading
export const preloadResources = (resources: Array<{ href: string; as: string; type?: string }>) => {
  if (typeof window === 'undefined') return;
  
  resources.forEach(({ href, as, type }) => {
    const link = document.createElement('link');
    link.rel = 'preload';
    link.href = href;
    link.as = as;
    if (type) link.type = type;
    if (as === 'font') link.crossOrigin = 'anonymous';
    document.head.appendChild(link);
  });
};

// Service Worker registration for caching
export const registerServiceWorker = async (): Promise<void> => {
  if (typeof window === 'undefined' || !('serviceWorker' in navigator)) {
    return;
  }
  
  try {
    const registration = await navigator.serviceWorker.register('/sw.js');
    console.log('Service Worker registered:', registration);
    
    // Update service worker when new version is available
    registration.addEventListener('updatefound', () => {
      const newWorker = registration.installing;
      if (newWorker) {
        newWorker.addEventListener('statechange', () => {
          if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
            // New version available, prompt user to refresh
            if (confirm('New version available! Refresh to update?')) {
              window.location.reload();
            }
          }
        });
      }
    });
  } catch (error) {
    console.error('Service Worker registration failed:', error);
  }
};

// Performance monitoring
export const performanceMonitor = {
  mark: (name: string) => {
    if (typeof window !== 'undefined' && 'performance' in window) {
      performance.mark(name);
    }
  },
  
  measure: (name: string, startMark: string, endMark?: string) => {
    if (typeof window !== 'undefined' && 'performance' in window) {
      try {
        performance.measure(name, startMark, endMark);
        const measure = performance.getEntriesByName(name, 'measure')[0];
        return measure.duration;
      } catch (error) {
        console.warn('Performance measurement failed:', error);
        return 0;
      }
    }
    return 0;
  },
  
  clearMarks: (name?: string) => {
    if (typeof window !== 'undefined' && 'performance' in window) {
      performance.clearMarks(name);
    }
  },
  
  clearMeasures: (name?: string) => {
    if (typeof window !== 'undefined' && 'performance' in window) {
      performance.clearMeasures(name);
    }
  },
};

// Export performance utilities
export const performanceUtils = {
  cache,
  debounce,
  throttle,
  memoize,
  createCachedApiCall,
  BatchProcessor,
  optimizeImageUrl,
  calculateVisibleRange,
  preloadResources,
  registerServiceWorker,
  performanceMonitor,
};
