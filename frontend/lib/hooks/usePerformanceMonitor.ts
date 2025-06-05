/**
 * Performance Monitoring Hook
 * Tracks page load times, component render times, and user interactions
 */

import { useEffect, useRef, useState } from 'react';

interface PerformanceMetrics {
  pageLoadTime: number;
  firstContentfulPaint: number;
  largestContentfulPaint: number;
  cumulativeLayoutShift: number;
  firstInputDelay: number;
  timeToInteractive: number;
}

interface ComponentMetrics {
  renderTime: number;
  rerenderCount: number;
  lastRenderTime: number;
}

export const usePerformanceMonitor = (componentName?: string) => {
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [componentMetrics, setComponentMetrics] = useState<ComponentMetrics>({
    renderTime: 0,
    rerenderCount: 0,
    lastRenderTime: 0,
  });
  const renderStartTime = useRef<number>(0);
  const rerenderCount = useRef<number>(0);

  // Track component render performance
  useEffect(() => {
    renderStartTime.current = performance.now();
    rerenderCount.current += 1;

    return () => {
      const renderTime = performance.now() - renderStartTime.current;
      setComponentMetrics(prev => ({
        renderTime: prev.renderTime + renderTime,
        rerenderCount: rerenderCount.current,
        lastRenderTime: renderTime,
      }));

      // Log slow renders in development
      if (process.env.NODE_ENV === 'development' && renderTime > 16) {
        console.warn(
          `Slow render detected in ${componentName || 'Unknown Component'}: ${renderTime.toFixed(2)}ms`
        );
      }
    };
  });

  // Track page performance metrics
  useEffect(() => {
    const measurePerformance = () => {
      try {
        const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
        const paint = performance.getEntriesByType('paint');
        
        const pageLoadTime = navigation.loadEventEnd - navigation.fetchStart;
        const firstContentfulPaint = paint.find(entry => entry.name === 'first-contentful-paint')?.startTime || 0;

        // Get Web Vitals
        const observer = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            if (entry.entryType === 'largest-contentful-paint') {
              setMetrics(prev => ({
                ...prev!,
                largestContentfulPaint: entry.startTime,
              }));
            }
            
            if (entry.entryType === 'layout-shift' && !(entry as any).hadRecentInput) {
              setMetrics(prev => ({
                ...prev!,
                cumulativeLayoutShift: (prev?.cumulativeLayoutShift || 0) + (entry as any).value,
              }));
            }
            
            if (entry.entryType === 'first-input') {
              setMetrics(prev => ({
                ...prev!,
                firstInputDelay: (entry as any).processingStart - entry.startTime,
              }));
            }
          }
        });

        observer.observe({ entryTypes: ['largest-contentful-paint', 'layout-shift', 'first-input'] });

        setMetrics({
          pageLoadTime,
          firstContentfulPaint,
          largestContentfulPaint: 0,
          cumulativeLayoutShift: 0,
          firstInputDelay: 0,
          timeToInteractive: 0,
        });

        // Cleanup observer after 10 seconds
        setTimeout(() => observer.disconnect(), 10000);
      } catch (error) {
        console.warn('Performance monitoring not supported:', error);
      }
    };

    // Wait for page to load
    if (document.readyState === 'complete') {
      measurePerformance();
    } else {
      window.addEventListener('load', measurePerformance);
      return () => window.removeEventListener('load', measurePerformance);
    }
  }, []);

  // Performance logging utility
  const logPerformance = (action: string, startTime: number) => {
    const duration = performance.now() - startTime;
    
    if (process.env.NODE_ENV === 'development') {
      console.log(`Performance: ${action} took ${duration.toFixed(2)}ms`);
    }

    // Send to analytics in production
    if (process.env.NODE_ENV === 'production' && duration > 1000) {
      // You can integrate with analytics services here
      console.warn(`Slow operation: ${action} took ${duration.toFixed(2)}ms`);
    }

    return duration;
  };

  // Measure async operations
  const measureAsync = async <T>(
    operation: () => Promise<T>,
    operationName: string
  ): Promise<T> => {
    const startTime = performance.now();
    try {
      const result = await operation();
      logPerformance(operationName, startTime);
      return result;
    } catch (error) {
      logPerformance(`${operationName} (failed)`, startTime);
      throw error;
    }
  };

  // Get performance score
  const getPerformanceScore = (): number => {
    if (!metrics) return 0;

    let score = 100;
    
    // Page load time scoring (target: <2s)
    if (metrics.pageLoadTime > 2000) score -= 20;
    else if (metrics.pageLoadTime > 1000) score -= 10;
    
    // First Contentful Paint scoring (target: <1.8s)
    if (metrics.firstContentfulPaint > 1800) score -= 15;
    else if (metrics.firstContentfulPaint > 1000) score -= 8;
    
    // Largest Contentful Paint scoring (target: <2.5s)
    if (metrics.largestContentfulPaint > 2500) score -= 20;
    else if (metrics.largestContentfulPaint > 1500) score -= 10;
    
    // Cumulative Layout Shift scoring (target: <0.1)
    if (metrics.cumulativeLayoutShift > 0.25) score -= 15;
    else if (metrics.cumulativeLayoutShift > 0.1) score -= 8;
    
    // First Input Delay scoring (target: <100ms)
    if (metrics.firstInputDelay > 300) score -= 15;
    else if (metrics.firstInputDelay > 100) score -= 8;

    return Math.max(0, score);
  };

  return {
    metrics,
    componentMetrics,
    logPerformance,
    measureAsync,
    getPerformanceScore,
  };
};

// Performance monitoring for API calls
export const useApiPerformance = () => {
  const [apiMetrics, setApiMetrics] = useState<Record<string, number>>({});

  const measureApiCall = async <T>(
    apiCall: () => Promise<T>,
    endpoint: string
  ): Promise<T> => {
    const startTime = performance.now();
    
    try {
      const result = await apiCall();
      const duration = performance.now() - startTime;
      
      setApiMetrics(prev => ({
        ...prev,
        [endpoint]: duration,
      }));

      // Log slow API calls
      if (duration > 1000) {
        console.warn(`Slow API call: ${endpoint} took ${duration.toFixed(2)}ms`);
      }

      return result;
    } catch (error) {
      const duration = performance.now() - startTime;
      console.error(`API call failed: ${endpoint} took ${duration.toFixed(2)}ms`, error);
      throw error;
    }
  };

  return {
    apiMetrics,
    measureApiCall,
  };
};

// Memory usage monitoring
export const useMemoryMonitor = () => {
  const [memoryInfo, setMemoryInfo] = useState<any>(null);

  useEffect(() => {
    const updateMemoryInfo = () => {
      if ('memory' in performance) {
        setMemoryInfo((performance as any).memory);
      }
    };

    updateMemoryInfo();
    const interval = setInterval(updateMemoryInfo, 5000);

    return () => clearInterval(interval);
  }, []);

  const checkMemoryUsage = () => {
    if (memoryInfo && memoryInfo.usedJSHeapSize > 50 * 1024 * 1024) { // 50MB
      console.warn('High memory usage detected:', {
        used: `${(memoryInfo.usedJSHeapSize / 1024 / 1024).toFixed(2)}MB`,
        total: `${(memoryInfo.totalJSHeapSize / 1024 / 1024).toFixed(2)}MB`,
        limit: `${(memoryInfo.jsHeapSizeLimit / 1024 / 1024).toFixed(2)}MB`,
      });
    }
  };

  useEffect(() => {
    checkMemoryUsage();
  }, [memoryInfo]);

  return {
    memoryInfo,
    checkMemoryUsage,
  };
};
