/**
 * Performance Monitoring and Optimization Utilities
 * Tracks performance metrics and provides optimization helpers
 */

// Performance metrics interface
export interface PerformanceMetrics {
  pageLoadTime: number
  apiResponseTimes: Record<string, number[]>
  componentRenderTimes: Record<string, number[]>
  memoryUsage: number
  connectionLatency: number
  errorCount: number
  timestamp: number
}

// Performance monitor class
class PerformanceMonitor {
  private metrics: PerformanceMetrics = {
    pageLoadTime: 0,
    apiResponseTimes: {},
    componentRenderTimes: {},
    memoryUsage: 0,
    connectionLatency: 0,
    errorCount: 0,
    timestamp: Date.now(),
  }

  private observers: PerformanceObserver[] = []

  constructor() {
    this.initializeObservers()
    this.measurePageLoad()
  }

  private initializeObservers() {
    // Measure navigation timing
    if ('PerformanceObserver' in window) {
      try {
        const navObserver = new PerformanceObserver((list) => {
          const entries = list.getEntries()
          entries.forEach((entry) => {
            if (entry.entryType === 'navigation') {
              const navEntry = entry as PerformanceNavigationTiming
              this.metrics.pageLoadTime = navEntry.loadEventEnd - navEntry.navigationStart
            }
          })
        })
        navObserver.observe({ entryTypes: ['navigation'] })
        this.observers.push(navObserver)
      } catch (error) {
        console.warn('Navigation timing observer not supported:', error)
      }

      // Measure resource timing
      try {
        const resourceObserver = new PerformanceObserver((list) => {
          const entries = list.getEntries()
          entries.forEach((entry) => {
            if (entry.entryType === 'resource') {
              const resourceEntry = entry as PerformanceResourceTiming
              if (resourceEntry.name.includes('/api/')) {
                const apiName = this.extractApiName(resourceEntry.name)
                const responseTime = resourceEntry.responseEnd - resourceEntry.requestStart
                
                if (!this.metrics.apiResponseTimes[apiName]) {
                  this.metrics.apiResponseTimes[apiName] = []
                }
                this.metrics.apiResponseTimes[apiName].push(responseTime)
                
                // Keep only last 10 measurements
                if (this.metrics.apiResponseTimes[apiName].length > 10) {
                  this.metrics.apiResponseTimes[apiName].shift()
                }
              }
            }
          })
        })
        resourceObserver.observe({ entryTypes: ['resource'] })
        this.observers.push(resourceObserver)
      } catch (error) {
        console.warn('Resource timing observer not supported:', error)
      }
    }
  }

  private extractApiName(url: string): string {
    try {
      const urlObj = new URL(url)
      const pathParts = urlObj.pathname.split('/')
      return pathParts.slice(-2).join('/') // Get last two path segments
    } catch {
      return 'unknown'
    }
  }

  private measurePageLoad() {
    if (document.readyState === 'complete') {
      this.recordPageLoadTime()
    } else {
      window.addEventListener('load', () => {
        this.recordPageLoadTime()
      })
    }
  }

  private recordPageLoadTime() {
    if ('performance' in window && 'timing' in performance) {
      const timing = performance.timing
      this.metrics.pageLoadTime = timing.loadEventEnd - timing.navigationStart
    }
  }

  // Public methods
  public measureApiCall<T>(apiName: string, apiCall: () => Promise<T>): Promise<T> {
    const startTime = performance.now()
    
    return apiCall()
      .then((result) => {
        const endTime = performance.now()
        const responseTime = endTime - startTime
        
        if (!this.metrics.apiResponseTimes[apiName]) {
          this.metrics.apiResponseTimes[apiName] = []
        }
        this.metrics.apiResponseTimes[apiName].push(responseTime)
        
        // Keep only last 10 measurements
        if (this.metrics.apiResponseTimes[apiName].length > 10) {
          this.metrics.apiResponseTimes[apiName].shift()
        }
        
        return result
      })
      .catch((error) => {
        const endTime = performance.now()
        const responseTime = endTime - startTime
        
        if (!this.metrics.apiResponseTimes[apiName]) {
          this.metrics.apiResponseTimes[apiName] = []
        }
        this.metrics.apiResponseTimes[apiName].push(responseTime)
        this.metrics.errorCount++
        
        throw error
      })
  }

  public measureComponentRender(componentName: string, renderFn: () => void) {
    const startTime = performance.now()
    renderFn()
    const endTime = performance.now()
    const renderTime = endTime - startTime
    
    if (!this.metrics.componentRenderTimes[componentName]) {
      this.metrics.componentRenderTimes[componentName] = []
    }
    this.metrics.componentRenderTimes[componentName].push(renderTime)
    
    // Keep only last 10 measurements
    if (this.metrics.componentRenderTimes[componentName].length > 10) {
      this.metrics.componentRenderTimes[componentName].shift()
    }
  }

  public measureMemoryUsage() {
    if ('memory' in performance) {
      const memory = (performance as any).memory
      this.metrics.memoryUsage = memory.usedJSHeapSize / 1024 / 1024 // Convert to MB
    }
  }

  public measureConnectionLatency(wsUrl: string): Promise<number> {
    return new Promise((resolve, reject) => {
      const startTime = performance.now()
      const ws = new WebSocket(wsUrl)
      
      ws.onopen = () => {
        const latency = performance.now() - startTime
        this.metrics.connectionLatency = latency
        ws.close()
        resolve(latency)
      }
      
      ws.onerror = () => {
        ws.close()
        reject(new Error('WebSocket connection failed'))
      }
      
      // Timeout after 5 seconds
      setTimeout(() => {
        if (ws.readyState === WebSocket.CONNECTING) {
          ws.close()
          reject(new Error('WebSocket connection timeout'))
        }
      }, 5000)
    })
  }

  public getMetrics(): PerformanceMetrics {
    this.measureMemoryUsage()
    return { ...this.metrics, timestamp: Date.now() }
  }

  public getAverageApiResponseTime(apiName: string): number {
    const times = this.metrics.apiResponseTimes[apiName]
    if (!times || times.length === 0) return 0
    return times.reduce((sum, time) => sum + time, 0) / times.length
  }

  public getAverageComponentRenderTime(componentName: string): number {
    const times = this.metrics.componentRenderTimes[componentName]
    if (!times || times.length === 0) return 0
    return times.reduce((sum, time) => sum + time, 0) / times.length
  }

  public exportMetrics(): string {
    return JSON.stringify(this.getMetrics(), null, 2)
  }

  public cleanup() {
    this.observers.forEach(observer => observer.disconnect())
    this.observers = []
  }
}

// Singleton instance
export const performanceMonitor = new PerformanceMonitor()

// React hooks for performance monitoring
export function usePerformanceMonitor() {
  const [metrics, setMetrics] = React.useState<PerformanceMetrics | null>(null)

  React.useEffect(() => {
    const updateMetrics = () => {
      setMetrics(performanceMonitor.getMetrics())
    }

    updateMetrics()
    const interval = setInterval(updateMetrics, 5000) // Update every 5 seconds

    return () => clearInterval(interval)
  }, [])

  return metrics
}

// Debounce utility for performance optimization
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null

  return (...args: Parameters<T>) => {
    if (timeout) {
      clearTimeout(timeout)
    }
    timeout = setTimeout(() => func(...args), wait)
  }
}

// Throttle utility for performance optimization
export function throttle<T extends (...args: any[]) => any>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle: boolean = false

  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args)
      inThrottle = true
      setTimeout(() => (inThrottle = false), limit)
    }
  }
}

// Lazy loading utility
export function createLazyComponent<T extends React.ComponentType<any>>(
  importFn: () => Promise<{ default: T }>
) {
  return React.lazy(() => {
    const startTime = performance.now()
    return importFn().then((module) => {
      const loadTime = performance.now() - startTime
      console.log(`Component loaded in ${loadTime.toFixed(2)}ms`)
      return module
    })
  })
}

// Memory leak detection
export function detectMemoryLeaks() {
  if ('memory' in performance) {
    const memory = (performance as any).memory
    const usedMB = memory.usedJSHeapSize / 1024 / 1024
    const totalMB = memory.totalJSHeapSize / 1024 / 1024
    const limitMB = memory.jsHeapSizeLimit / 1024 / 1024

    console.log(`Memory usage: ${usedMB.toFixed(2)}MB / ${totalMB.toFixed(2)}MB (limit: ${limitMB.toFixed(2)}MB)`)

    if (usedMB > limitMB * 0.8) {
      console.warn('High memory usage detected! Possible memory leak.')
      return true
    }
  }
  return false
}

// Import React for hooks
import React from 'react'
