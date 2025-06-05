/**
 * Performance Monitor Hook Tests
 * Tests for the usePerformanceMonitor hook
 */

import { renderHook, act } from '@testing-library/react';
import { usePerformanceMonitor } from '../../lib/hooks/usePerformanceMonitor';

// Mock performance API
const mockPerformance = {
  now: jest.fn(() => 1000),
  mark: jest.fn(),
  measure: jest.fn(),
  getEntriesByType: jest.fn(() => []),
  getEntriesByName: jest.fn(() => []),
  clearMarks: jest.fn(),
  clearMeasures: jest.fn(),
};

Object.defineProperty(global, 'performance', {
  value: mockPerformance,
  writable: true,
});

// Mock PerformanceObserver
global.PerformanceObserver = jest.fn().mockImplementation((callback) => ({
  observe: jest.fn(),
  disconnect: jest.fn(),
}));

describe('usePerformanceMonitor', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockPerformance.now.mockReturnValue(1000);
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('should initialize with default values', () => {
    const { result } = renderHook(() => usePerformanceMonitor('TestComponent'));

    expect(result.current.metrics).toBeNull();
    expect(result.current.componentMetrics).toEqual({
      renderTime: 0,
      rerenderCount: 0,
      lastRenderTime: 0,
    });
    expect(typeof result.current.logPerformance).toBe('function');
    expect(typeof result.current.measureAsync).toBe('function');
    expect(typeof result.current.getPerformanceScore).toBe('function');
  });

  it('should track component render times', () => {
    const { result, rerender } = renderHook(() => usePerformanceMonitor('TestComponent'));

    // Simulate render time
    mockPerformance.now.mockReturnValueOnce(1000).mockReturnValueOnce(1050);

    act(() => {
      rerender();
    });

    // Component metrics should be updated after rerender
    expect(result.current.componentMetrics.rerenderCount).toBeGreaterThan(0);
  });

  it('should log performance correctly', () => {
    const consoleSpy = jest.spyOn(console, 'log').mockImplementation();
    const { result } = renderHook(() => usePerformanceMonitor('TestComponent'));

    const startTime = 1000;
    mockPerformance.now.mockReturnValue(1100);

    act(() => {
      const duration = result.current.logPerformance('test-action', startTime);
      expect(duration).toBe(100);
    });

    consoleSpy.mockRestore();
  });

  it('should measure async operations', async () => {
    const { result } = renderHook(() => usePerformanceMonitor('TestComponent'));

    const mockAsyncOperation = jest.fn().mockResolvedValue('test-result');
    mockPerformance.now.mockReturnValueOnce(1000).mockReturnValueOnce(1200);

    await act(async () => {
      const operationResult = await result.current.measureAsync(
        mockAsyncOperation,
        'test-async-operation'
      );
      expect(operationResult).toBe('test-result');
      expect(mockAsyncOperation).toHaveBeenCalled();
    });
  });

  it('should handle async operation errors', async () => {
    const { result } = renderHook(() => usePerformanceMonitor('TestComponent'));

    const mockAsyncOperation = jest.fn().mockRejectedValue(new Error('Test error'));

    await act(async () => {
      await expect(
        result.current.measureAsync(mockAsyncOperation, 'failing-operation')
      ).rejects.toThrow('Test error');
    });
  });

  it('should calculate performance score', () => {
    const { result } = renderHook(() => usePerformanceMonitor('TestComponent'));

    // Mock good performance metrics
    result.current.metrics = {
      pageLoadTime: 1000,
      firstContentfulPaint: 800,
      largestContentfulPaint: 1200,
      cumulativeLayoutShift: 0.05,
      firstInputDelay: 50,
      timeToInteractive: 1500,
    };

    act(() => {
      const score = result.current.getPerformanceScore();
      expect(score).toBeGreaterThan(80); // Good performance should score high
    });
  });

  it('should calculate low performance score for poor metrics', () => {
    const { result } = renderHook(() => usePerformanceMonitor('TestComponent'));

    // Mock poor performance metrics
    result.current.metrics = {
      pageLoadTime: 5000,
      firstContentfulPaint: 3000,
      largestContentfulPaint: 4000,
      cumulativeLayoutShift: 0.5,
      firstInputDelay: 500,
      timeToInteractive: 6000,
    };

    act(() => {
      const score = result.current.getPerformanceScore();
      expect(score).toBeLessThan(50); // Poor performance should score low
    });
  });

  it('should warn about slow renders in development', () => {
    const originalEnv = process.env.NODE_ENV;
    process.env.NODE_ENV = 'development';
    
    const consoleWarnSpy = jest.spyOn(console, 'warn').mockImplementation();
    
    // Mock slow render (>16ms)
    mockPerformance.now.mockReturnValueOnce(1000).mockReturnValueOnce(1020);

    const { unmount } = renderHook(() => usePerformanceMonitor('SlowComponent'));
    
    act(() => {
      unmount();
    });

    expect(consoleWarnSpy).toHaveBeenCalledWith(
      expect.stringContaining('Slow render detected in SlowComponent')
    );

    consoleWarnSpy.mockRestore();
    process.env.NODE_ENV = originalEnv;
  });

  it('should not warn about slow renders in production', () => {
    const originalEnv = process.env.NODE_ENV;
    process.env.NODE_ENV = 'production';
    
    const consoleWarnSpy = jest.spyOn(console, 'warn').mockImplementation();
    
    // Mock slow render (>16ms)
    mockPerformance.now.mockReturnValueOnce(1000).mockReturnValueOnce(1020);

    const { unmount } = renderHook(() => usePerformanceMonitor('SlowComponent'));
    
    act(() => {
      unmount();
    });

    expect(consoleWarnSpy).not.toHaveBeenCalled();

    consoleWarnSpy.mockRestore();
    process.env.NODE_ENV = originalEnv;
  });
});

describe('useApiPerformance', () => {
  it('should measure API call performance', async () => {
    const { useApiPerformance } = require('../../lib/hooks/usePerformanceMonitor');
    const { result } = renderHook(() => useApiPerformance());

    const mockApiCall = jest.fn().mockResolvedValue({ data: 'test' });
    mockPerformance.now.mockReturnValueOnce(1000).mockReturnValueOnce(1100);

    await act(async () => {
      const apiResult = await result.current.measureApiCall(mockApiCall, '/api/test');
      expect(apiResult).toEqual({ data: 'test' });
      expect(result.current.apiMetrics['/api/test']).toBe(100);
    });
  });

  it('should handle API call errors', async () => {
    const { useApiPerformance } = require('../../lib/hooks/usePerformanceMonitor');
    const { result } = renderHook(() => useApiPerformance());

    const mockApiCall = jest.fn().mockRejectedValue(new Error('API Error'));
    const consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation();

    await act(async () => {
      await expect(
        result.current.measureApiCall(mockApiCall, '/api/error')
      ).rejects.toThrow('API Error');
    });

    expect(consoleErrorSpy).toHaveBeenCalledWith(
      expect.stringContaining('API call failed: /api/error'),
      expect.any(Error)
    );

    consoleErrorSpy.mockRestore();
  });
});

describe('useMemoryMonitor', () => {
  it('should monitor memory usage', () => {
    const { useMemoryMonitor } = require('../../lib/hooks/usePerformanceMonitor');
    
    // Mock performance.memory
    Object.defineProperty(performance, 'memory', {
      value: {
        usedJSHeapSize: 30 * 1024 * 1024, // 30MB
        totalJSHeapSize: 50 * 1024 * 1024, // 50MB
        jsHeapSizeLimit: 100 * 1024 * 1024, // 100MB
      },
      configurable: true,
    });

    const { result } = renderHook(() => useMemoryMonitor());

    expect(result.current.memoryInfo).toBeDefined();
    expect(typeof result.current.checkMemoryUsage).toBe('function');
  });

  it('should warn about high memory usage', () => {
    const { useMemoryMonitor } = require('../../lib/hooks/usePerformanceMonitor');
    
    // Mock high memory usage
    Object.defineProperty(performance, 'memory', {
      value: {
        usedJSHeapSize: 60 * 1024 * 1024, // 60MB (high)
        totalJSHeapSize: 80 * 1024 * 1024,
        jsHeapSizeLimit: 100 * 1024 * 1024,
      },
      configurable: true,
    });

    const consoleWarnSpy = jest.spyOn(console, 'warn').mockImplementation();

    const { result } = renderHook(() => useMemoryMonitor());

    act(() => {
      result.current.checkMemoryUsage();
    });

    expect(consoleWarnSpy).toHaveBeenCalledWith(
      'High memory usage detected:',
      expect.objectContaining({
        used: expect.stringContaining('MB'),
        total: expect.stringContaining('MB'),
        limit: expect.stringContaining('MB'),
      })
    );

    consoleWarnSpy.mockRestore();
  });
});
