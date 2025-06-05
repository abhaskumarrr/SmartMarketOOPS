/**
 * Lazy Loading Utilities
 * Provides optimized lazy loading for components and resources
 */

import React, { Suspense, ComponentType, lazy } from 'react';
import { CircularProgress, Box, Skeleton } from '@mui/material';

// Loading fallback components
export const LoadingSpinner: React.FC<{ size?: number }> = ({ size = 40 }) => (
  <Box
    display="flex"
    justifyContent="center"
    alignItems="center"
    minHeight="200px"
    width="100%"
  >
    <CircularProgress size={size} />
  </Box>
);

export const LoadingSkeleton: React.FC<{
  variant?: 'text' | 'rectangular' | 'circular';
  width?: number | string;
  height?: number | string;
  count?: number;
}> = ({ variant = 'rectangular', width = '100%', height = 200, count = 1 }) => (
  <Box sx={{ p: 2 }}>
    {Array.from({ length: count }).map((_, index) => (
      <Skeleton
        key={index}
        variant={variant}
        width={width}
        height={height}
        sx={{ mb: count > 1 ? 1 : 0 }}
      />
    ))}
  </Box>
);

export const ChartLoadingSkeleton: React.FC = () => (
  <Box sx={{ p: 2 }}>
    <Skeleton variant="text" width="60%" height={40} sx={{ mb: 2 }} />
    <Skeleton variant="rectangular" width="100%" height={300} sx={{ mb: 1 }} />
    <Box sx={{ display: 'flex', gap: 1 }}>
      <Skeleton variant="rectangular" width="20%" height={30} />
      <Skeleton variant="rectangular" width="20%" height={30} />
      <Skeleton variant="rectangular" width="20%" height={30} />
    </Box>
  </Box>
);

export const TableLoadingSkeleton: React.FC<{ rows?: number; columns?: number }> = ({
  rows = 5,
  columns = 4,
}) => (
  <Box sx={{ p: 2 }}>
    {Array.from({ length: rows }).map((_, rowIndex) => (
      <Box key={rowIndex} sx={{ display: 'flex', gap: 2, mb: 1 }}>
        {Array.from({ length: columns }).map((_, colIndex) => (
          <Skeleton
            key={colIndex}
            variant="text"
            width={`${100 / columns}%`}
            height={40}
          />
        ))}
      </Box>
    ))}
  </Box>
);

// Enhanced lazy loading with error boundaries
interface LazyComponentProps {
  fallback?: React.ComponentType;
  errorFallback?: React.ComponentType<{ error: Error; retry: () => void }>;
  retryCount?: number;
}

const DefaultErrorFallback: React.FC<{ error: Error; retry: () => void }> = ({
  error,
  retry,
}) => (
  <Box
    display="flex"
    flexDirection="column"
    alignItems="center"
    justifyContent="center"
    minHeight="200px"
    p={3}
  >
    <Box mb={2} textAlign="center">
      <h3>Failed to load component</h3>
      <p style={{ color: '#666', fontSize: '14px' }}>
        {error.message || 'An unexpected error occurred'}
      </p>
    </Box>
    <button
      onClick={retry}
      style={{
        padding: '8px 16px',
        backgroundColor: '#1976d2',
        color: 'white',
        border: 'none',
        borderRadius: '4px',
        cursor: 'pointer',
      }}
    >
      Retry
    </button>
  </Box>
);

class LazyErrorBoundary extends React.Component<
  {
    children: React.ReactNode;
    fallback: React.ComponentType<{ error: Error; retry: () => void }>;
    onRetry: () => void;
  },
  { hasError: boolean; error: Error | null }
> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Lazy component error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError && this.state.error) {
      const FallbackComponent = this.props.fallback;
      return (
        <FallbackComponent
          error={this.state.error}
          retry={() => {
            this.setState({ hasError: false, error: null });
            this.props.onRetry();
          }}
        />
      );
    }

    return this.props.children;
  }
}

export const createLazyComponent = <P extends object>(
  importFn: () => Promise<{ default: ComponentType<P> }>,
  options: LazyComponentProps = {}
) => {
  const {
    fallback: FallbackComponent = LoadingSpinner,
    errorFallback: ErrorFallbackComponent = DefaultErrorFallback,
    retryCount = 3,
  } = options;

  let retries = 0;

  const LazyComponent = lazy(() => {
    return importFn().catch((error) => {
      if (retries < retryCount) {
        retries++;
        console.warn(`Lazy loading failed, retrying (${retries}/${retryCount}):`, error);
        return new Promise((resolve) => {
          setTimeout(() => resolve(importFn()), 1000 * retries);
        });
      }
      throw error;
    });
  });

  const WrappedComponent: React.FC<P> = (props) => {
    const [key, setKey] = React.useState(0);

    return (
      <LazyErrorBoundary
        fallback={ErrorFallbackComponent}
        onRetry={() => {
          retries = 0;
          setKey(prev => prev + 1);
        }}
      >
        <Suspense key={key} fallback={<FallbackComponent />}>
          <LazyComponent {...props} />
        </Suspense>
      </LazyErrorBoundary>
    );
  };

  return WrappedComponent;
};

// Preload utility for critical components
export const preloadComponent = (importFn: () => Promise<any>) => {
  const componentImport = importFn();
  return componentImport;
};

// Intersection Observer based lazy loading for images and components
export const useLazyLoad = (threshold = 0.1) => {
  const [isVisible, setIsVisible] = React.useState(false);
  const [hasLoaded, setHasLoaded] = React.useState(false);
  const elementRef = React.useRef<HTMLElement>(null);

  React.useEffect(() => {
    const element = elementRef.current;
    if (!element || hasLoaded) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          setHasLoaded(true);
          observer.unobserve(element);
        }
      },
      { threshold }
    );

    observer.observe(element);

    return () => {
      if (element) {
        observer.unobserve(element);
      }
    };
  }, [threshold, hasLoaded]);

  return { elementRef, isVisible, hasLoaded };
};

// Lazy image component with progressive loading
export const LazyImage: React.FC<{
  src: string;
  alt: string;
  placeholder?: string;
  className?: string;
  style?: React.CSSProperties;
  onLoad?: () => void;
  onError?: () => void;
}> = ({ src, alt, placeholder, className, style, onLoad, onError }) => {
  const { elementRef, isVisible } = useLazyLoad();
  const [imageLoaded, setImageLoaded] = React.useState(false);
  const [imageError, setImageError] = React.useState(false);

  const handleLoad = () => {
    setImageLoaded(true);
    onLoad?.();
  };

  const handleError = () => {
    setImageError(true);
    onError?.();
  };

  return (
    <div ref={elementRef} className={className} style={style}>
      {isVisible && !imageError && (
        <img
          src={src}
          alt={alt}
          onLoad={handleLoad}
          onError={handleError}
          style={{
            opacity: imageLoaded ? 1 : 0,
            transition: 'opacity 0.3s ease',
            ...style,
          }}
        />
      )}
      {(!isVisible || !imageLoaded) && !imageError && placeholder && (
        <img
          src={placeholder}
          alt={`${alt} placeholder`}
          style={{
            filter: 'blur(5px)',
            transition: 'filter 0.3s ease',
            ...style,
          }}
        />
      )}
      {imageError && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: '#f5f5f5',
            color: '#666',
            ...style,
          }}
        >
          Failed to load image
        </div>
      )}
    </div>
  );
};

// Resource preloading utilities
export const preloadResource = (href: string, as: string) => {
  if (typeof window !== 'undefined') {
    const link = document.createElement('link');
    link.rel = 'preload';
    link.href = href;
    link.as = as;
    document.head.appendChild(link);
  }
};

export const preloadScript = (src: string) => {
  if (typeof window !== 'undefined') {
    const link = document.createElement('link');
    link.rel = 'preload';
    link.href = src;
    link.as = 'script';
    document.head.appendChild(link);
  }
};

export const preloadFont = (href: string) => {
  if (typeof window !== 'undefined') {
    const link = document.createElement('link');
    link.rel = 'preload';
    link.href = href;
    link.as = 'font';
    link.type = 'font/woff2';
    link.crossOrigin = 'anonymous';
    document.head.appendChild(link);
  }
};
