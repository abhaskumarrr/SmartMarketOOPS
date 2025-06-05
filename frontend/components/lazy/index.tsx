/**
 * Lazy-loaded Components Index
 * Centralized exports for all lazy-loaded components
 */

import React from 'react';
import { createLazyComponent, ChartLoadingSkeleton, LoadingSkeleton, TableLoadingSkeleton } from '../../lib/utils/lazyLoading';

// Chart Components
export const LazyTradingViewChart = createLazyComponent(
  () => import('../charts/TradingViewChart'),
  {
    fallback: ChartLoadingSkeleton,
  }
);

export const LazyEnhancedTradingViewChart = createLazyComponent(
  () => import('../charts/EnhancedTradingViewChart'),
  {
    fallback: ChartLoadingSkeleton,
  }
);

export const LazyLightweightChart = createLazyComponent(
  () => import('../charts/LightweightChart'),
  {
    fallback: ChartLoadingSkeleton,
  }
);

// Bot Management Components
export const LazyBotManagementDashboard = createLazyComponent(
  () => import('../bots/BotManagementDashboard'),
  {
    fallback: () => <LoadingSkeleton height={500} />,
  }
);

export const LazyEnhancedBotDashboard = createLazyComponent(
  () => import('../bots/EnhancedBotDashboard'),
  {
    fallback: () => <LoadingSkeleton height={600} />,
  }
);

export const LazyBotPerformanceMonitor = createLazyComponent(
  () => import('../bots/BotPerformanceMonitor'),
  {
    fallback: () => <LoadingSkeleton height={500} />,
  }
);

export const LazyBacktestingFramework = createLazyComponent(
  () => import('../bots/BacktestingFramework'),
  {
    fallback: () => <LoadingSkeleton height={600} />,
  }
);

// Trading Components
export const LazyRealTimeTradingDashboard = createLazyComponent(
  () => import('../trading/RealTimeTradingDashboard'),
  {
    fallback: () => <LoadingSkeleton height={600} />,
  }
);

// Monitoring Components
export const LazyEnhancedMonitoringDashboard = createLazyComponent(
  () => import('../monitoring/EnhancedMonitoringDashboard'),
  {
    fallback: () => <LoadingSkeleton height={600} />,
  }
);

// Intelligence Components
export const LazyMLIntelligenceDashboard = createLazyComponent(
  () => import('../intelligence/MLIntelligenceDashboard'),
  {
    fallback: () => <LoadingSkeleton height={600} />,
  }
);

// Settings Components
export const LazyNotificationSettings = createLazyComponent(
  () => import('../settings/NotificationSettings'),
  {
    fallback: () => <LoadingSkeleton height={300} />,
  }
);

export const LazyApiKeyManagement = createLazyComponent(
  () => import('../settings/ApiKeyManagement'),
  {
    fallback: () => <LoadingSkeleton height={400} />,
  }
);

// Preload critical components
export const preloadCriticalComponents = () => {
  // Preload components that are likely to be needed soon
  import('../charts/TradingViewChart');
  import('../bots/BotManagementDashboard');
  import('../trading/RealTimeTradingDashboard');
};

// Preload components based on route
export const preloadRouteComponents = (route: string) => {
  switch (route) {
    case '/dashboard':
      import('../trading/RealTimeTradingDashboard');
      import('../charts/TradingViewChart');
      break;
    case '/bots':
      import('../bots/BotManagementDashboard');
      import('../bots/EnhancedBotDashboard');
      break;
    case '/trading':
      import('../trading/RealTimeTradingDashboard');
      import('../charts/TradingViewChart');
      break;
    case '/monitoring':
      import('../monitoring/EnhancedMonitoringDashboard');
      break;
    default:
      break;
  }
};
