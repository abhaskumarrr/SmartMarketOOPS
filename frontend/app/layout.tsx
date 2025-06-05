/**
 * SmartMarketOOPS Professional Trading Platform
 * Unified layout with consistent theming and performance optimizations
 */

'use client';

import './globals.css';
import { Inter } from 'next/font/google';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline } from '@mui/material';
import { useState, useEffect } from 'react';
import { usePerformanceMonitor } from '../lib/hooks/usePerformanceMonitor';
import { registerServiceWorker, preloadResources } from '../lib/services/performanceService';
import { preloadCriticalComponents } from '../components/lazy/index';

const inter = Inter({ subsets: ['latin'] });

// Unified theme configuration
const unifiedTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#3b82f6',
      dark: '#1d4ed8',
    },
    secondary: {
      main: '#10b981',
      dark: '#059669',
    },
    background: {
      default: '#020617', // slate-950
      paper: '#0f172a',   // slate-900
    },
    text: {
      primary: '#f8fafc',   // slate-50
      secondary: '#cbd5e1', // slate-300
    },
    error: {
      main: '#ef4444',
      dark: '#dc2626',
    },
    warning: {
      main: '#f59e0b',
      dark: '#d97706',
    },
    success: {
      main: '#10b981',
      dark: '#059669',
    },
    divider: 'rgba(148, 163, 184, 0.1)', // slate-400 with opacity
  },
  typography: {
    fontFamily: inter.style.fontFamily,
    h1: {
      fontWeight: 700,
      color: '#f8fafc',
    },
    h2: {
      fontWeight: 600,
      color: '#f8fafc',
    },
    h3: {
      fontWeight: 600,
      color: '#f8fafc',
    },
    body1: {
      color: '#cbd5e1',
    },
    body2: {
      color: '#94a3b8',
    },
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          backgroundColor: '#020617',
          color: '#f8fafc',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backgroundColor: '#0f172a',
          border: '1px solid rgba(148, 163, 184, 0.1)',
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: '#0f172a',
          borderBottom: '1px solid rgba(148, 163, 184, 0.1)',
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundColor: '#0f172a',
          borderRight: '1px solid rgba(148, 163, 184, 0.1)',
        },
      },
    },
  },
});

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [mounted, setMounted] = useState(false);
  const { metrics, getPerformanceScore } = usePerformanceMonitor('RootLayout');

  useEffect(() => {
    setMounted(true);

    // Initialize performance optimizations
    const initializeOptimizations = async () => {
      // Register service worker for caching
      await registerServiceWorker();

      // Preload critical resources
      preloadResources([
        { href: '/manifest.json', as: 'manifest' },
        { href: '/_next/static/css/app/layout.css', as: 'style' },
        { href: '/_next/static/chunks/main.js', as: 'script' },
      ]);

      // Preload critical components
      preloadCriticalComponents();
    };

    initializeOptimizations();

    // Log performance metrics in development
    if (process.env.NODE_ENV === 'development') {
      setTimeout(() => {
        const score = getPerformanceScore();
        console.log('Performance Score:', score);
        console.log('Performance Metrics:', metrics);
      }, 3000);
    }
  }, []);

  if (!mounted) {
    return null;
  }

  return (
    <html lang="en" className="dark">
      <head>
        <title>SmartMarketOOPS - Professional Trading Platform</title>
        <meta name="description" content="Advanced AI-powered trading platform with real-time market data and Delta Exchange integration" />
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no" />
        <meta name="theme-color" content="#1976d2" />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
        <meta name="apple-mobile-web-app-title" content="SmartMarket" />

        {/* PWA Manifest */}
        <link rel="manifest" href="/manifest.json" />

        {/* Preconnect to external domains */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />

        {/* DNS prefetch for external resources */}
        <link rel="dns-prefetch" href="//api.delta.exchange" />
        <link rel="dns-prefetch" href="//cdn.jsdelivr.net" />

        {/* Favicon and icons */}
        <link rel="icon" href="/favicon.ico" />
        <link rel="apple-touch-icon" href="/icon-192x192.png" />

        {/* Critical CSS inlining would go here in production */}
        <style dangerouslySetInnerHTML={{
          __html: `
            /* Critical CSS for above-the-fold content */
            body { margin: 0; padding: 0; }
            .loading-spinner {
              display: flex;
              justify-content: center;
              align-items: center;
              height: 100vh;
              background: #020617;
            }
          `
        }} />
      </head>
      <body
        className={`${inter.className} bg-slate-950 text-white antialiased`}
        suppressHydrationWarning={true}
      >
        <ThemeProvider theme={unifiedTheme}>
          <CssBaseline />
          <div className="min-h-screen bg-slate-950">
            {children}
          </div>
        </ThemeProvider>

        {/* Performance monitoring script */}
        {process.env.NODE_ENV === 'production' && (
          <script
            dangerouslySetInnerHTML={{
              __html: `
                // Basic performance monitoring
                window.addEventListener('load', function() {
                  setTimeout(function() {
                    const perfData = performance.getEntriesByType('navigation')[0];
                    if (perfData && perfData.loadEventEnd > 2000) {
                      console.warn('Slow page load detected:', perfData.loadEventEnd + 'ms');
                    }
                  }, 0);
                });
              `
            }}
          />
        )}
      </body>
    </html>
  );
}
