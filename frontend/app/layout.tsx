/**
 * SmartMarketOOPS Professional Trading Platform
 * Unified layout with consistent theming across all pages
 */

'use client';

import './globals.css';
import { Inter } from 'next/font/google';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline } from '@mui/material';
import { useState, useEffect } from 'react';

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

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return null;
  }

  return (
    <html lang="en" className="dark">
      <head>
        <title>SmartMarketOOPS - Professional Trading Platform</title>
        <meta name="description" content="Advanced AI-powered trading platform with real-time market data and Delta Exchange integration" />
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
      </body>
    </html>
  );
}
