/**
 * Unified Navigation Component
 * Consistent navigation across all pages (App Router and Pages Router)
 */

'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  AppBar,
  Toolbar,
  Typography,
  Box,
  Button,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemButton,
  Chip,
  useTheme,
  useMediaQuery,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  TrendingUp as TradingIcon,
  SmartToy as BotsIcon,
  Analytics as AnalyticsIcon,
  AccountBalance as PortfolioIcon,
  Settings as SettingsIcon,
  Menu as MenuIcon,
  FlashOn as ZapIcon,
  Circle,
} from '@mui/icons-material';

interface NavigationItem {
  label: string;
  href: string;
  icon: React.ReactNode;
  badge?: string;
  description?: string;
}

const navigationItems: NavigationItem[] = [
  {
    label: 'Dashboard',
    href: '/dashboard',
    icon: <DashboardIcon />,
    description: 'Overview & Analytics',
  },
  {
    label: 'Paper Trading',
    href: '/paper-trading',
    icon: <TradingIcon />,
    badge: 'Live',
    description: 'Delta Exchange Testnet',
  },
  {
    label: 'Trading Bots',
    href: '/bots',
    icon: <BotsIcon />,
    description: 'AI-Powered Automation',
  },
  {
    label: 'Portfolio',
    href: '/portfolio',
    icon: <PortfolioIcon />,
    description: 'Holdings & Performance',
  },
  {
    label: 'Analytics',
    href: '/analytics',
    icon: <AnalyticsIcon />,
    description: 'Market Intelligence',
  },
  {
    label: 'Settings',
    href: '/settings',
    icon: <SettingsIcon />,
    description: 'Platform Configuration',
  },
];

interface UnifiedNavigationProps {
  connectionStatus?: 'connected' | 'disconnected' | 'connecting';
  showConnectionStatus?: boolean;
}

export function UnifiedNavigation({ 
  connectionStatus = 'connected',
  showConnectionStatus = true 
}: UnifiedNavigationProps) {
  const pathname = usePathname();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [mobileOpen, setMobileOpen] = useState(false);

  const isActive = (href: string) => {
    if (href === '/dashboard') {
      return pathname === '/' || pathname === '/dashboard';
    }
    return pathname?.startsWith(href) || false;
  };

  const getConnectionColor = () => {
    switch (connectionStatus) {
      case 'connected':
        return 'success';
      case 'connecting':
        return 'warning';
      case 'disconnected':
        return 'error';
      default:
        return 'default';
    }
  };

  const getConnectionText = () => {
    switch (connectionStatus) {
      case 'connected':
        return 'Connected';
      case 'connecting':
        return 'Connecting...';
      case 'disconnected':
        return 'Disconnected';
      default:
        return 'Unknown';
    }
  };

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const drawer = (
    <Box sx={{ width: 280, height: '100%', bgcolor: 'background.paper' }}>
      {/* Logo Section */}
      <Box sx={{ p: 3, borderBottom: 1, borderColor: 'divider' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Box
            sx={{
              width: 40,
              height: 40,
              background: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
              borderRadius: 2,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <ZapIcon sx={{ color: 'white', fontSize: 24 }} />
          </Box>
          <Box>
            <Typography variant="h6" sx={{ fontWeight: 700, color: 'text.primary' }}>
              SmartMarket
            </Typography>
            <Typography variant="caption" sx={{ color: 'text.secondary' }}>
              Trading Platform
            </Typography>
          </Box>
        </Box>
      </Box>

      {/* Navigation Items */}
      <List sx={{ px: 2, py: 1 }}>
        {navigationItems.map((item) => (
          <ListItem key={item.href} disablePadding sx={{ mb: 0.5 }}>
            <ListItemButton
              component={Link}
              href={item.href}
              selected={isActive(item.href)}
              onClick={() => isMobile && setMobileOpen(false)}
              sx={{
                borderRadius: 2,
                '&.Mui-selected': {
                  bgcolor: 'primary.main',
                  color: 'primary.contrastText',
                  '&:hover': {
                    bgcolor: 'primary.dark',
                  },
                  '& .MuiListItemIcon-root': {
                    color: 'primary.contrastText',
                  },
                },
                '&:hover': {
                  bgcolor: 'action.hover',
                },
              }}
            >
              <ListItemIcon sx={{ minWidth: 40 }}>
                {item.icon}
              </ListItemIcon>
              <ListItemText
                primary={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography variant="body2" sx={{ fontWeight: 500 }}>
                      {item.label}
                    </Typography>
                    {item.badge && (
                      <Chip
                        label={item.badge}
                        size="small"
                        color="success"
                        sx={{ height: 20, fontSize: '0.7rem' }}
                      />
                    )}
                  </Box>
                }
                secondary={item.description}
                secondaryTypographyProps={{
                  variant: 'caption',
                  sx: { color: 'text.secondary', fontSize: '0.7rem' }
                }}
              />
            </ListItemButton>
          </ListItem>
        ))}
      </List>

      {/* Connection Status */}
      {showConnectionStatus && (
        <Box sx={{ p: 2, mt: 'auto', borderTop: 1, borderColor: 'divider' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
            <Circle
              sx={{
                fontSize: 8,
                color: `${getConnectionColor()}.main`,
                animation: connectionStatus === 'connected' ? 'pulse 2s infinite' : 'none',
              }}
            />
            <Typography variant="caption" sx={{ color: 'text.secondary' }}>
              {getConnectionText()}
            </Typography>
          </Box>
        </Box>
      )}
    </Box>
  );

  return (
    <>
      {/* Top App Bar */}
      <AppBar
        position="fixed"
        elevation={0}
        sx={{
          zIndex: theme.zIndex.drawer + 1,
          bgcolor: 'background.paper',
          borderBottom: 1,
          borderColor: 'divider',
        }}
      >
        <Toolbar>
          {/* Mobile Menu Button */}
          {isMobile && (
            <IconButton
              color="inherit"
              aria-label="open drawer"
              edge="start"
              onClick={handleDrawerToggle}
              sx={{ mr: 2 }}
            >
              <MenuIcon />
            </IconButton>
          )}

          {/* Desktop Logo */}
          {!isMobile && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mr: 4 }}>
              <Box
                sx={{
                  width: 32,
                  height: 32,
                  background: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
                  borderRadius: 1.5,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <ZapIcon sx={{ color: 'white', fontSize: 20 }} />
              </Box>
              <Typography variant="h6" sx={{ fontWeight: 700, color: 'text.primary' }}>
                SmartMarket
              </Typography>
            </Box>
          )}

          {/* Desktop Navigation */}
          {!isMobile && (
            <Box sx={{ display: 'flex', gap: 1, flexGrow: 1 }}>
              {navigationItems.slice(0, 5).map((item) => (
                <Button
                  key={item.href}
                  component={Link}
                  href={item.href}
                  variant={isActive(item.href) ? 'contained' : 'text'}
                  size="small"
                  startIcon={item.icon}
                  endIcon={item.badge && (
                    <Chip
                      label={item.badge}
                      size="small"
                      color="success"
                      sx={{ height: 16, fontSize: '0.6rem' }}
                    />
                  )}
                  sx={{
                    textTransform: 'none',
                    fontWeight: 500,
                    px: 2,
                  }}
                >
                  {item.label}
                </Button>
              ))}
            </Box>
          )}

          {/* Connection Status (Desktop) */}
          {!isMobile && showConnectionStatus && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Circle
                sx={{
                  fontSize: 8,
                  color: `${getConnectionColor()}.main`,
                  animation: connectionStatus === 'connected' ? 'pulse 2s infinite' : 'none',
                }}
              />
              <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                {getConnectionText()}
              </Typography>
            </Box>
          )}
        </Toolbar>
      </AppBar>

      {/* Mobile Drawer */}
      <Drawer
        variant="temporary"
        open={mobileOpen}
        onClose={handleDrawerToggle}
        ModalProps={{
          keepMounted: true, // Better open performance on mobile.
        }}
        sx={{
          display: { xs: 'block', md: 'none' },
          '& .MuiDrawer-paper': {
            boxSizing: 'border-box',
            width: 280,
          },
        }}
      >
        {drawer}
      </Drawer>

      {/* Desktop Drawer */}
      <Drawer
        variant="permanent"
        sx={{
          display: { xs: 'none', md: 'block' },
          '& .MuiDrawer-paper': {
            boxSizing: 'border-box',
            width: 280,
            top: 64, // Height of AppBar
            height: 'calc(100% - 64px)',
          },
        }}
        open
      >
        {drawer}
      </Drawer>

      {/* Add pulse animation */}
      <style jsx global>{`
        @keyframes pulse {
          0%, 100% {
            opacity: 1;
          }
          50% {
            opacity: 0.5;
          }
        }
      `}</style>
    </>
  );
}
