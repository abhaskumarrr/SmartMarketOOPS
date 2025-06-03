/**
 * Unified Page Wrapper Component
 * Provides consistent layout structure for all pages
 */

'use client';

import React from 'react';
import { Box, Container, useTheme, useMediaQuery } from '@mui/material';
import { UnifiedNavigation } from './UnifiedNavigation';

interface UnifiedPageWrapperProps {
  children: React.ReactNode;
  maxWidth?: 'xs' | 'sm' | 'md' | 'lg' | 'xl' | false;
  disablePadding?: boolean;
  connectionStatus?: 'connected' | 'disconnected' | 'connecting';
  showConnectionStatus?: boolean;
  fullHeight?: boolean;
}

export function UnifiedPageWrapper({
  children,
  maxWidth = false,
  disablePadding = false,
  connectionStatus = 'connected',
  showConnectionStatus = true,
  fullHeight = true,
}: UnifiedPageWrapperProps) {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', bgcolor: 'background.default' }}>
      {/* Navigation */}
      <UnifiedNavigation
        connectionStatus={connectionStatus}
        showConnectionStatus={showConnectionStatus}
      />

      {/* Main Content Area */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          pt: 8, // Account for AppBar height
          pl: { xs: 0, md: '280px' }, // Account for drawer width on desktop
          minHeight: fullHeight ? '100vh' : 'auto',
          bgcolor: 'background.default',
        }}
      >
        {maxWidth !== false ? (
          <Container
            maxWidth={maxWidth}
            sx={{
              py: disablePadding ? 0 : 3,
              px: disablePadding ? 0 : { xs: 2, sm: 3 },
              height: fullHeight ? 'calc(100vh - 64px)' : 'auto',
            }}
          >
            {children}
          </Container>
        ) : (
          <Box
            sx={{
              p: disablePadding ? 0 : { xs: 2, sm: 3 },
              height: fullHeight ? 'calc(100vh - 64px)' : 'auto',
            }}
          >
            {children}
          </Box>
        )}
      </Box>
    </Box>
  );
}
