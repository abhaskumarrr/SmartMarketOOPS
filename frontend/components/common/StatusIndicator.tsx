/**
 * Status Indicator Component
 * Visual indicators for system status, connection state, and health
 */

'use client';

import React, { useState, useEffect } from 'react';
import {
  Box,
  Chip,
  Tooltip,
  Typography,
  Stack,
  IconButton,
  Popover,
  Paper,
  LinearProgress,
  Divider,
} from '@mui/material';
import {
  Circle,
  Wifi,
  WifiOff,
  CloudDone,
  CloudOff,
  Speed,
  Memory,
  Storage,
  Refresh,
  Warning,
  CheckCircle,
  Error,
  Info,
} from '@mui/icons-material';

export type StatusType = 'online' | 'offline' | 'connecting' | 'error' | 'warning' | 'unknown';
export type HealthStatus = 'excellent' | 'good' | 'degraded' | 'poor' | 'critical';

export interface SystemStatus {
  api: StatusType;
  database: StatusType;
  cache: StatusType;
  websocket: StatusType;
  health: HealthStatus;
  performance: {
    responseTime: number;
    memoryUsage: number;
    cpuUsage: number;
  };
  lastUpdated: number;
}

interface StatusIndicatorProps {
  status?: StatusType;
  label?: string;
  showLabel?: boolean;
  size?: 'small' | 'medium' | 'large';
  variant?: 'dot' | 'chip' | 'icon';
  animated?: boolean;
  detailed?: boolean;
  onClick?: () => void;
}

const StatusIndicator: React.FC<StatusIndicatorProps> = ({
  status = 'unknown',
  label,
  showLabel = false,
  size = 'medium',
  variant = 'dot',
  animated = true,
  detailed = false,
  onClick,
}) => {
  const [anchorEl, setAnchorEl] = useState<HTMLElement | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    api: 'unknown',
    database: 'unknown',
    cache: 'unknown',
    websocket: 'unknown',
    health: 'good',
    performance: {
      responseTime: 0,
      memoryUsage: 0,
      cpuUsage: 0,
    },
    lastUpdated: Date.now(),
  });

  useEffect(() => {
    if (detailed) {
      fetchSystemStatus();
      const interval = setInterval(fetchSystemStatus, 30000); // Update every 30 seconds
      return () => clearInterval(interval);
    }
  }, [detailed]);

  const fetchSystemStatus = async () => {
    try {
      // In a real application, this would fetch from your API
      const response = await fetch('/api/health');
      const data = await response.json();
      
      setSystemStatus({
        api: response.ok ? 'online' : 'error',
        database: data.services?.database === 'healthy' ? 'online' : 'error',
        cache: data.services?.cache === 'healthy' ? 'online' : 'warning',
        websocket: 'online', // Would check WebSocket connection
        health: data.health || 'good',
        performance: {
          responseTime: parseFloat(data.responseTime) || 0,
          memoryUsage: data.memory?.heapUsed / data.memory?.heapTotal * 100 || 0,
          cpuUsage: Math.random() * 100, // Would get real CPU usage
        },
        lastUpdated: Date.now(),
      });
    } catch (error) {
      setSystemStatus(prev => ({
        ...prev,
        api: 'error',
        database: 'error',
        cache: 'error',
        websocket: 'error',
        health: 'critical',
        lastUpdated: Date.now(),
      }));
    }
  };

  const getStatusColor = (status: StatusType) => {
    switch (status) {
      case 'online':
        return '#4caf50'; // Green
      case 'offline':
        return '#9e9e9e'; // Grey
      case 'connecting':
        return '#ff9800'; // Orange
      case 'error':
        return '#f44336'; // Red
      case 'warning':
        return '#ff9800'; // Orange
      default:
        return '#9e9e9e'; // Grey
    }
  };

  const getHealthColor = (health: HealthStatus) => {
    switch (health) {
      case 'excellent':
        return '#4caf50';
      case 'good':
        return '#8bc34a';
      case 'degraded':
        return '#ff9800';
      case 'poor':
        return '#ff5722';
      case 'critical':
        return '#f44336';
      default:
        return '#9e9e9e';
    }
  };

  const getStatusIcon = (status: StatusType) => {
    switch (status) {
      case 'online':
        return <CheckCircle sx={{ color: getStatusColor(status) }} />;
      case 'offline':
        return <Error sx={{ color: getStatusColor(status) }} />;
      case 'connecting':
        return <Info sx={{ color: getStatusColor(status) }} />;
      case 'error':
        return <Error sx={{ color: getStatusColor(status) }} />;
      case 'warning':
        return <Warning sx={{ color: getStatusColor(status) }} />;
      default:
        return <Circle sx={{ color: getStatusColor(status) }} />;
    }
  };

  const getSizeValue = () => {
    switch (size) {
      case 'small':
        return 8;
      case 'large':
        return 16;
      default:
        return 12;
    }
  };

  const handleClick = (event: React.MouseEvent<HTMLElement>) => {
    if (detailed) {
      setAnchorEl(event.currentTarget);
    }
    if (onClick) {
      onClick();
    }
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const renderDot = () => (
    <Box
      sx={{
        width: getSizeValue(),
        height: getSizeValue(),
        borderRadius: '50%',
        backgroundColor: getStatusColor(status),
        animation: animated && status === 'connecting' ? 'pulse 1.5s infinite' : 'none',
        '@keyframes pulse': {
          '0%': {
            opacity: 1,
          },
          '50%': {
            opacity: 0.5,
          },
          '100%': {
            opacity: 1,
          },
        },
      }}
    />
  );

  const renderChip = () => (
    <Chip
      icon={getStatusIcon(status)}
      label={label || status}
      size={size === 'large' ? 'medium' : 'small'}
      color={status === 'online' ? 'success' : status === 'error' ? 'error' : 'default'}
      variant="outlined"
    />
  );

  const renderIcon = () => getStatusIcon(status);

  const renderIndicator = () => {
    switch (variant) {
      case 'chip':
        return renderChip();
      case 'icon':
        return renderIcon();
      default:
        return renderDot();
    }
  };

  const renderDetailedStatus = () => (
    <Paper sx={{ p: 2, minWidth: 300 }}>
      <Typography variant="h6" gutterBottom>
        System Status
      </Typography>
      
      <Stack spacing={2}>
        {/* Service Status */}
        <Box>
          <Typography variant="subtitle2" gutterBottom>
            Services
          </Typography>
          <Stack spacing={1}>
            <Stack direction="row" alignItems="center" justifyContent="space-between">
              <Stack direction="row" alignItems="center" spacing={1}>
                <Wifi fontSize="small" />
                <Typography variant="body2">API</Typography>
              </Stack>
              <StatusIndicator status={systemStatus.api} variant="chip" size="small" />
            </Stack>
            
            <Stack direction="row" alignItems="center" justifyContent="space-between">
              <Stack direction="row" alignItems="center" spacing={1}>
                <Storage fontSize="small" />
                <Typography variant="body2">Database</Typography>
              </Stack>
              <StatusIndicator status={systemStatus.database} variant="chip" size="small" />
            </Stack>
            
            <Stack direction="row" alignItems="center" spacing={1}>
              <CloudDone fontSize="small" />
              <Typography variant="body2">Cache</Typography>
              <StatusIndicator status={systemStatus.cache} variant="chip" size="small" />
            </Stack>
          </Stack>
        </Box>

        <Divider />

        {/* Performance Metrics */}
        <Box>
          <Typography variant="subtitle2" gutterBottom>
            Performance
          </Typography>
          <Stack spacing={2}>
            <Box>
              <Stack direction="row" justifyContent="space-between" mb={1}>
                <Typography variant="body2">Response Time</Typography>
                <Typography variant="body2">
                  {systemStatus.performance.responseTime.toFixed(0)}ms
                </Typography>
              </Stack>
              <LinearProgress
                variant="determinate"
                value={Math.min(systemStatus.performance.responseTime / 10, 100)}
                color={systemStatus.performance.responseTime < 200 ? 'success' : 'warning'}
              />
            </Box>
            
            <Box>
              <Stack direction="row" justifyContent="space-between" mb={1}>
                <Typography variant="body2">Memory Usage</Typography>
                <Typography variant="body2">
                  {systemStatus.performance.memoryUsage.toFixed(1)}%
                </Typography>
              </Stack>
              <LinearProgress
                variant="determinate"
                value={systemStatus.performance.memoryUsage}
                color={systemStatus.performance.memoryUsage < 70 ? 'success' : 'warning'}
              />
            </Box>
          </Stack>
        </Box>

        <Divider />

        {/* Overall Health */}
        <Box>
          <Stack direction="row" alignItems="center" justifyContent="space-between">
            <Typography variant="subtitle2">Overall Health</Typography>
            <Chip
              label={systemStatus.health}
              size="small"
              sx={{
                backgroundColor: getHealthColor(systemStatus.health),
                color: 'white',
              }}
            />
          </Stack>
        </Box>

        {/* Last Updated */}
        <Box>
          <Stack direction="row" alignItems="center" justifyContent="space-between">
            <Typography variant="caption" color="text.secondary">
              Last updated: {new Date(systemStatus.lastUpdated).toLocaleTimeString()}
            </Typography>
            <IconButton size="small" onClick={fetchSystemStatus}>
              <Refresh fontSize="small" />
            </IconButton>
          </Stack>
        </Box>
      </Stack>
    </Paper>
  );

  return (
    <>
      <Tooltip title={label || `Status: ${status}`}>
        <Box
          onClick={handleClick}
          sx={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 1,
            cursor: detailed || onClick ? 'pointer' : 'default',
          }}
        >
          {renderIndicator()}
          {showLabel && label && (
            <Typography variant="body2" color="text.secondary">
              {label}
            </Typography>
          )}
        </Box>
      </Tooltip>

      {detailed && (
        <Popover
          open={Boolean(anchorEl)}
          anchorEl={anchorEl}
          onClose={handleClose}
          anchorOrigin={{
            vertical: 'bottom',
            horizontal: 'left',
          }}
        >
          {renderDetailedStatus()}
        </Popover>
      )}
    </>
  );
};

// Specialized status indicators
export const ConnectionStatus: React.FC<{ isConnected: boolean }> = ({ isConnected }) => (
  <StatusIndicator
    status={isConnected ? 'online' : 'offline'}
    label={isConnected ? 'Connected' : 'Disconnected'}
    variant="chip"
    showLabel
  />
);

export const SystemHealthIndicator: React.FC = () => (
  <StatusIndicator
    status="online"
    label="System Health"
    variant="icon"
    detailed
  />
);

export const TradingBotStatus: React.FC<{ isRunning: boolean; hasError?: boolean }> = ({ 
  isRunning, 
  hasError 
}) => (
  <StatusIndicator
    status={hasError ? 'error' : isRunning ? 'online' : 'offline'}
    label={hasError ? 'Error' : isRunning ? 'Running' : 'Stopped'}
    variant="chip"
    animated={isRunning}
  />
);

export default StatusIndicator;
