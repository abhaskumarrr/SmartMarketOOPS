/**
 * Bot Status Monitor Component
 * Real-time monitoring of bot status, health, and performance
 */

'use client';

import React, { useState, useEffect, useRef } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Typography,
  Box,
  Chip,
  LinearProgress,
  Alert,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Tabs,
  Tab,
  IconButton,
  Tooltip,
  Badge,
} from '@mui/material';
import {
  CheckCircle as SuccessIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  Refresh as RefreshIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Timeline as TimelineIcon,
  Speed as SpeedIcon,
} from '@mui/icons-material';
import { Bot, BotStatus, BotError, BotLog } from '../../types/bot';
import { botService } from '../../lib/services/botService';

interface BotStatusMonitorProps {
  bot: Bot;
  autoRefresh?: boolean;
  refreshInterval?: number; // in milliseconds
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index, ...other }) => {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`bot-status-tabpanel-${index}`}
      aria-labelledby={`bot-status-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
};

export const BotStatusMonitor: React.FC<BotStatusMonitorProps> = ({
  bot,
  autoRefresh = true,
  refreshInterval = 5000,
}) => {
  const [status, setStatus] = useState<BotStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tabValue, setTabValue] = useState(0);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  
  const wsRef = useRef<WebSocket | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    fetchBotStatus();
    
    if (autoRefresh) {
      // Set up WebSocket connection for real-time updates
      connectWebSocket();
      
      // Fallback polling
      intervalRef.current = setInterval(fetchBotStatus, refreshInterval);
    }

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [bot.id, autoRefresh, refreshInterval]);

  const connectWebSocket = () => {
    try {
      wsRef.current = botService.connectToBot(bot.id, (updatedStatus) => {
        setStatus(updatedStatus);
        setLastUpdate(new Date());
      });
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
    }
  };

  const fetchBotStatus = async () => {
    try {
      setError(null);
      const response = await botService.getBotStatus(bot.id);
      if (response.success) {
        setStatus(response.data);
        setLastUpdate(new Date());
      } else {
        setError(response.message || 'Failed to fetch bot status');
      }
    } catch (error) {
      console.error('Error fetching bot status:', error);
      setError(error instanceof Error ? error.message : 'Failed to fetch bot status');
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const getHealthColor = (health: string) => {
    switch (health) {
      case 'good': return 'success';
      case 'warning': return 'warning';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  const getHealthIcon = (health: string) => {
    switch (health) {
      case 'good': return <SuccessIcon color="success" />;
      case 'warning': return <WarningIcon color="warning" />;
      case 'error': return <ErrorIcon color="error" />;
      default: return <InfoIcon color="disabled" />;
    }
  };

  const formatLogLevel = (level: string) => {
    switch (level) {
      case 'error': return <ErrorIcon color="error" fontSize="small" />;
      case 'warning': return <WarningIcon color="warning" fontSize="small" />;
      case 'info': return <InfoIcon color="info" fontSize="small" />;
      case 'debug': return <InfoIcon color="disabled" fontSize="small" />;
      default: return <InfoIcon color="disabled" fontSize="small" />;
    }
  };

  const formatMetricValue = (key: string, value: any) => {
    if (typeof value === 'number') {
      if (key.includes('percentage') || key.includes('rate')) {
        return `${value.toFixed(2)}%`;
      }
      if (key.includes('pnl') || key.includes('profit') || key.includes('loss')) {
        return value >= 0 ? `+$${value.toFixed(2)}` : `-$${Math.abs(value).toFixed(2)}`;
      }
      return value.toFixed(2);
    }
    return String(value);
  };

  if (loading) {
    return (
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', p: 4 }}>
            <LinearProgress sx={{ width: '100%' }} />
          </Box>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent>
          <Alert severity="error">{error}</Alert>
        </CardContent>
      </Card>
    );
  }

  if (!status) {
    return (
      <Card>
        <CardContent>
          <Alert severity="info">No status data available for this bot.</Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader
        title={
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Typography variant="h6">
              Bot Status Monitor - {status.name}
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Chip
                icon={getHealthIcon(status.health)}
                label={status.health.toUpperCase()}
                color={getHealthColor(status.health) as any}
                size="small"
              />
              <Tooltip title="Refresh Status">
                <IconButton onClick={fetchBotStatus} size="small">
                  <RefreshIcon />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>
        }
        subheader={`Last updated: ${lastUpdate.toLocaleTimeString()}`}
      />

      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={tabValue} onChange={handleTabChange} aria-label="bot status tabs">
          <Tab label="Overview" />
          <Tab 
            label={
              <Badge badgeContent={status.errors.length} color="error">
                Logs
              </Badge>
            } 
          />
          <Tab label="Metrics" />
        </Tabs>
      </Box>

      <TabPanel value={tabValue} index={0}>
        <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 3 }}>
          <Box sx={{ flex: 1 }}>
            <Typography variant="h6" gutterBottom>
              Basic Information
            </Typography>
            <List dense>
              <ListItem>
                <ListItemText
                  primary="Symbol"
                  secondary={status.symbol}
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="Strategy"
                  secondary={status.strategy}
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="Timeframe"
                  secondary={status.timeframe}
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="Status"
                  secondary={
                    <Chip
                      label={status.isActive ? 'Active' : 'Inactive'}
                      color={status.isActive ? 'success' : 'default'}
                      size="small"
                    />
                  }
                />
              </ListItem>
            </List>
          </Box>

          <Box sx={{ flex: 1 }}>
            <Typography variant="h6" gutterBottom>
              Performance Summary
            </Typography>
            <List dense>
              <ListItem>
                <ListItemIcon>
                  <TimelineIcon />
                </ListItemIcon>
                <ListItemText
                  primary="Active Positions"
                  secondary={status.activePositions}
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  {(status.metrics.totalPnL || 0) >= 0 ?
                    <TrendingUpIcon color="success" /> :
                    <TrendingDownIcon color="error" />
                  }
                </ListItemIcon>
                <ListItemText
                  primary="Total P&L"
                  secondary={formatMetricValue('totalPnL', status.metrics.totalPnL || 0)}
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <SpeedIcon />
                </ListItemIcon>
                <ListItemText
                  primary="Win Rate"
                  secondary={formatMetricValue('winRate', status.metrics.winRate || 0)}
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="Total Trades"
                  secondary={status.metrics.totalTrades || 0}
                />
              </ListItem>
            </List>
          </Box>
        </Box>
      </TabPanel>

      <TabPanel value={tabValue} index={1}>
        <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 3 }}>
          <Box sx={{ flex: 1 }}>
            <Typography variant="h6" gutterBottom color="error">
              Recent Errors ({status.errors.length})
            </Typography>
            {status.errors.length === 0 ? (
              <Alert severity="success">No errors reported</Alert>
            ) : (
              <List dense>
                {status.errors.slice(0, 10).map((error: BotError, index) => (
                  <ListItem key={index}>
                    <ListItemIcon>
                      <ErrorIcon color="error" fontSize="small" />
                    </ListItemIcon>
                    <ListItemText
                      primary={error.message}
                      secondary={`${new Date(error.timestamp).toLocaleString()}${error.code ? ` (${error.code})` : ''}`}
                    />
                  </ListItem>
                ))}
              </List>
            )}
          </Box>

          <Box sx={{ flex: 1 }}>
            <Typography variant="h6" gutterBottom>
              Recent Logs ({status.logs.length})
            </Typography>
            {status.logs.length === 0 ? (
              <Alert severity="info">No logs available</Alert>
            ) : (
              <List dense>
                {status.logs.slice(0, 10).map((log: BotLog, index) => (
                  <ListItem key={index}>
                    <ListItemIcon>
                      {formatLogLevel(log.level)}
                    </ListItemIcon>
                    <ListItemText
                      primary={log.message}
                      secondary={`${new Date(log.timestamp).toLocaleString()} - ${log.level.toUpperCase()}`}
                    />
                  </ListItem>
                ))}
              </List>
            )}
          </Box>
        </Box>
      </TabPanel>

      <TabPanel value={tabValue} index={2}>
        <Typography variant="h6" gutterBottom>
          Performance Metrics
        </Typography>
        <Box sx={{
          display: 'grid',
          gridTemplateColumns: {
            xs: '1fr',
            sm: 'repeat(2, 1fr)',
            md: 'repeat(3, 1fr)'
          },
          gap: 2
        }}>
          {Object.entries(status.metrics).map(([key, value]) => (
            <Card variant="outlined" key={key}>
              <CardContent>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  {key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                </Typography>
                <Typography variant="h6">
                  {formatMetricValue(key, value)}
                </Typography>
              </CardContent>
            </Card>
          ))}
        </Box>
      </TabPanel>
    </Card>
  );
};

export default BotStatusMonitor;
