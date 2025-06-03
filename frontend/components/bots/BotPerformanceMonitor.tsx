/**
 * Bot Performance Monitor
 * Real-time monitoring dashboard for trading bot performance
 */

'use client';

import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Chip,
  LinearProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Tooltip,
  Switch,
  FormControlLabel,
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  CheckCircle as CheckCircleIcon,
  Refresh as RefreshIcon,
  Pause as PauseIcon,
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Timeline as TimelineIcon,
  AccountBalance as AccountBalanceIcon,
  Speed as SpeedIcon,
  Info as InfoIcon,
} from '@mui/icons-material';
import { Bot, BotStatus } from '../../types/bot';
import { botService } from '../../lib/services/botService';

interface BotPerformanceMonitorProps {
  bot: Bot;
  onBotUpdate?: (bot: Bot) => void;
}

interface PerformanceMetrics {
  totalPnL: number;
  totalPnLPercent: number;
  todayPnL: number;
  todayPnLPercent: number;
  winRate: number;
  totalTrades: number;
  activePositions: number;
  avgTradeTime: number;
  sharpeRatio: number;
  maxDrawdown: number;
}

export const BotPerformanceMonitor: React.FC<BotPerformanceMonitorProps> = ({
  bot,
  onBotUpdate,
}) => {
  const [botStatus, setBotStatus] = useState<BotStatus | null>(null);
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [activeTab, setActiveTab] = useState(0);

  useEffect(() => {
    loadBotStatus();
    
    if (autoRefresh) {
      const interval = setInterval(loadBotStatus, 5000); // Refresh every 5 seconds
      return () => clearInterval(interval);
    }
  }, [bot.id, autoRefresh]);

  const loadBotStatus = async () => {
    try {
      setError(null);
      const response = await botService.getBotStatus(bot.id);
      
      if (response.success) {
        setBotStatus(response.data);
        
        // Extract performance metrics
        const performanceMetrics: PerformanceMetrics = {
          totalPnL: response.data.metrics?.profitLoss || 0,
          totalPnLPercent: response.data.metrics?.profitLossPercent || 0,
          todayPnL: response.data.metrics?.todayPnL || 0,
          todayPnLPercent: response.data.metrics?.todayPnLPercent || 0,
          winRate: response.data.metrics?.successRate || 0,
          totalTrades: response.data.metrics?.tradesExecuted || 0,
          activePositions: response.data.activePositions || 0,
          avgTradeTime: response.data.metrics?.averageTradeTime || 0,
          sharpeRatio: response.data.metrics?.sharpeRatio || 0,
          maxDrawdown: response.data.metrics?.maxDrawdown || 0,
        };
        
        setMetrics(performanceMetrics);
      } else {
        setError(response.message || 'Failed to load bot status');
      }
    } catch (error) {
      console.error('Error loading bot status:', error);
      setError(error instanceof Error ? error.message : 'Failed to load bot status');
    } finally {
      setLoading(false);
    }
  };

  const handleBotControl = async (action: 'start' | 'stop' | 'pause') => {
    try {
      setError(null);
      let response;
      
      switch (action) {
        case 'start':
          response = await botService.startBot(bot.id);
          break;
        case 'stop':
          response = await botService.stopBot(bot.id);
          break;
        case 'pause':
          response = await botService.pauseBot(bot.id);
          break;
      }
      
      if (response.success) {
        await loadBotStatus();
        if (onBotUpdate) {
          // Refresh bot data
          const botResponse = await botService.getBot(bot.id);
          if (botResponse.success) {
            onBotUpdate(botResponse.data);
          }
        }
      } else {
        setError(response.message || `Failed to ${action} bot`);
      }
    } catch (error) {
      console.error(`Error ${action}ing bot:`, error);
      setError(error instanceof Error ? error.message : `Failed to ${action} bot`);
    }
  };

  const getHealthColor = (health: string) => {
    switch (health) {
      case 'excellent':
      case 'good':
        return 'success';
      case 'degraded':
        return 'warning';
      case 'poor':
      case 'critical':
        return 'error';
      default:
        return 'default';
    }
  };

  const getHealthIcon = (health: string) => {
    switch (health) {
      case 'excellent':
      case 'good':
        return <CheckCircleIcon />;
      case 'degraded':
        return <WarningIcon />;
      case 'poor':
      case 'critical':
        return <ErrorIcon />;
      default:
        return <InfoIcon />;
    }
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  const formatPercentage = (value: number) => {
    return `${value.toFixed(2)}%`;
  };

  if (loading) {
    return (
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <LinearProgress sx={{ flex: 1 }} />
            <Typography variant="body2">Loading bot status...</Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  return (
    <Box>
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h5">
              {bot.name} Performance Monitor
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={autoRefresh}
                    onChange={(e) => setAutoRefresh(e.target.checked)}
                    size="small"
                  />
                }
                label="Auto Refresh"
              />
              <Tooltip title="Refresh Now">
                <IconButton onClick={loadBotStatus} size="small">
                  <RefreshIcon />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {/* Bot Status Header */}
          <Grid container spacing={2} sx={{ mb: 3 }}>
            <Grid item xs={12} sm={6} md={3}>
              <Card variant="outlined">
                <CardContent sx={{ textAlign: 'center' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1 }}>
                    {botStatus?.isRunning ? (
                      <CheckCircleIcon color="success" />
                    ) : (
                      <StopIcon color="error" />
                    )}
                    <Typography variant="h6" sx={{ ml: 1 }}>
                      {botStatus?.isRunning ? 'Running' : 'Stopped'}
                    </Typography>
                  </Box>
                  <Typography variant="body2" color="text.secondary">
                    Status
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <Card variant="outlined">
                <CardContent sx={{ textAlign: 'center' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1 }}>
                    {getHealthIcon(botStatus?.health || 'unknown')}
                    <Typography variant="h6" sx={{ ml: 1 }}>
                      {botStatus?.health || 'Unknown'}
                    </Typography>
                  </Box>
                  <Typography variant="body2" color="text.secondary">
                    Health
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <Card variant="outlined">
                <CardContent sx={{ textAlign: 'center' }}>
                  <Typography variant="h6">
                    {metrics?.activePositions || 0}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Active Positions
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <Card variant="outlined">
                <CardContent sx={{ textAlign: 'center' }}>
                  <Typography variant="h6">
                    {botStatus?.metrics?.latency || 0}ms
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Latency
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Bot Controls */}
          <Box sx={{ display: 'flex', gap: 1, mb: 3 }}>
            <Tooltip title="Start Bot">
              <IconButton
                onClick={() => handleBotControl('start')}
                disabled={botStatus?.isRunning}
                color="success"
              >
                <PlayIcon />
              </IconButton>
            </Tooltip>
            <Tooltip title="Pause Bot">
              <IconButton
                onClick={() => handleBotControl('pause')}
                disabled={!botStatus?.isRunning}
                color="warning"
              >
                <PauseIcon />
              </IconButton>
            </Tooltip>
            <Tooltip title="Stop Bot">
              <IconButton
                onClick={() => handleBotControl('stop')}
                disabled={!botStatus?.isRunning}
                color="error"
              >
                <StopIcon />
              </IconButton>
            </Tooltip>
          </Box>
        </CardContent>
      </Card>

      {/* Performance Metrics */}
      {metrics && (
        <Card>
          <CardContent>
            <Tabs value={activeTab} onChange={(_, newValue) => setActiveTab(newValue)}>
              <Tab label="Performance" />
              <Tab label="Trades" />
              <Tab label="Logs" />
              <Tab label="Errors" />
            </Tabs>

            <Box sx={{ mt: 2 }}>
              {activeTab === 0 && <PerformanceTab metrics={metrics} />}
              {activeTab === 1 && <TradesTab botStatus={botStatus} />}
              {activeTab === 2 && <LogsTab botStatus={botStatus} />}
              {activeTab === 3 && <ErrorsTab botStatus={botStatus} />}
            </Box>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

// Performance Tab Component
const PerformanceTab: React.FC<{ metrics: PerformanceMetrics }> = ({ metrics }) => {
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  const formatPercentage = (value: number) => {
    return `${value.toFixed(2)}%`;
  };

  return (
    <Grid container spacing={2}>
      <Grid item xs={12} sm={6} md={4}>
        <Card variant="outlined">
          <CardContent sx={{ textAlign: 'center' }}>
            <Typography variant="h4" color={metrics.totalPnL >= 0 ? 'success.main' : 'error.main'}>
              {formatCurrency(metrics.totalPnL)}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Total P&L ({formatPercentage(metrics.totalPnLPercent)})
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} sm={6} md={4}>
        <Card variant="outlined">
          <CardContent sx={{ textAlign: 'center' }}>
            <Typography variant="h4" color={metrics.todayPnL >= 0 ? 'success.main' : 'error.main'}>
              {formatCurrency(metrics.todayPnL)}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Today's P&L ({formatPercentage(metrics.todayPnLPercent)})
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} sm={6} md={4}>
        <Card variant="outlined">
          <CardContent sx={{ textAlign: 'center' }}>
            <Typography variant="h4" color="primary">
              {formatPercentage(metrics.winRate)}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Win Rate
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} sm={6} md={4}>
        <Card variant="outlined">
          <CardContent sx={{ textAlign: 'center' }}>
            <Typography variant="h4">
              {metrics.totalTrades}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Total Trades
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} sm={6} md={4}>
        <Card variant="outlined">
          <CardContent sx={{ textAlign: 'center' }}>
            <Typography variant="h4">
              {metrics.sharpeRatio.toFixed(2)}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Sharpe Ratio
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} sm={6} md={4}>
        <Card variant="outlined">
          <CardContent sx={{ textAlign: 'center' }}>
            <Typography variant="h4" color="error">
              {formatPercentage(metrics.maxDrawdown)}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Max Drawdown
            </Typography>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

// Trades Tab Component
const TradesTab: React.FC<{ botStatus: BotStatus | null }> = ({ botStatus }) => {
  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Recent Trades
      </Typography>
      <Typography variant="body2" color="text.secondary">
        Trade history will be displayed here
      </Typography>
    </Box>
  );
};

// Logs Tab Component
const LogsTab: React.FC<{ botStatus: BotStatus | null }> = ({ botStatus }) => {
  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Bot Logs
      </Typography>
      {botStatus?.logs && botStatus.logs.length > 0 ? (
        <List>
          {botStatus.logs.slice(-10).map((log, index) => (
            <React.Fragment key={index}>
              <ListItem>
                <ListItemIcon>
                  {log.level === 'error' ? (
                    <ErrorIcon color="error" />
                  ) : log.level === 'warning' ? (
                    <WarningIcon color="warning" />
                  ) : (
                    <InfoIcon color="info" />
                  )}
                </ListItemIcon>
                <ListItemText
                  primary={log.message}
                  secondary={new Date(log.timestamp).toLocaleString()}
                />
              </ListItem>
              {index < botStatus.logs.length - 1 && <Divider />}
            </React.Fragment>
          ))}
        </List>
      ) : (
        <Typography variant="body2" color="text.secondary">
          No logs available
        </Typography>
      )}
    </Box>
  );
};

// Errors Tab Component
const ErrorsTab: React.FC<{ botStatus: BotStatus | null }> = ({ botStatus }) => {
  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Bot Errors
      </Typography>
      {botStatus?.errors && botStatus.errors.length > 0 ? (
        <List>
          {botStatus.errors.slice(-10).map((error, index) => (
            <React.Fragment key={index}>
              <ListItem>
                <ListItemIcon>
                  <ErrorIcon color="error" />
                </ListItemIcon>
                <ListItemText
                  primary={error.message}
                  secondary={`${error.code ? `Code: ${error.code} | ` : ''}${new Date(error.timestamp).toLocaleString()}`}
                />
              </ListItem>
              {index < botStatus.errors.length - 1 && <Divider />}
            </React.Fragment>
          ))}
        </List>
      ) : (
        <Typography variant="body2" color="text.secondary">
          No errors reported
        </Typography>
      )}
    </Box>
  );
};

export default BotPerformanceMonitor;
