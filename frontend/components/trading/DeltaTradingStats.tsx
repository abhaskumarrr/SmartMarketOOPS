/**
 * Delta Trading Stats Component
 * Displays key performance metrics and statistics
 */

'use client';

import React from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Chip,
  LinearProgress,
  useTheme,
} from '@mui/material';
import {
  SmartToy as BotIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  AccountBalance as BalanceIcon,
  SwapHoriz as TradesIcon,
  Speed as PerformanceIcon,
} from '@mui/icons-material';
import { BotManagerStatus } from '../../lib/api/deltaTradingApi';

interface DeltaTradingStatsProps {
  status: BotManagerStatus;
  sx?: any;
}

export function DeltaTradingStats({ status, sx }: DeltaTradingStatsProps) {
  const theme = useTheme();

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value);
  };

  const formatNumber = (value: number) => {
    return new Intl.NumberFormat('en-IN').format(value);
  };

  const getPnLColor = (pnl: number) => {
    if (pnl > 0) return 'success.main';
    if (pnl < 0) return 'error.main';
    return 'text.secondary';
  };

  const getPnLIcon = (pnl: number) => {
    if (pnl > 0) return <TrendingUpIcon />;
    if (pnl < 0) return <TrendingDownIcon />;
    return <BalanceIcon />;
  };

  const getActiveBotsPercentage = () => {
    if (status.totalBots === 0) return 0;
    return (status.runningBots / status.totalBots) * 100;
  };

  const stats = [
    {
      title: 'Total Bots',
      value: formatNumber(status.totalBots),
      subtitle: `${status.runningBots} running`,
      icon: <BotIcon />,
      color: 'primary.main',
      progress: getActiveBotsPercentage(),
    },
    {
      title: 'Total P&L',
      value: formatCurrency(status.totalPnL),
      subtitle: status.totalTrades > 0 ? `${formatNumber(status.totalTrades)} trades` : 'No trades yet',
      icon: getPnLIcon(status.totalPnL),
      color: getPnLColor(status.totalPnL),
    },
    {
      title: 'Running Bots',
      value: formatNumber(status.runningBots),
      subtitle: `${status.pausedBots} paused, ${status.stoppedBots} stopped`,
      icon: <PerformanceIcon />,
      color: 'success.main',
    },
    {
      title: 'Total Trades',
      value: formatNumber(status.totalTrades),
      subtitle: status.totalBots > 0 ? `${(status.totalTrades / status.totalBots).toFixed(1)} avg per bot` : 'No bots active',
      icon: <TradesIcon />,
      color: 'info.main',
    },
  ];

  return (
    <Grid container spacing={3} sx={sx}>
      {stats.map((stat, index) => (
        <Grid item xs={12} sm={6} md={3} key={index}>
          <Paper
            sx={{
              p: 3,
              height: '100%',
              background: `linear-gradient(135deg, ${theme.palette.background.paper} 0%, rgba(59, 130, 246, 0.05) 100%)`,
              border: `1px solid ${theme.palette.divider}`,
              transition: 'all 0.3s ease',
              '&:hover': {
                transform: 'translateY(-2px)',
                boxShadow: theme.shadows[8],
              },
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', mb: 2 }}>
              <Box>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                  {stat.title}
                </Typography>
                <Typography variant="h4" sx={{ fontWeight: 700, color: stat.color, mb: 0.5 }}>
                  {stat.value}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {stat.subtitle}
                </Typography>
              </Box>
              <Box
                sx={{
                  p: 1.5,
                  borderRadius: 2,
                  bgcolor: `${stat.color}15`,
                  color: stat.color,
                }}
              >
                {stat.icon}
              </Box>
            </Box>

            {/* Progress bar for active bots */}
            {stat.progress !== undefined && (
              <Box sx={{ mt: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="caption" color="text.secondary">
                    Active Rate
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {stat.progress.toFixed(1)}%
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={stat.progress}
                  sx={{
                    height: 6,
                    borderRadius: 3,
                    bgcolor: 'rgba(255, 255, 255, 0.1)',
                    '& .MuiLinearProgress-bar': {
                      borderRadius: 3,
                      bgcolor: stat.color,
                    },
                  }}
                />
              </Box>
            )}
          </Paper>
        </Grid>
      ))}

      {/* Status Summary */}
      <Grid item xs={12}>
        <Paper sx={{ p: 2, bgcolor: 'background.paper', border: `1px solid ${theme.palette.divider}` }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Typography variant="body2" color="text.secondary">
                System Status:
              </Typography>
              <Chip 
                label={`${status.exchange} ${status.environment}`} 
                color="primary" 
                size="small" 
                variant="outlined"
              />
            </Box>
            
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              {status.runningBots > 0 && (
                <Chip 
                  label={`${status.runningBots} Running`} 
                  color="success" 
                  size="small"
                />
              )}
              {status.pausedBots > 0 && (
                <Chip 
                  label={`${status.pausedBots} Paused`} 
                  color="warning" 
                  size="small"
                />
              )}
              {status.stoppedBots > 0 && (
                <Chip 
                  label={`${status.stoppedBots} Stopped`} 
                  color="default" 
                  size="small"
                />
              )}
              {status.errorBots > 0 && (
                <Chip 
                  label={`${status.errorBots} Error`} 
                  color="error" 
                  size="small"
                />
              )}
            </Box>
            
            <Typography variant="caption" color="text.secondary">
              Last updated: {new Date(status.timestamp).toLocaleTimeString()}
            </Typography>
          </Box>
        </Paper>
      </Grid>
    </Grid>
  );
}
