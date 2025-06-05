/**
 * Delta Bot List Component
 * Displays and manages trading bots with actions
 */

'use client';

import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  IconButton,
  Button,
  Menu,
  MenuItem,
  Tooltip,
  useTheme,
  CircularProgress,
} from '@mui/material';
import {
  PlayArrow as StartIcon,
  Stop as StopIcon,
  Pause as PauseIcon,
  MoreVert as MoreIcon,
  Delete as DeleteIcon,
  Settings as SettingsIcon,
  Assessment as PerformanceIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
} from '@mui/icons-material';
import { BotStatus } from '../../lib/api/deltaTradingApi';

interface DeltaBotListProps {
  bots: BotStatus[];
  onBotAction: (action: string, botId: string) => Promise<void>;
  refreshing: boolean;
}

export function DeltaBotList({ bots, onBotAction, refreshing }: DeltaBotListProps) {
  const theme = useTheme();
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [selectedBot, setSelectedBot] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState<string | null>(null);

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>, botId: string) => {
    setAnchorEl(event.currentTarget);
    setSelectedBot(botId);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
    setSelectedBot(null);
  };

  const handleAction = async (action: string, botId: string) => {
    try {
      setActionLoading(`${action}-${botId}`);
      await onBotAction(action, botId);
    } finally {
      setActionLoading(null);
      handleMenuClose();
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'success';
      case 'paused':
        return 'warning';
      case 'stopped':
        return 'default';
      case 'error':
        return 'error';
      default:
        return 'default';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
        return <StartIcon sx={{ fontSize: 16 }} />;
      case 'paused':
        return <PauseIcon sx={{ fontSize: 16 }} />;
      case 'stopped':
        return <StopIcon sx={{ fontSize: 16 }} />;
      case 'error':
        return <DeleteIcon sx={{ fontSize: 16 }} />;
      default:
        return null;
    }
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value);
  };

  const formatPercentage = (value: string) => {
    const num = parseFloat(value);
    return `${num.toFixed(1)}%`;
  };

  const getPnLColor = (pnl: number) => {
    if (pnl > 0) return 'success.main';
    if (pnl < 0) return 'error.main';
    return 'text.secondary';
  };

  const getPnLIcon = (pnl: number) => {
    if (pnl > 0) return <TrendingUpIcon sx={{ fontSize: 16, color: 'success.main' }} />;
    if (pnl < 0) return <TrendingDownIcon sx={{ fontSize: 16, color: 'error.main' }} />;
    return null;
  };

  if (bots.length === 0) {
    return (
      <Paper sx={{ p: 4, textAlign: 'center', bgcolor: 'background.paper' }}>
        <Typography variant="h6" color="text.secondary" sx={{ mb: 2 }}>
          No Trading Bots Found
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Create your first trading bot to get started with automated trading on Delta Exchange.
        </Typography>
      </Paper>
    );
  }

  return (
    <Paper sx={{ bgcolor: 'background.paper', border: `1px solid ${theme.palette.divider}` }}>
      <Box sx={{ p: 3, borderBottom: `1px solid ${theme.palette.divider}` }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Trading Bots ({bots.length})
          </Typography>
          {refreshing && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <CircularProgress size={16} />
              <Typography variant="caption" color="text.secondary">
                Refreshing...
              </Typography>
            </Box>
          )}
        </Box>
      </Box>

      <TableContainer>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Bot Name</TableCell>
              <TableCell>Symbol</TableCell>
              <TableCell>Status</TableCell>
              <TableCell align="right">Capital</TableCell>
              <TableCell align="right">P&L</TableCell>
              <TableCell align="right">Trades</TableCell>
              <TableCell align="right">Win Rate</TableCell>
              <TableCell align="right">Last Activity</TableCell>
              <TableCell align="center">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {bots.map((bot) => (
              <TableRow 
                key={bot.id}
                sx={{ 
                  '&:hover': { bgcolor: 'action.hover' },
                  '&:last-child td, &:last-child th': { border: 0 }
                }}
              >
                <TableCell>
                  <Box>
                    <Typography variant="body2" sx={{ fontWeight: 500 }}>
                      {bot.name}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {bot.id}
                    </Typography>
                  </Box>
                </TableCell>
                
                <TableCell>
                  <Chip 
                    label={bot.symbol} 
                    size="small" 
                    variant="outlined"
                    color="primary"
                  />
                </TableCell>
                
                <TableCell>
                  <Chip
                    icon={getStatusIcon(bot.status)}
                    label={bot.status.charAt(0).toUpperCase() + bot.status.slice(1)}
                    size="small"
                    color={getStatusColor(bot.status) as any}
                    variant={bot.status === 'running' ? 'filled' : 'outlined'}
                  />
                </TableCell>
                
                <TableCell align="right">
                  <Typography variant="body2">
                    {formatCurrency(bot.config.capital)}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {bot.config.leverage}x leverage
                  </Typography>
                </TableCell>
                
                <TableCell align="right">
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 0.5 }}>
                    {getPnLIcon(bot.totalPnL)}
                    <Typography 
                      variant="body2" 
                      sx={{ 
                        color: getPnLColor(bot.totalPnL),
                        fontWeight: 500
                      }}
                    >
                      {formatCurrency(bot.totalPnL)}
                    </Typography>
                  </Box>
                </TableCell>
                
                <TableCell align="right">
                  <Typography variant="body2">
                    {bot.totalTrades}
                  </Typography>
                </TableCell>
                
                <TableCell align="right">
                  <Typography 
                    variant="body2"
                    sx={{ 
                      color: parseFloat(bot.winRate) >= 50 ? 'success.main' : 'text.secondary'
                    }}
                  >
                    {formatPercentage(bot.winRate)}
                  </Typography>
                </TableCell>
                
                <TableCell align="right">
                  <Typography variant="caption" color="text.secondary">
                    {bot.lastActivity ? new Date(bot.lastActivity).toLocaleString() : 'Never'}
                  </Typography>
                </TableCell>
                
                <TableCell align="center">
                  <Box sx={{ display: 'flex', gap: 0.5 }}>
                    {/* Quick Actions */}
                    {bot.status === 'stopped' && (
                      <Tooltip title="Start Bot">
                        <IconButton
                          size="small"
                          onClick={() => handleAction('start', bot.id)}
                          disabled={actionLoading === `start-${bot.id}`}
                          sx={{ color: 'success.main' }}
                        >
                          {actionLoading === `start-${bot.id}` ? (
                            <CircularProgress size={16} />
                          ) : (
                            <StartIcon />
                          )}
                        </IconButton>
                      </Tooltip>
                    )}
                    
                    {bot.status === 'running' && (
                      <Tooltip title="Pause Bot">
                        <IconButton
                          size="small"
                          onClick={() => handleAction('pause', bot.id)}
                          disabled={actionLoading === `pause-${bot.id}`}
                          sx={{ color: 'warning.main' }}
                        >
                          {actionLoading === `pause-${bot.id}` ? (
                            <CircularProgress size={16} />
                          ) : (
                            <PauseIcon />
                          )}
                        </IconButton>
                      </Tooltip>
                    )}
                    
                    {bot.status === 'paused' && (
                      <Tooltip title="Resume Bot">
                        <IconButton
                          size="small"
                          onClick={() => handleAction('resume', bot.id)}
                          disabled={actionLoading === `resume-${bot.id}`}
                          sx={{ color: 'success.main' }}
                        >
                          {actionLoading === `resume-${bot.id}` ? (
                            <CircularProgress size={16} />
                          ) : (
                            <StartIcon />
                          )}
                        </IconButton>
                      </Tooltip>
                    )}
                    
                    {(bot.status === 'running' || bot.status === 'paused') && (
                      <Tooltip title="Stop Bot">
                        <IconButton
                          size="small"
                          onClick={() => handleAction('stop', bot.id)}
                          disabled={actionLoading === `stop-${bot.id}`}
                          sx={{ color: 'error.main' }}
                        >
                          {actionLoading === `stop-${bot.id}` ? (
                            <CircularProgress size={16} />
                          ) : (
                            <StopIcon />
                          )}
                        </IconButton>
                      </Tooltip>
                    )}
                    
                    {/* More Actions Menu */}
                    <IconButton
                      size="small"
                      onClick={(e) => handleMenuOpen(e, bot.id)}
                    >
                      <MoreIcon />
                    </IconButton>
                  </Box>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Actions Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={() => selectedBot && handleAction('performance', selectedBot)}>
          <PerformanceIcon sx={{ mr: 1 }} />
          View Performance
        </MenuItem>
        <MenuItem onClick={() => selectedBot && handleAction('settings', selectedBot)}>
          <SettingsIcon sx={{ mr: 1 }} />
          Edit Settings
        </MenuItem>
        <MenuItem 
          onClick={() => selectedBot && handleAction('remove', selectedBot)}
          sx={{ color: 'error.main' }}
        >
          <DeleteIcon sx={{ mr: 1 }} />
          Remove Bot
        </MenuItem>
      </Menu>
    </Paper>
  );
}
