/**
 * Bot List Component
 * Displays a list of user's trading bots with status and controls
 */

'use client';

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Typography,
  Button,
  Chip,
  IconButton,
  Menu,
  MenuItem,
  Box,
  Alert,
  CircularProgress,
  Tooltip,
} from '@mui/material';
import {
  PlayArrow as StartIcon,
  Stop as StopIcon,
  Pause as PauseIcon,
  MoreVert as MoreIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Settings as SettingsIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
} from '@mui/icons-material';
import { Bot, BotStatus } from '../../types/bot';
import { botService } from '../../lib/services/botService';

interface BotListProps {
  onEditBot: (bot: Bot) => void;
  onDeleteBot: (bot: Bot) => void;
  onConfigureRisk: (bot: Bot) => void;
  refreshTrigger?: number;
}

interface BotCardProps {
  bot: Bot;
  status?: BotStatus;
  onStart: () => void;
  onStop: () => void;
  onPause: () => void;
  onEdit: () => void;
  onDelete: () => void;
  onConfigureRisk: () => void;
  loading: boolean;
}

const BotCard: React.FC<BotCardProps> = ({
  bot,
  status,
  onStart,
  onStop,
  onPause,
  onEdit,
  onDelete,
  onConfigureRisk,
  loading,
}) => {
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);

  const handleMenuClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const getStatusColor = (isActive: boolean, health?: string) => {
    if (!isActive) return 'default';
    switch (health) {
      case 'good': return 'success';
      case 'warning': return 'warning';
      case 'error': return 'error';
      default: return 'info';
    }
  };

  const getStatusText = (isActive: boolean, health?: string) => {
    if (!isActive) return 'Stopped';
    switch (health) {
      case 'good': return 'Running';
      case 'warning': return 'Warning';
      case 'error': return 'Error';
      default: return 'Unknown';
    }
  };

  const formatPnL = (pnl?: number) => {
    if (pnl === undefined) return 'N/A';
    const isPositive = pnl >= 0;
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', color: isPositive ? 'success.main' : 'error.main' }}>
        {isPositive ? <TrendingUpIcon fontSize="small" /> : <TrendingDownIcon fontSize="small" />}
        <Typography variant="body2" sx={{ ml: 0.5 }}>
          {isPositive ? '+' : ''}{pnl.toFixed(2)}%
        </Typography>
      </Box>
    );
  };

  return (
    <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <CardHeader
        title={
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Typography variant="h6" component="div">
              {bot.name}
            </Typography>
            <Chip
              label={getStatusText(bot.isActive, status?.health)}
              color={getStatusColor(bot.isActive, status?.health)}
              size="small"
            />
          </Box>
        }
        subheader={`${bot.symbol} • ${bot.strategy} • ${bot.timeframe}`}
        action={
          <IconButton onClick={handleMenuClick} disabled={loading}>
            <MoreIcon />
          </IconButton>
        }
      />
      
      <CardContent sx={{ flexGrow: 1 }}>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
            <Box>
              <Typography variant="body2" color="text.secondary">
                Active Positions
              </Typography>
              <Typography variant="h6">
                {status?.activePositions || 0}
              </Typography>
            </Box>
            <Box>
              <Typography variant="body2" color="text.secondary">
                Performance
              </Typography>
              {formatPnL(status?.metrics?.totalPnLPercentage)}
            </Box>
          </Box>
          <Box>
            <Typography variant="body2" color="text.secondary">
              Last Update
            </Typography>
            <Typography variant="body2">
              {status?.lastUpdate ? new Date(status.lastUpdate).toLocaleString() : 'Never'}
            </Typography>
          </Box>
        </Box>

        <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
          {!bot.isActive ? (
            <Button
              variant="contained"
              color="success"
              startIcon={<StartIcon />}
              onClick={onStart}
              disabled={loading}
              size="small"
            >
              Start
            </Button>
          ) : (
            <>
              <Button
                variant="contained"
                color="warning"
                startIcon={<PauseIcon />}
                onClick={onPause}
                disabled={loading}
                size="small"
              >
                Pause
              </Button>
              <Button
                variant="contained"
                color="error"
                startIcon={<StopIcon />}
                onClick={onStop}
                disabled={loading}
                size="small"
              >
                Stop
              </Button>
            </>
          )}
        </Box>
      </CardContent>

      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={() => { onEdit(); handleMenuClose(); }}>
          <EditIcon sx={{ mr: 1 }} />
          Edit Bot
        </MenuItem>
        <MenuItem onClick={() => { onConfigureRisk(); handleMenuClose(); }}>
          <SettingsIcon sx={{ mr: 1 }} />
          Risk Settings
        </MenuItem>
        <MenuItem onClick={() => { onDelete(); handleMenuClose(); }} sx={{ color: 'error.main' }}>
          <DeleteIcon sx={{ mr: 1 }} />
          Delete Bot
        </MenuItem>
      </Menu>
    </Card>
  );
};

export const BotList: React.FC<BotListProps> = ({
  onEditBot,
  onDeleteBot,
  onConfigureRisk,
  refreshTrigger = 0,
}) => {
  const [bots, setBots] = useState<Bot[]>([]);
  const [botStatuses, setBotStatuses] = useState<Record<string, BotStatus>>({});
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState<Record<string, boolean>>({});
  const [error, setError] = useState<string | null>(null);

  const fetchBots = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await botService.getBots();
      if (response.success) {
        setBots(response.data);
        
        // Fetch status for each bot
        const statusPromises = response.data.map(async (bot) => {
          try {
            const statusResponse = await botService.getBotStatus(bot.id);
            return { botId: bot.id, status: statusResponse.data };
          } catch (error) {
            console.error(`Error fetching status for bot ${bot.id}:`, error);
            return { botId: bot.id, status: null };
          }
        });
        
        const statuses = await Promise.all(statusPromises);
        const statusMap: Record<string, BotStatus> = {};
        statuses.forEach(({ botId, status }) => {
          if (status) {
            statusMap[botId] = status;
          }
        });
        setBotStatuses(statusMap);
      } else {
        setError(response.message || 'Failed to fetch bots');
      }
    } catch (error) {
      console.error('Error fetching bots:', error);
      setError(error instanceof Error ? error.message : 'Failed to fetch bots');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchBots();
  }, [refreshTrigger]);

  const handleBotAction = async (botId: string, action: () => Promise<any>, actionName: string) => {
    try {
      setActionLoading(prev => ({ ...prev, [botId]: true }));
      await action();
      
      // Refresh bot data after action
      await fetchBots();
    } catch (error) {
      console.error(`Error ${actionName} bot:`, error);
      setError(error instanceof Error ? error.message : `Failed to ${actionName} bot`);
    } finally {
      setActionLoading(prev => ({ ...prev, [botId]: false }));
    }
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        {error}
      </Alert>
    );
  }

  if (bots.length === 0) {
    return (
      <Card>
        <CardContent sx={{ textAlign: 'center', py: 4 }}>
          <Typography variant="h6" color="text.secondary" gutterBottom>
            No Trading Bots Found
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Create your first trading bot to get started with automated trading.
          </Typography>
        </CardContent>
      </Card>
    );
  }

  return (
    <Box sx={{
      display: 'grid',
      gridTemplateColumns: {
        xs: '1fr',
        sm: 'repeat(2, 1fr)',
        md: 'repeat(3, 1fr)'
      },
      gap: 3
    }}>
      {bots.map((bot) => (
        <BotCard
          key={bot.id}
          bot={bot}
          status={botStatuses[bot.id]}
          onStart={() => handleBotAction(bot.id, () => botService.startBot(bot.id), 'starting')}
          onStop={() => handleBotAction(bot.id, () => botService.stopBot(bot.id), 'stopping')}
          onPause={() => handleBotAction(bot.id, () => botService.pauseBot(bot.id), 'pausing')}
          onEdit={() => onEditBot(bot)}
          onDelete={() => onDeleteBot(bot)}
          onConfigureRisk={() => onConfigureRisk(bot)}
          loading={actionLoading[bot.id] || false}
        />
      ))}
    </Box>
  );
};

export default BotList;
