import React, { useEffect, useState } from 'react';
import { Box, Tooltip, Badge, CircularProgress, IconButton, Menu, MenuItem, Typography } from '@mui/material';
import {
  WifiOff as DisconnectedIcon,
  Wifi as ConnectedIcon,
  ErrorOutline as ErrorIcon,
  Refresh as RefreshIcon,
  MoreVert as MoreIcon
} from '@mui/icons-material';
import WebSocketService, { ConnectionStatus } from '../../lib/websocketService';

export interface WebSocketStatusProps {
  showTooltip?: boolean;
  size?: 'small' | 'medium' | 'large';
  isConnected?: boolean;
}

const WebSocketStatus: React.FC<WebSocketStatusProps> = ({
  showTooltip = true,
  size = 'small',
  isConnected,
}) => {
  const [status, setStatus] = useState<ConnectionStatus>(
    isConnected !== undefined 
      ? (isConnected ? ConnectionStatus.CONNECTED : ConnectionStatus.DISCONNECTED)
      : ConnectionStatus.DISCONNECTED
  );
  const [lastStatusChange, setLastStatusChange] = useState<Date>(new Date());
  const [reconnectCount, setReconnectCount] = useState<number>(0);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const open = Boolean(anchorEl);
  const wsService = WebSocketService.getInstance();

  useEffect(() => {
    // If isConnected prop is provided, use it to control the status
    if (isConnected !== undefined) {
      setStatus(isConnected ? ConnectionStatus.CONNECTED : ConnectionStatus.DISCONNECTED);
      return;
    }

    // Otherwise, use the WebSocketService
    setStatus(wsService.getStatus());

    // Listen for status changes
    const handleStatusChange = (newStatus: ConnectionStatus) => {
      setStatus(newStatus);
      setLastStatusChange(new Date());
      
      if (newStatus === ConnectionStatus.CONNECTING) {
        setReconnectCount(prev => prev + 1);
      }
    };

    wsService.on('status', handleStatusChange);

    // Periodically check connection health
    const healthCheckInterval = setInterval(() => {
      wsService.checkConnection();
    }, 60000); // Check every minute

    // Clean up on unmount
    return () => {
      wsService.removeListener('status', handleStatusChange);
      clearInterval(healthCheckInterval);
    };
  }, [wsService, isConnected]);

  // Determine icon and color based on status
  const getStatusInfo = () => {
    switch (status) {
      case ConnectionStatus.CONNECTED:
        return {
          icon: <ConnectedIcon fontSize={size} />,
          color: 'success.main',
          tooltip: 'WebSocket Connected',
        };
      case ConnectionStatus.CONNECTING:
        return {
          icon: <CircularProgress size={size === 'small' ? 16 : size === 'medium' ? 20 : 24} />,
          color: 'warning.main',
          tooltip: 'Connecting...',
        };
      case ConnectionStatus.ERROR:
        return {
          icon: <ErrorIcon fontSize={size} />,
          color: 'error.main',
          tooltip: 'Connection Error',
        };
      case ConnectionStatus.DISCONNECTED:
      default:
        return {
          icon: <DisconnectedIcon fontSize={size} />,
          color: 'text.disabled',
          tooltip: 'Disconnected',
        };
    }
  };

  const { icon, color, tooltip } = getStatusInfo();

  const handleClick = () => {
    if (status === ConnectionStatus.DISCONNECTED || status === ConnectionStatus.ERROR) {
      wsService.forceReconnect().catch(console.error);
    }
  };
  
  const handleMenuClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };
  
  const handleForceReconnect = () => {
    wsService.forceReconnect().catch(console.error);
    handleMenuClose();
  };
  
  // Format the time since last status change
  const getTimeSinceLastChange = () => {
    const now = new Date();
    const diffMs = now.getTime() - lastStatusChange.getTime();
    const diffSec = Math.floor(diffMs / 1000);
    
    if (diffSec < 60) {
      return `${diffSec}s ago`;
    } else if (diffSec < 3600) {
      return `${Math.floor(diffSec / 60)}m ago`;
    } else {
      return `${Math.floor(diffSec / 3600)}h ago`;
    }
  };

  const statusIndicator = (
    <Box sx={{ display: 'flex', alignItems: 'center' }}>
      <Badge
        color="secondary"
        variant="dot"
        invisible={status !== ConnectionStatus.CONNECTED}
        overlap="circular"
      >
        <Box
          sx={{
            color,
            display: 'flex',
            alignItems: 'center',
            cursor: 'pointer',
            '&:hover': {
              opacity: 0.8,
            },
          }}
          onClick={handleClick}
        >
          {icon}
        </Box>
      </Badge>
      
      <IconButton
        size="small"
        onClick={handleMenuClick}
        sx={{ ml: 0.5, color: 'text.secondary' }}
      >
        <MoreIcon fontSize="small" />
      </IconButton>
      
      <Menu
        anchorEl={anchorEl}
        open={open}
        onClose={handleMenuClose}
        MenuListProps={{
          'aria-labelledby': 'websocket-status-menu',
        }}
      >
        <MenuItem disabled>
          <Typography variant="body2">
            Status: <b>{status}</b>
          </Typography>
        </MenuItem>
        <MenuItem disabled>
          <Typography variant="body2">
            Since: {getTimeSinceLastChange()}
          </Typography>
        </MenuItem>
        <MenuItem disabled>
          <Typography variant="body2">
            Reconnects: {reconnectCount}
          </Typography>
        </MenuItem>
        <MenuItem onClick={handleForceReconnect}>
          <RefreshIcon fontSize="small" sx={{ mr: 1 }} />
          Force Reconnect
        </MenuItem>
      </Menu>
    </Box>
  );

  if (showTooltip) {
    return (
      <Tooltip title={tooltip} arrow>
        {statusIndicator}
      </Tooltip>
    );
  }

  return statusIndicator;
};

export default WebSocketStatus; 