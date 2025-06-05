/**
 * Notification Center Component
 * Displays and manages all user notifications
 */

'use client';

import React, { useState, useEffect } from 'react';
import {
  Box,
  IconButton,
  Badge,
  Popover,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Typography,
  Button,
  Chip,
  Divider,
  Stack,
  Paper,
  Tooltip,
  Avatar,
} from '@mui/material';
import {
  Notifications,
  NotificationsActive,
  CheckCircle,
  Error,
  Warning,
  Info,
  Close,
  MarkEmailRead,
  Delete,
  Settings,
} from '@mui/icons-material';
import { formatDistanceToNow } from 'date-fns';
import { notificationService, NotificationData, NotificationType } from '../../lib/services/notificationService';

interface NotificationCenterProps {
  maxDisplayed?: number;
  autoMarkAsRead?: boolean;
  showSettings?: boolean;
}

const NotificationCenter: React.FC<NotificationCenterProps> = ({
  maxDisplayed = 10,
  autoMarkAsRead = true,
  showSettings = true,
}) => {
  const [anchorEl, setAnchorEl] = useState<HTMLButtonElement | null>(null);
  const [notifications, setNotifications] = useState<NotificationData[]>([]);
  const [unreadCount, setUnreadCount] = useState(0);

  useEffect(() => {
    // Initial load
    updateNotifications();

    // Set up polling for new notifications
    const interval = setInterval(updateNotifications, 5000);

    return () => clearInterval(interval);
  }, []);

  const updateNotifications = () => {
    const allNotifications = notificationService.getAll();
    const unread = notificationService.getUnread();
    
    setNotifications(allNotifications.slice(0, maxDisplayed));
    setUnreadCount(unread.length);
  };

  const handleClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    setAnchorEl(event.currentTarget);
    
    if (autoMarkAsRead) {
      // Mark all as read when opening
      setTimeout(() => {
        notificationService.markAllAsRead();
        updateNotifications();
      }, 1000);
    }
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleMarkAsRead = (id: string) => {
    notificationService.markAsRead(id);
    updateNotifications();
  };

  const handleMarkAllAsRead = () => {
    notificationService.markAllAsRead();
    updateNotifications();
  };

  const handleClearAll = () => {
    notificationService.clear();
    updateNotifications();
  };

  const handleCloseNotification = (id: string) => {
    notificationService.close(id);
    updateNotifications();
  };

  const getNotificationIcon = (type: NotificationType) => {
    switch (type) {
      case 'success':
        return <CheckCircle color="success" />;
      case 'error':
        return <Error color="error" />;
      case 'warning':
        return <Warning color="warning" />;
      case 'info':
      default:
        return <Info color="info" />;
    }
  };

  const getNotificationColor = (type: NotificationType) => {
    switch (type) {
      case 'success':
        return 'success';
      case 'error':
        return 'error';
      case 'warning':
        return 'warning';
      case 'info':
      default:
        return 'info';
    }
  };

  const open = Boolean(anchorEl);

  return (
    <>
      <Tooltip title="Notifications">
        <IconButton
          onClick={handleClick}
          color="inherit"
          aria-label="notifications"
        >
          <Badge badgeContent={unreadCount} color="error">
            {unreadCount > 0 ? <NotificationsActive /> : <Notifications />}
          </Badge>
        </IconButton>
      </Tooltip>

      <Popover
        open={open}
        anchorEl={anchorEl}
        onClose={handleClose}
        anchorOrigin={{
          vertical: 'bottom',
          horizontal: 'right',
        }}
        transformOrigin={{
          vertical: 'top',
          horizontal: 'right',
        }}
        PaperProps={{
          sx: {
            width: 400,
            maxHeight: 600,
            overflow: 'hidden',
          },
        }}
      >
        <Box>
          {/* Header */}
          <Box
            sx={{
              p: 2,
              borderBottom: 1,
              borderColor: 'divider',
              bgcolor: 'background.paper',
            }}
          >
            <Stack direction="row" alignItems="center" justifyContent="space-between">
              <Typography variant="h6">
                Notifications
                {unreadCount > 0 && (
                  <Chip
                    label={unreadCount}
                    size="small"
                    color="error"
                    sx={{ ml: 1 }}
                  />
                )}
              </Typography>
              
              <Stack direction="row" spacing={1}>
                {unreadCount > 0 && (
                  <Tooltip title="Mark all as read">
                    <IconButton size="small" onClick={handleMarkAllAsRead}>
                      <MarkEmailRead fontSize="small" />
                    </IconButton>
                  </Tooltip>
                )}
                
                <Tooltip title="Clear all">
                  <IconButton size="small" onClick={handleClearAll}>
                    <Delete fontSize="small" />
                  </IconButton>
                </Tooltip>
                
                {showSettings && (
                  <Tooltip title="Settings">
                    <IconButton size="small">
                      <Settings fontSize="small" />
                    </IconButton>
                  </Tooltip>
                )}
              </Stack>
            </Stack>
          </Box>

          {/* Notifications List */}
          <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
            {notifications.length === 0 ? (
              <Box
                sx={{
                  p: 4,
                  textAlign: 'center',
                  color: 'text.secondary',
                }}
              >
                <Notifications sx={{ fontSize: 48, mb: 2, opacity: 0.5 }} />
                <Typography variant="body2">
                  No notifications yet
                </Typography>
              </Box>
            ) : (
              <List disablePadding>
                {notifications.map((notification, index) => (
                  <React.Fragment key={notification.id}>
                    <ListItem
                      sx={{
                        bgcolor: notification.read ? 'transparent' : 'action.hover',
                        '&:hover': {
                          bgcolor: 'action.selected',
                        },
                      }}
                    >
                      <ListItemIcon>
                        <Avatar
                          sx={{
                            width: 32,
                            height: 32,
                            bgcolor: `${getNotificationColor(notification.type)}.light`,
                          }}
                        >
                          {getNotificationIcon(notification.type)}
                        </Avatar>
                      </ListItemIcon>
                      
                      <ListItemText
                        primary={
                          <Typography
                            variant="body2"
                            sx={{
                              fontWeight: notification.read ? 'normal' : 'medium',
                              pr: 1,
                            }}
                          >
                            {notification.message}
                          </Typography>
                        }
                        secondary={
                          <Stack direction="row" alignItems="center" spacing={1} mt={0.5}>
                            <Chip
                              label={notification.type}
                              size="small"
                              color={getNotificationColor(notification.type) as any}
                              variant="outlined"
                            />
                            <Typography variant="caption" color="text.secondary">
                              {formatDistanceToNow(notification.timestamp, { addSuffix: true })}
                            </Typography>
                          </Stack>
                        }
                      />
                      
                      <Stack direction="column" spacing={1}>
                        {!notification.read && (
                          <Tooltip title="Mark as read">
                            <IconButton
                              size="small"
                              onClick={() => handleMarkAsRead(notification.id)}
                            >
                              <MarkEmailRead fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        )}
                        
                        <Tooltip title="Close">
                          <IconButton
                            size="small"
                            onClick={() => handleCloseNotification(notification.id)}
                          >
                            <Close fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </Stack>
                    </ListItem>
                    
                    {index < notifications.length - 1 && <Divider />}
                  </React.Fragment>
                ))}
              </List>
            )}
          </Box>

          {/* Footer */}
          {notifications.length > 0 && (
            <Box
              sx={{
                p: 2,
                borderTop: 1,
                borderColor: 'divider',
                bgcolor: 'background.paper',
              }}
            >
              <Stack direction="row" spacing={2}>
                <Button
                  size="small"
                  variant="outlined"
                  fullWidth
                  onClick={handleMarkAllAsRead}
                  disabled={unreadCount === 0}
                >
                  Mark All Read
                </Button>
                
                <Button
                  size="small"
                  variant="outlined"
                  fullWidth
                  onClick={handleClearAll}
                  color="error"
                >
                  Clear All
                </Button>
              </Stack>
            </Box>
          )}
        </Box>
      </Popover>
    </>
  );
};

export default NotificationCenter;
