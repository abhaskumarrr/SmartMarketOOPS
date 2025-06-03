import React, { useState } from 'react';
import { Box, Toolbar, AppBar, IconButton, Typography, Badge, useMediaQuery, useTheme, Avatar, Menu, MenuItem, Tooltip } from '@mui/material';
import {
  Menu as MenuIcon,
  Notifications as NotificationsIcon,
  AccountCircle as AccountCircleIcon,
  DarkMode as DarkModeIcon,
  LightMode as LightModeIcon,
} from '@mui/icons-material';
import { Sidebar } from './NavigationSidebar';
import WebSocketStatus from './WebSocketStatus';
import { useThemeContext } from '../../lib/contexts/ThemeContext';
import { useTradingContext } from '../../lib/contexts/TradingContext';

// Define the drawer width
const drawerWidth = 240;
const drawerCollapsedWidth = 72;

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const theme = useTheme();
  const { mode, toggleColorMode } = useThemeContext();
  const { isWebSocketConnected } = useTradingContext();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  
  // State for drawer and menu
  const [drawerOpen, setDrawerOpen] = useState(!isMobile);
  const [mobileDrawerOpen, setMobileDrawerOpen] = useState(false);
  const [accountMenuAnchor, setAccountMenuAnchor] = useState<null | HTMLElement>(null);
  const [notificationsMenuAnchor, setNotificationsMenuAnchor] = useState<null | HTMLElement>(null);
  
  // Toggle drawer handler
  const toggleDrawer = () => {
    setDrawerOpen(!drawerOpen);
  };
  
  // Handle mobile drawer toggle
  const handleMobileDrawerToggle = () => {
    setMobileDrawerOpen(!mobileDrawerOpen);
  };
  
  // Handle account menu
  const handleAccountMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAccountMenuAnchor(event.currentTarget);
  };
  
  const handleAccountMenuClose = () => {
    setAccountMenuAnchor(null);
  };
  
  // Handle notifications menu
  const handleNotificationsMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setNotificationsMenuAnchor(event.currentTarget);
  };
  
  const handleNotificationsMenuClose = () => {
    setNotificationsMenuAnchor(null);
  };

  return (
    <Box sx={{ display: 'flex', height: '100vh' }}>
      {/* App Bar */}
      <AppBar 
        position="fixed" 
        sx={{
          width: { 
            xs: '100%', 
            md: `calc(100% - ${drawerOpen ? drawerWidth : drawerCollapsedWidth}px)` 
          },
          ml: { 
            xs: 0, 
            md: `${drawerOpen ? drawerWidth : drawerCollapsedWidth}px` 
          },
          boxShadow: 'none',
          borderBottom: '1px solid',
          borderBottomColor: 'divider',
          bgcolor: 'background.paper',
          color: 'text.primary',
          transition: theme.transitions.create(['width', 'margin'], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
          }),
          zIndex: theme.zIndex.drawer + 1,
        }}
      >
        <Toolbar sx={{ 
          display: 'flex', 
          justifyContent: 'space-between',
          px: { xs: 1, sm: 2 }
        }}>
          {/* Mobile menu icon */}
          {isMobile && (
            <IconButton
              color="inherit"
              aria-label="open drawer"
              edge="start"
              onClick={handleMobileDrawerToggle}
              sx={{ mr: 2 }}
            >
              <MenuIcon />
            </IconButton>
          )}
          
          {/* WebSocket Status Indicator */}
          <WebSocketStatus isConnected={isWebSocketConnected} />
          
          {/* Right side icons */}
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            {/* Theme toggle on mobile */}
            {isMobile && (
              <IconButton color="inherit" onClick={toggleColorMode} size="large">
                {mode === 'dark' ? <LightModeIcon /> : <DarkModeIcon />}
              </IconButton>
            )}
            
            {/* Notifications */}
            <Tooltip title="Notifications">
              <IconButton
                color="inherit"
                onClick={handleNotificationsMenuOpen}
                size="large"
              >
                <Badge badgeContent={3} color="error">
                  <NotificationsIcon />
                </Badge>
              </IconButton>
            </Tooltip>
            
            {/* User profile */}
            <Tooltip title="Account">
              <IconButton
                color="inherit"
                onClick={handleAccountMenuOpen}
                size="large"
                edge="end"
                sx={{ ml: 1 }}
              >
                <Avatar sx={{ width: 32, height: 32, bgcolor: 'primary.main' }}>
                  <AccountCircleIcon />
                </Avatar>
              </IconButton>
            </Tooltip>
          </Box>
        </Toolbar>
      </AppBar>
      
      {/* Sidebar */}
      <Sidebar />
      
      {/* Account Menu */}
      <Menu
        anchorEl={accountMenuAnchor}
        id="account-menu"
        open={Boolean(accountMenuAnchor)}
        onClose={handleAccountMenuClose}
        transformOrigin={{ horizontal: 'right', vertical: 'top' }}
        anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
        PaperProps={{
          elevation: 3,
          sx: {
            mt: 1,
            overflow: 'visible',
            width: 200,
            borderRadius: 2,
            '&:before': {
              content: '""',
              display: 'block',
              position: 'absolute',
              top: 0,
              right: 14,
              width: 10,
              height: 10,
              bgcolor: 'background.paper',
              transform: 'translateY(-50%) rotate(45deg)',
              zIndex: 0,
            },
          },
        }}
      >
        <MenuItem onClick={handleAccountMenuClose}>Profile</MenuItem>
        <MenuItem onClick={handleAccountMenuClose}>My Account</MenuItem>
        <MenuItem onClick={handleAccountMenuClose}>Settings</MenuItem>
        <MenuItem onClick={handleAccountMenuClose}>Logout</MenuItem>
      </Menu>
      
      {/* Notifications Menu */}
      <Menu
        anchorEl={notificationsMenuAnchor}
        id="notifications-menu"
        open={Boolean(notificationsMenuAnchor)}
        onClose={handleNotificationsMenuClose}
        transformOrigin={{ horizontal: 'right', vertical: 'top' }}
        anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
        PaperProps={{
          elevation: 3,
          sx: {
            mt: 1,
            overflow: 'visible',
            width: 320,
            maxHeight: 400,
            borderRadius: 2,
            '&:before': {
              content: '""',
              display: 'block',
              position: 'absolute',
              top: 0,
              right: 14,
              width: 10,
              height: 10,
              bgcolor: 'background.paper',
              transform: 'translateY(-50%) rotate(45deg)',
              zIndex: 0,
            },
          },
        }}
      >
        <MenuItem sx={{ borderBottom: '1px solid', borderBottomColor: 'divider' }}>
          <Typography variant="subtitle1" fontWeight="bold">Notifications</Typography>
        </MenuItem>
        <MenuItem onClick={handleNotificationsMenuClose}>
          <Box>
            <Typography variant="body2" fontWeight="bold">BTC Buy Signal</Typography>
            <Typography variant="caption" color="text.secondary">
              Strong buy signal detected at $40,250. 87% confidence.
            </Typography>
          </Box>
        </MenuItem>
        <MenuItem onClick={handleNotificationsMenuClose}>
          <Box>
            <Typography variant="body2" fontWeight="bold">Account Alert</Typography>
            <Typography variant="caption" color="text.secondary">
              API key for Binance will expire in 3 days.
            </Typography>
          </Box>
        </MenuItem>
        <MenuItem onClick={handleNotificationsMenuClose}>
          <Box>
            <Typography variant="body2" fontWeight="bold">ETH Sell Signal</Typography>
            <Typography variant="caption" color="text.secondary">
              Sell signal for ETH at $2,845. 75% confidence.
            </Typography>
          </Box>
        </MenuItem>
        <MenuItem sx={{ borderTop: '1px solid', borderTopColor: 'divider', justifyContent: 'center' }}>
          <Typography variant="body2" color="primary">
            View all notifications
          </Typography>
        </MenuItem>
      </Menu>
      
      {/* Main content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          width: { 
            xs: '100%', 
            md: `calc(100% - ${drawerOpen ? drawerWidth : drawerCollapsedWidth}px)` 
          },
          ml: { 
            xs: 0, 
            md: `${drawerOpen ? drawerWidth : drawerCollapsedWidth}px` 
          },
          transition: theme.transitions.create(['width', 'margin'], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
          }),
          height: '100vh',
          overflow: 'auto',
          bgcolor: 'background.default',
        }}
      >
        <Toolbar /> {/* Empty toolbar to push content below app bar */}
        {children}
      </Box>
    </Box>
  );
};

export default Layout; 