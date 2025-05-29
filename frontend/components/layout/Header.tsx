import React, { useState } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Box,
  Badge,
  Breadcrumbs,
  Link,
  Chip,
  Avatar,
  Menu,
  MenuItem,
  Divider,
  useMediaQuery,
  Theme
} from '@mui/material';
import {
  Menu as MenuIcon,
  Notifications as NotificationsIcon,
  AccountCircle as AccountCircleIcon,
  Settings as SettingsIcon,
  Help as HelpIcon,
  NavigateNext as NavigateNextIcon,
  Search as SearchIcon,
} from '@mui/icons-material';
import ThemeToggle from './ThemeToggle';
import WebSocketStatus from './WebSocketStatus';
import SettingsPanel from '../SettingsPanel';

interface HeaderProps {
  darkMode: boolean;
  toggleDarkMode: () => void;
  onMenuToggle: () => void;
  open: boolean;
  title: string;
  breadcrumbs?: { text: string; href?: string }[];
}

export const Header: React.FC<HeaderProps> = ({
  darkMode,
  toggleDarkMode,
  onMenuToggle,
  open,
  title,
  breadcrumbs = [],
}) => {
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const isMobile = useMediaQuery((theme: Theme) => theme.breakpoints.down('sm'));

  const handleProfileMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleSettingsOpen = () => {
    setSettingsOpen(true);
  };

  const handleSettingsClose = () => {
    setSettingsOpen(false);
  };

  return (
    <>
      <AppBar
        position="fixed"
        sx={{
          zIndex: (theme) => theme.zIndex.drawer + 1,
          transition: (theme) =>
            theme.transitions.create(['width', 'margin'], {
              easing: theme.transitions.easing.sharp,
              duration: theme.transitions.duration.leavingScreen,
            }),
          ...(open && {
            marginLeft: '240px',
            width: 'calc(100% - 240px)',
            transition: (theme) =>
              theme.transitions.create(['width', 'margin'], {
                easing: theme.transitions.easing.sharp,
                duration: theme.transitions.duration.enteringScreen,
              }),
          }),
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            onClick={onMenuToggle}
            edge="start"
            sx={{
              marginRight: 2,
            }}
          >
            <MenuIcon />
          </IconButton>

          <Box sx={{ display: { xs: 'none', sm: 'block' } }}>
            <Typography variant="h6" noWrap component="div">
              {title}
            </Typography>
          </Box>

          {breadcrumbs.length > 0 && !isMobile && (
            <Breadcrumbs
              separator={<NavigateNextIcon fontSize="small" />}
              aria-label="breadcrumb"
              sx={{ ml: 2, color: 'text.primary' }}
            >
              {breadcrumbs.map((item, index) => {
                const isLast = index === breadcrumbs.length - 1;
                return item.href && !isLast ? (
                  <Link
                    key={item.text}
                    color="inherit"
                    href={item.href}
                    underline="hover"
                  >
                    {item.text}
                  </Link>
                ) : (
                  <Typography
                    key={item.text}
                    color={isLast ? 'text.primary' : 'inherit'}
                    sx={{ fontWeight: isLast ? 'bold' : 'normal' }}
                  >
                    {item.text}
                  </Typography>
                );
              })}
            </Breadcrumbs>
          )}

          <Box sx={{ flexGrow: 1 }} />

          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <WebSocketStatus />
            
            <IconButton color="inherit" size="large">
              <Badge badgeContent={4} color="error">
                <NotificationsIcon />
              </Badge>
            </IconButton>

            <ThemeToggle darkMode={darkMode} toggleDarkMode={toggleDarkMode} />

            <IconButton 
              color="inherit" 
              size="large" 
              onClick={handleSettingsOpen}
              aria-label="settings"
            >
              <SettingsIcon />
            </IconButton>

            <IconButton
              edge="end"
              aria-label="account of current user"
              aria-haspopup="true"
              onClick={handleProfileMenuOpen}
              color="inherit"
              size="large"
            >
              <AccountCircleIcon />
            </IconButton>

            <Menu
              anchorEl={anchorEl}
              anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
              transformOrigin={{ vertical: 'top', horizontal: 'right' }}
              id="menu-appbar"
              keepMounted
              open={Boolean(anchorEl)}
              onClose={handleMenuClose}
            >
              <Box sx={{ py: 1, px: 2 }}>
                <Typography variant="subtitle1">John Doe</Typography>
                <Typography variant="body2" color="text.secondary">
                  john.doe@example.com
                </Typography>
                <Chip
                  label="Pro Plan"
                  size="small"
                  color="primary"
                  sx={{ mt: 1 }}
                />
              </Box>
              <Divider />
              <MenuItem onClick={handleMenuClose}>My Profile</MenuItem>
              <MenuItem onClick={handleMenuClose}>Account Settings</MenuItem>
              <MenuItem onClick={handleMenuClose}>Logout</MenuItem>
            </Menu>
          </Box>
        </Toolbar>
      </AppBar>

      <SettingsPanel 
        open={settingsOpen} 
        onClose={handleSettingsClose} 
      />
    </>
  );
};

export default Header; 