import React, { useState } from 'react';
import { 
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Box,
  Typography,
  Divider,
  Collapse,
  Tooltip,
  IconButton,
  useMediaQuery,
  useTheme
} from '@mui/material';
import { 
  Dashboard as DashboardIcon,
  ShowChart as ShowChartIcon,
  AccountBalanceWallet as WalletIcon,
  AccountBalance as AccountBalanceIcon,
  SwapHoriz as SwapHorizIcon,
  TrendingUp as TrendingUpIcon,
  BarChart as BarChartIcon,
  SignalCellularAlt as SignalIcon,
  Assessment as AssessmentIcon,
  Settings as SettingsIcon,
  ExpandLess as ExpandLessIcon,
  ExpandMore as ExpandMoreIcon,
  ChevronLeft as ChevronLeftIcon,
  ChevronRight as ChevronRightIcon,
  Brightness4 as DarkModeIcon,
  Brightness7 as LightModeIcon
} from '@mui/icons-material';
import Link from 'next/link';
import { useRouter } from 'next/router';
import { useThemeContext } from '../../lib/contexts/ThemeContext';

// Define the drawer width
const drawerWidth = 240;
const drawerCollapsedWidth = 72;

// Sidebar component
const Sidebar: React.FC<{
  open?: boolean;
  onToggle?: () => void;
  mobileOpen?: boolean;
  onMobileClose?: () => void;
}> = ({ 
  open = true, 
  onToggle,
  mobileOpen = false,
  onMobileClose
}) => {
  const theme = useTheme();
  const { mode, toggleColorMode } = useThemeContext();
  const router = useRouter();
  const pathname = router.pathname;
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  
  // State for collapsible menus
  const [tradingOpen, setTradingOpen] = useState(pathname.includes('/trading'));
  const [analysisOpen, setAnalysisOpen] = useState(pathname.includes('/analysis'));

  // Toggle submenu handlers
  const handleTradingClick = () => {
    setTradingOpen(!tradingOpen);
  };

  const handleAnalysisClick = () => {
    setAnalysisOpen(!analysisOpen);
  };

  // Create the sidebar content
  const sidebarContent = (
    <>
      <Toolbar sx={{ 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: open ? 'space-between' : 'center',
        py: 2,
      }}>
        {open && (
          <Typography variant="h6" fontWeight="bold" noWrap>
            SmartMarket
          </Typography>
        )}
        {!open && (
          <Typography variant="h6" fontWeight="bold" noWrap>
            SM
          </Typography>
        )}
        
        {!isMobile && (
          <IconButton onClick={onToggle} edge="end">
            {open ? <ChevronLeftIcon /> : <ChevronRightIcon />}
          </IconButton>
        )}
      </Toolbar>
      
      <Divider />
      
      <List>
        <ListItem disablePadding>
          <ListItemButton 
            component={Link} 
            href="/dashboard"
            selected={pathname === '/dashboard'}
          >
            <ListItemIcon>
              <DashboardIcon />
            </ListItemIcon>
            {open && <ListItemText primary="Dashboard" />}
          </ListItemButton>
        </ListItem>
        
        <ListItem disablePadding>
          <ListItemButton 
            component={Link} 
            href="/markets"
            selected={pathname === '/markets'}
          >
            <ListItemIcon>
              <ShowChartIcon />
            </ListItemIcon>
            {open && <ListItemText primary="Markets" />}
          </ListItemButton>
        </ListItem>
        
        <ListItem disablePadding>
          <ListItemButton 
            component={Link} 
            href="/portfolio"
            selected={pathname === '/portfolio'}
          >
            <ListItemIcon>
              <WalletIcon />
            </ListItemIcon>
            {open && <ListItemText primary="Portfolio" />}
          </ListItemButton>
        </ListItem>
        
        <ListItem disablePadding>
          <ListItemButton onClick={handleTradingClick}>
            <ListItemIcon>
              <SwapHorizIcon />
            </ListItemIcon>
            {open && (
              <>
                <ListItemText primary="Trading" />
                {tradingOpen ? <ExpandLessIcon /> : <ExpandMoreIcon />}
              </>
            )}
          </ListItemButton>
        </ListItem>
        
        <Collapse in={tradingOpen && open} timeout="auto" unmountOnExit>
          <List component="div" disablePadding>
            <ListItemButton 
              component={Link} 
              href="/trading/spot"
              selected={pathname === '/trading/spot'}
              sx={{ pl: 4 }}
            >
              <ListItemIcon>
                <SwapHorizIcon />
              </ListItemIcon>
              <ListItemText primary="Spot" />
            </ListItemButton>
            <ListItemButton 
              component={Link} 
              href="/trading/futures"
              selected={pathname === '/trading/futures'}
              sx={{ pl: 4 }}
            >
              <ListItemIcon>
                <TrendingUpIcon />
              </ListItemIcon>
              <ListItemText primary="Futures" />
            </ListItemButton>
          </List>
        </Collapse>
        
        {!open && (
          <>
            <ListItem disablePadding>
              <Tooltip title="Spot Trading" placement="right">
                <ListItemButton 
                  component={Link} 
                  href="/trading/spot"
                  selected={pathname === '/trading/spot'}
                >
                  <ListItemIcon>
                    <SwapHorizIcon />
                  </ListItemIcon>
                </ListItemButton>
              </Tooltip>
            </ListItem>
            <ListItem disablePadding>
              <Tooltip title="Futures Trading" placement="right">
                <ListItemButton 
                  component={Link} 
                  href="/trading/futures"
                  selected={pathname === '/trading/futures'}
                >
                  <ListItemIcon>
                    <TrendingUpIcon />
                  </ListItemIcon>
                </ListItemButton>
              </Tooltip>
            </ListItem>
          </>
        )}
        
        <ListItem disablePadding>
          <ListItemButton onClick={handleAnalysisClick}>
            <ListItemIcon>
              <BarChartIcon />
            </ListItemIcon>
            {open && (
              <>
                <ListItemText primary="Analysis" />
                {analysisOpen ? <ExpandLessIcon /> : <ExpandMoreIcon />}
              </>
            )}
          </ListItemButton>
        </ListItem>
        
        <Collapse in={analysisOpen && open} timeout="auto" unmountOnExit>
          <List component="div" disablePadding>
            <ListItemButton 
              component={Link} 
              href="/analysis/signals"
              selected={pathname === '/analysis/signals'}
              sx={{ pl: 4 }}
            >
              <ListItemIcon>
                <SignalIcon />
              </ListItemIcon>
              <ListItemText primary="Signals" />
            </ListItemButton>
            <ListItemButton 
              component={Link} 
              href="/analysis/performance"
              selected={pathname === '/analysis/performance'}
              sx={{ pl: 4 }}
            >
              <ListItemIcon>
                <AssessmentIcon />
              </ListItemIcon>
              <ListItemText primary="Performance" />
            </ListItemButton>
          </List>
        </Collapse>
        
        {!open && (
          <>
            <ListItem disablePadding>
              <Tooltip title="Trading Signals" placement="right">
                <ListItemButton 
                  component={Link} 
                  href="/analysis/signals"
                  selected={pathname === '/analysis/signals'}
                >
                  <ListItemIcon>
                    <SignalIcon />
                  </ListItemIcon>
                </ListItemButton>
              </Tooltip>
            </ListItem>
            <ListItem disablePadding>
              <Tooltip title="Performance Metrics" placement="right">
                <ListItemButton 
                  component={Link} 
                  href="/analysis/performance"
                  selected={pathname === '/analysis/performance'}
                >
                  <ListItemIcon>
                    <AssessmentIcon />
                  </ListItemIcon>
                </ListItemButton>
              </Tooltip>
            </ListItem>
          </>
        )}
      </List>
      
      <Box sx={{ flexGrow: 1 }} />
      
      <List>
        <ListItem disablePadding>
          <ListItemButton onClick={toggleColorMode}>
            <ListItemIcon>
              {mode === 'dark' ? <LightModeIcon /> : <DarkModeIcon />}
            </ListItemIcon>
            {open && <ListItemText primary={mode === 'dark' ? 'Light Mode' : 'Dark Mode'} />}
          </ListItemButton>
        </ListItem>
        <ListItem disablePadding>
          <ListItemButton 
            component={Link} 
            href="/settings"
            selected={pathname === '/settings'}
          >
            <ListItemIcon>
              <SettingsIcon />
            </ListItemIcon>
            {open && <ListItemText primary="Settings" />}
          </ListItemButton>
        </ListItem>
      </List>
    </>
  );

  return (
    <>
      {/* Desktop Drawer */}
      {!isMobile && (
        <Drawer
          variant="permanent"
          sx={{
            width: open ? drawerWidth : drawerCollapsedWidth,
            flexShrink: 0,
            [`& .MuiDrawer-paper`]: { 
              width: open ? drawerWidth : drawerCollapsedWidth, 
              boxSizing: 'border-box',
              overflowX: 'hidden',
              transition: theme.transitions.create('width', {
                easing: theme.transitions.easing.sharp,
                duration: theme.transitions.duration.enteringScreen,
              }),
            },
          }}
        >
          {sidebarContent}
        </Drawer>
      )}
      
      {/* Mobile Drawer */}
      {isMobile && (
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={onMobileClose}
          ModalProps={{
            keepMounted: true, // Better open performance on mobile
          }}
          sx={{
            display: { xs: 'block', md: 'none' },
            '& .MuiDrawer-paper': { width: drawerWidth },
          }}
        >
          {sidebarContent}
        </Drawer>
      )}
    </>
  );
};

export default Sidebar; 