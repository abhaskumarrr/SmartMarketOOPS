import React from 'react';
import { IconButton, Tooltip, Switch, FormControlLabel, Box, useMediaQuery, Theme } from '@mui/material';
import { Brightness4 as DarkModeIcon, Brightness7 as LightModeIcon } from '@mui/icons-material';

interface ThemeToggleProps {
  darkMode: boolean;
  toggleDarkMode: () => void;
  variant?: 'icon' | 'switch' | 'combined';
  size?: 'small' | 'medium' | 'large';
  tooltipPlacement?: 'top' | 'right' | 'bottom' | 'left';
  showLabel?: boolean;
}

const ThemeToggle: React.FC<ThemeToggleProps> = ({
  darkMode,
  toggleDarkMode,
  variant = 'icon',
  size = 'medium',
  tooltipPlacement = 'bottom',
  showLabel = false
}) => {
  const isMobile = useMediaQuery((theme: Theme) => theme.breakpoints.down('sm'));
  const label = darkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode';
  
  // Icon button variant
  if (variant === 'icon') {
    return (
      <Tooltip title={label} placement={tooltipPlacement} arrow>
        <IconButton 
          onClick={toggleDarkMode} 
          color="inherit"
          aria-label={label}
          size={size}
          sx={{
            transition: 'transform 0.3s ease',
            '&:hover': {
              transform: 'rotate(12deg)'
            }
          }}
        >
          {darkMode ? <LightModeIcon /> : <DarkModeIcon />}
        </IconButton>
      </Tooltip>
    );
  }
  
  // Switch variant
  if (variant === 'switch') {
    return (
      <FormControlLabel
        control={
          <Switch
            checked={darkMode}
            onChange={toggleDarkMode}
            name="themeMode"
            size={size === 'large' ? 'medium' : size}
            color="primary"
            inputProps={{ 'aria-label': label }}
          />
        }
        label={showLabel ? (darkMode ? 'Dark' : 'Light') : ''}
        sx={{ 
          ml: 0,
          mr: 0
        }}
      />
    );
  }
  
  // Combined variant (icon + switch)
  return (
    <Box 
      sx={{ 
        display: 'flex', 
        alignItems: 'center',
        border: theme => `1px solid ${theme.palette.divider}`,
        borderRadius: 1,
        px: 1,
        backgroundColor: theme => 
          theme.palette.mode === 'dark' 
            ? 'rgba(255, 255, 255, 0.05)' 
            : 'rgba(0, 0, 0, 0.04)'
      }}
    >
      <LightModeIcon 
        fontSize={size} 
        color={!darkMode ? 'primary' : 'disabled'} 
        sx={{ opacity: !darkMode ? 1 : 0.5 }}
      />
      
      <Switch
        checked={darkMode}
        onChange={toggleDarkMode}
        name="themeMode"
        size={size === 'large' ? 'medium' : size}
        color="primary"
        inputProps={{ 'aria-label': label }}
      />
      
      <DarkModeIcon 
        fontSize={size} 
        color={darkMode ? 'primary' : 'disabled'}
        sx={{ opacity: darkMode ? 1 : 0.5 }}
      />
    </Box>
  );
};

export default ThemeToggle; 