import React from 'react';
import {
  Box,
  FormControl,
  FormControlLabel,
  FormLabel,
  Radio,
  RadioGroup,
  Slider,
  Typography,
  Divider,
  Grid,
  FormHelperText,
  useTheme,
  Box as MuiBox,
} from '@mui/material';
import { usePreferences } from '../../lib/contexts/PreferencesContext';

// Color palette options
const PRIMARY_COLORS = [
  { value: '#3f51b5', name: 'Indigo' },
  { value: '#2196f3', name: 'Blue' },
  { value: '#009688', name: 'Teal' },
  { value: '#4caf50', name: 'Green' },
  { value: '#ff9800', name: 'Orange' },
  { value: '#f44336', name: 'Red' },
  { value: '#9c27b0', name: 'Purple' },
  { value: '#795548', name: 'Brown' },
];

const ACCENT_COLORS = [
  { value: '#f50057', name: 'Pink' },
  { value: '#651fff', name: 'Deep Purple' },
  { value: '#2979ff', name: 'Light Blue' },
  { value: '#00e676', name: 'Light Green' },
  { value: '#ffea00', name: 'Yellow' },
  { value: '#ff3d00', name: 'Deep Orange' },
  { value: '#1de9b6', name: 'Teal Accent' },
  { value: '#ff6d00', name: 'Orange Accent' },
];

// Color swatch component
const ColorSwatch = ({ 
  color, 
  selected, 
  onClick 
}: { 
  color: string; 
  selected: boolean; 
  onClick: () => void;
}) => (
  <Box
    sx={{
      width: 40,
      height: 40,
      backgroundColor: color,
      borderRadius: 1,
      cursor: 'pointer',
      border: selected ? '3px solid' : '1px solid',
      borderColor: selected ? 'primary.main' : 'divider',
      boxShadow: selected ? 3 : 1,
      transition: 'all 0.2s',
      '&:hover': {
        transform: 'scale(1.05)',
      },
    }}
    onClick={onClick}
  />
);

const ThemeSettings: React.FC = () => {
  const { preferences, updatePreference } = usePreferences();
  const { theme } = preferences;
  const muiTheme = useTheme();
  
  const handleThemeModeChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    updatePreference('theme', {
      mode: event.target.value as any
    });
  };
  
  const handlePrimaryColorChange = (color: string) => {
    updatePreference('theme', {
      primaryColor: color
    });
  };
  
  const handleAccentColorChange = (color: string) => {
    updatePreference('theme', {
      accentColor: color
    });
  };
  
  const handleFontScaleChange = (_event: Event, newValue: number | number[]) => {
    updatePreference('theme', {
      fontScale: newValue as number
    });
  };
  
  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Theme Settings
      </Typography>
      <Divider sx={{ mb: 3 }} />
      
      <Grid container spacing={3}>
        {/* Theme Mode */}
        <Grid item xs={12}>
          <FormControl component="fieldset">
            <FormLabel component="legend">Color Mode</FormLabel>
            <RadioGroup
              aria-label="theme-mode"
              name="theme-mode"
              value={theme.mode}
              onChange={handleThemeModeChange}
              row
            >
              <FormControlLabel value="light" control={<Radio />} label="Light" />
              <FormControlLabel value="dark" control={<Radio />} label="Dark" />
              <FormControlLabel value="system" control={<Radio />} label="System Default" />
            </RadioGroup>
            <FormHelperText>
              {theme.mode === 'system' 
                ? 'Follows your device settings'
                : `${theme.mode.charAt(0).toUpperCase() + theme.mode.slice(1)} mode is active`}
            </FormHelperText>
          </FormControl>
        </Grid>
        
        {/* Primary Color */}
        <Grid item xs={12}>
          <FormControl component="fieldset" fullWidth>
            <FormLabel component="legend">Primary Color</FormLabel>
            <Box 
              sx={{ 
                display: 'grid', 
                gridTemplateColumns: 'repeat(auto-fill, 40px)', 
                gap: 1,
                mt: 2,
              }}
            >
              {PRIMARY_COLORS.map(color => (
                <ColorSwatch 
                  key={color.value}
                  color={color.value}
                  selected={theme.primaryColor === color.value}
                  onClick={() => handlePrimaryColorChange(color.value)}
                />
              ))}
            </Box>
            <FormHelperText>
              Used for buttons, tabs, and primary elements
            </FormHelperText>
          </FormControl>
        </Grid>
        
        {/* Accent Color */}
        <Grid item xs={12}>
          <FormControl component="fieldset" fullWidth>
            <FormLabel component="legend">Accent Color</FormLabel>
            <Box 
              sx={{ 
                display: 'grid', 
                gridTemplateColumns: 'repeat(auto-fill, 40px)', 
                gap: 1,
                mt: 2,
              }}
            >
              {ACCENT_COLORS.map(color => (
                <ColorSwatch 
                  key={color.value}
                  color={color.value}
                  selected={theme.accentColor === color.value}
                  onClick={() => handleAccentColorChange(color.value)}
                />
              ))}
            </Box>
            <FormHelperText>
              Used for highlights, links, and secondary elements
            </FormHelperText>
          </FormControl>
        </Grid>
        
        {/* Font Scale */}
        <Grid item xs={12}>
          <FormControl component="fieldset" fullWidth>
            <FormLabel component="legend">Font Size Scale</FormLabel>
            <Box sx={{ px: 2, pt: 2, pb: 1 }}>
              <Slider
                value={theme.fontScale}
                onChange={handleFontScaleChange}
                min={0.8}
                max={1.2}
                step={0.05}
                marks={[
                  { value: 0.8, label: 'Smaller' },
                  { value: 1, label: 'Default' },
                  { value: 1.2, label: 'Larger' }
                ]}
                valueLabelDisplay="auto"
                valueLabelFormat={(value) => `${Math.round((value - 1) * 100)}%`}
              />
            </Box>
            <FormHelperText>
              Adjust overall text size throughout the application
            </FormHelperText>
          </FormControl>
        </Grid>
        
        {/* Theme Preview */}
        <Grid item xs={12}>
          <Typography variant="subtitle1" gutterBottom>
            Theme Preview
          </Typography>
          <Box 
            sx={{ 
              border: 1, 
              borderColor: 'divider', 
              borderRadius: 1, 
              p: 2,
              bgcolor: theme.mode === 'dark' ? 'grey.900' : 'grey.100',
              color: theme.mode === 'dark' ? 'common.white' : 'common.black',
            }}
          >
            <Typography variant="h6" sx={{ color: theme.primaryColor }}>
              Primary Color Heading
            </Typography>
            <Typography variant="body1" sx={{ mb: 1 }}>
              This is how your theme will look with the selected settings.
            </Typography>
            <Box 
              sx={{ 
                display: 'flex',
                gap: 1,
                '& a': {
                  color: theme.accentColor,
                  textDecoration: 'none',
                  '&:hover': {
                    textDecoration: 'underline',
                  }
                }
              }}
            >
              <a href="#">Accent color link</a>
              <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                Secondary text
              </Typography>
            </Box>
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ThemeSettings; 