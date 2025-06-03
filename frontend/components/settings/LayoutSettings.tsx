import React from 'react';
import {
  Box,
  FormControl,
  FormControlLabel,
  FormLabel,
  Radio,
  RadioGroup,
  Switch,
  Typography,
  Divider,
  Checkbox,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Paper,
} from '@mui/material';
import { usePreferences } from '../../lib/contexts/PreferencesContext';

// Available widgets
const AVAILABLE_WIDGETS = [
  { id: 'chart', name: 'Price Chart' },
  { id: 'trades', name: 'Trade History' },
  { id: 'predictions', name: 'Predictions' },
  { id: 'performance', name: 'Performance Metrics' },
  { id: 'signals', name: 'Trading Signals' },
  { id: 'news', name: 'Market News' },
  { id: 'alerts', name: 'Alerts & Notifications' },
  { id: 'watchlist', name: 'Watchlist' },
];

const LayoutSettings: React.FC = () => {
  const { preferences, updatePreference } = usePreferences();
  const { layout } = preferences;
  
  const handleLayoutChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    updatePreference('layout', {
      dashboardLayout: event.target.value as any
    });
  };
  
  const handleSidebarToggle = (event: React.ChangeEvent<HTMLInputElement>) => {
    updatePreference('layout', {
      sidebarCollapsed: event.target.checked
    });
  };
  
  const handleWidgetToggle = (widgetId: string) => {
    const currentWidgets = [...layout.visibleWidgets];
    
    if (currentWidgets.includes(widgetId)) {
      // Remove widget
      const updatedWidgets = currentWidgets.filter(id => id !== widgetId);
      updatePreference('layout', {
        visibleWidgets: updatedWidgets
      });
    } else {
      // Add widget
      updatePreference('layout', {
        visibleWidgets: [...currentWidgets, widgetId]
      });
    }
  };
  
  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Layout Settings
      </Typography>
      <Divider sx={{ mb: 3 }} />
      
      {/* Layout Type */}
      <Box sx={{ mb: 4 }}>
        <FormControl component="fieldset">
          <FormLabel component="legend">Dashboard Layout</FormLabel>
          <RadioGroup
            aria-label="dashboard-layout"
            name="dashboard-layout"
            value={layout.dashboardLayout}
            onChange={handleLayoutChange}
          >
            <FormControlLabel value="compact" control={<Radio />} label="Compact (Small screens)" />
            <FormControlLabel value="expanded" control={<Radio />} label="Expanded (Large screens)" />
            <FormControlLabel value="custom" control={<Radio />} label="Custom (Drag & drop)" />
          </RadioGroup>
        </FormControl>
      </Box>
      
      {/* Sidebar Toggle */}
      <Box sx={{ mb: 4 }}>
        <FormControl component="fieldset">
          <FormLabel component="legend">Sidebar Options</FormLabel>
          <FormControlLabel
            control={
              <Switch 
                checked={layout.sidebarCollapsed} 
                onChange={handleSidebarToggle}
              />
            }
            label="Start with collapsed sidebar"
          />
        </FormControl>
      </Box>
      
      {/* Visible Widgets */}
      <Box>
        <FormControl component="fieldset" sx={{ width: '100%' }}>
          <FormLabel component="legend">Visible Widgets</FormLabel>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1, mb: 2 }}>
            Select which widgets to show on your dashboard
          </Typography>
          
          <Paper variant="outlined" sx={{ mt: 1 }}>
            <List dense>
              {AVAILABLE_WIDGETS.map((widget) => {
                const isChecked = layout.visibleWidgets.includes(widget.id);
                
                return (
                  <ListItem
                    key={widget.id}
                    component="div"
                    sx={{ cursor: 'pointer' }}
                    onClick={() => handleWidgetToggle(widget.id)}
                  >
                    <ListItemIcon>
                      <Checkbox
                        edge="start"
                        checked={isChecked}
                        tabIndex={-1}
                        disableRipple
                      />
                    </ListItemIcon>
                    <ListItemText primary={widget.name} />
                  </ListItem>
                );
              })}
            </List>
          </Paper>
        </FormControl>
      </Box>
    </Box>
  );
};

export default LayoutSettings; 