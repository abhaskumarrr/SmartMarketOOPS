import React from 'react';
import {
  Box,
  FormControl,
  FormControlLabel,
  FormGroup,
  FormLabel,
  Switch,
  Typography,
  Divider,
  Slider,
  
  FormHelperText,
} from '@mui/material';
import { usePreferences } from '../../lib/contexts/PreferencesContext';

const NotificationSettings: React.FC = () => {
  const { preferences, updatePreference } = usePreferences();
  const { notifications } = preferences;
  
  const handleToggleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    updatePreference('notifications', {
      [event.target.name]: event.target.checked
    });
  };
  
  const handleConfidenceChange = (_event: Event, newValue: number | number[]) => {
    updatePreference('notifications', {
      minConfidenceThreshold: newValue as number
    });
  };
  
  const confidenceValueText = (value: number) => {
    return `${Math.round(value * 100)}%`;
  };
  
  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Notification Settings
      </Typography>
      <Divider sx={{ mb: 3 }} />
      
      <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", sm: "repeat(2, 1fr)", md: "repeat(3, 1fr)" }, gap: 3 }}>
        {/* Notification Toggles */}
        <Box>
          <FormControl component="fieldset" variant="standard">
            <FormLabel component="legend">Notification Types</FormLabel>
            <FormGroup>
              <FormControlLabel
                control={
                  <Switch 
                    checked={notifications.enablePriceAlerts} 
                    onChange={handleToggleChange}
                    name="enablePriceAlerts"
                  />
                }
                label="Price Alerts"
              />
              <FormControlLabel
                control={
                  <Switch 
                    checked={notifications.enableTradeNotifications} 
                    onChange={handleToggleChange}
                    name="enableTradeNotifications"
                  />
                }
                label="Trade Executions"
              />
              <FormControlLabel
                control={
                  <Switch 
                    checked={notifications.enablePredictionAlerts} 
                    onChange={handleToggleChange}
                    name="enablePredictionAlerts"
                  />
                }
                label="AI Prediction Alerts"
              />
              <FormControlLabel
                control={
                  <Switch 
                    checked={notifications.soundAlerts} 
                    onChange={handleToggleChange}
                    name="soundAlerts"
                  />
                }
                label="Sound Alerts"
              />
            </FormGroup>
          </FormControl>
        </Box>
        
        {/* Confidence Threshold */}
        <Box>
          <Box sx={{ width: '100%', mt: 3 }}>
            <FormControl fullWidth>
              <FormLabel id="confidence-threshold-label">
                Minimum Confidence Threshold
              </FormLabel>
              <Box sx={{ px: 2, pt: 2, pb: 1 }}>
                <Slider
                  aria-labelledby="confidence-threshold-label"
                  value={notifications.minConfidenceThreshold}
                  onChange={handleConfidenceChange}
                  getAriaValueText={confidenceValueText}
                  valueLabelDisplay="auto"
                  valueLabelFormat={confidenceValueText}
                  step={0.05}
                  marks={[
                    { value: 0, label: '0%' },
                    { value: 0.5, label: '50%' },
                    { value: 1, label: '100%' }
                  ]}
                  min={0}
                  max={1}
                  sx={{ mt: 1 }}
                />
              </Box>
              <FormHelperText>
                Only show alerts for predictions with confidence above this threshold
              </FormHelperText>
            </FormControl>
          </Box>
        </Box>
      </Box>
    </Box>
  );
};

export default NotificationSettings; 