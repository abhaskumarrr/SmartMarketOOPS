import React from 'react';
import {
  Box,
  FormControl,
  FormControlLabel,
  FormGroup,
  FormLabel,
  MenuItem,
  Select,
  Switch,
  Typography,
  Divider,
  Grid,
  Chip,
  OutlinedInput,
  SelectChangeEvent,
  FormHelperText,
} from '@mui/material';
import { usePreferences } from '../../lib/contexts/PreferencesContext';

// Available indicators for charts
const AVAILABLE_INDICATORS = [
  'SMA', 'EMA', 'MACD', 'RSI', 'Bollinger Bands', 'Volume', 'VWAP',
  'Stochastic', 'ATR', 'OBV', 'Ichimoku', 'Fibonacci', 'Pivot Points'
];

const ChartSettings: React.FC = () => {
  const { preferences, updatePreference } = usePreferences();
  const { chart } = preferences;
  
  const handleTimeframeChange = (event: SelectChangeEvent<string>) => {
    updatePreference('chart', {
      defaultTimeframe: event.target.value as any
    });
  };
  
  const handleChartStyleChange = (event: SelectChangeEvent<string>) => {
    updatePreference('chart', {
      chartStyle: event.target.value as any
    });
  };
  
  const handleChartThemeChange = (event: SelectChangeEvent<string>) => {
    updatePreference('chart', {
      chartTheme: event.target.value as any
    });
  };
  
  const handleToggleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    updatePreference('chart', {
      [event.target.name]: event.target.checked
    });
  };
  
  const handleIndicatorChange = (event: SelectChangeEvent<string[]>) => {
    const value = event.target.value;
    
    // On autofill we get a stringified value.
    const indicators = typeof value === 'string' ? value.split(',') : value;
    
    updatePreference('chart', {
      defaultIndicators: indicators
    });
  };
  
  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Chart Preferences
      </Typography>
      <Divider sx={{ mb: 3 }} />
      
      <Grid container spacing={3}>
        {/* Default Timeframe */}
        <Grid item xs={12} sm={6}>
          <FormControl fullWidth>
            <FormLabel id="default-timeframe-label">Default Timeframe</FormLabel>
            <Select
              labelId="default-timeframe-label"
              id="default-timeframe"
              value={chart.defaultTimeframe}
              onChange={handleTimeframeChange}
              size="small"
            >
              <MenuItem value="1m">1 Minute</MenuItem>
              <MenuItem value="5m">5 Minutes</MenuItem>
              <MenuItem value="15m">15 Minutes</MenuItem>
              <MenuItem value="1h">1 Hour</MenuItem>
              <MenuItem value="4h">4 Hours</MenuItem>
              <MenuItem value="1d">1 Day</MenuItem>
              <MenuItem value="1w">1 Week</MenuItem>
            </Select>
            <FormHelperText>
              Sets the default time interval for chart candles
            </FormHelperText>
          </FormControl>
        </Grid>
        
        {/* Chart Style */}
        <Grid item xs={12} sm={6}>
          <FormControl fullWidth>
            <FormLabel id="chart-style-label">Chart Style</FormLabel>
            <Select
              labelId="chart-style-label"
              id="chart-style"
              value={chart.chartStyle}
              onChange={handleChartStyleChange}
              size="small"
            >
              <MenuItem value="candles">Candlestick</MenuItem>
              <MenuItem value="line">Line</MenuItem>
              <MenuItem value="bars">Bars</MenuItem>
            </Select>
            <FormHelperText>
              Sets how price data is visualized
            </FormHelperText>
          </FormControl>
        </Grid>
        
        {/* Chart Theme */}
        <Grid item xs={12} sm={6}>
          <FormControl fullWidth>
            <FormLabel id="chart-theme-label">Chart Theme</FormLabel>
            <Select
              labelId="chart-theme-label"
              id="chart-theme"
              value={chart.chartTheme}
              onChange={handleChartThemeChange}
              size="small"
            >
              <MenuItem value="light">Light</MenuItem>
              <MenuItem value="dark">Dark</MenuItem>
              <MenuItem value="system">System (Auto)</MenuItem>
            </Select>
            <FormHelperText>
              Chart-specific theme (can differ from app theme)
            </FormHelperText>
          </FormControl>
        </Grid>
        
        {/* Toggle Options */}
        <Grid item xs={12}>
          <FormControl component="fieldset" variant="standard">
            <FormLabel component="legend">Display Options</FormLabel>
            <FormGroup>
              <FormControlLabel
                control={
                  <Switch 
                    checked={chart.showPredictions} 
                    onChange={handleToggleChange}
                    name="showPredictions"
                  />
                }
                label="Show AI Predictions on Chart"
              />
              <FormControlLabel
                control={
                  <Switch 
                    checked={chart.showTrades} 
                    onChange={handleToggleChange}
                    name="showTrades"
                  />
                }
                label="Show Trade Executions on Chart"
              />
            </FormGroup>
          </FormControl>
        </Grid>
        
        {/* Default Indicators */}
        <Grid item xs={12}>
          <FormControl fullWidth>
            <FormLabel id="default-indicators-label">Default Indicators</FormLabel>
            <Select
              labelId="default-indicators-label"
              id="default-indicators"
              multiple
              value={chart.defaultIndicators}
              onChange={handleIndicatorChange}
              input={<OutlinedInput id="select-multiple-indicators" />}
              renderValue={(selected) => (
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                  {selected.map((value) => (
                    <Chip key={value} label={value} />
                  ))}
                </Box>
              )}
              size="small"
              sx={{ minHeight: '56px' }}
            >
              {AVAILABLE_INDICATORS.map((indicator) => (
                <MenuItem
                  key={indicator}
                  value={indicator}
                >
                  {indicator}
                </MenuItem>
              ))}
            </Select>
            <FormHelperText>
              Select indicators to show by default when opening a chart
            </FormHelperText>
          </FormControl>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ChartSettings; 