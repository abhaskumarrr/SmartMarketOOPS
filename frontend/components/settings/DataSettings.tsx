import React, { useState } from 'react';
import {
  Box,
  FormControl,
  FormControlLabel,
  FormLabel,
  MenuItem,
  Radio,
  RadioGroup,
  Select,
  SelectChangeEvent,
  TextField,
  Typography,
  Divider,
  Alert,
  Button,
  
  FormHelperText,
} from '@mui/material';
import { usePreferences } from '../../lib/contexts/PreferencesContext';

const DataSettings: React.FC = () => {
  const { preferences, updatePreference } = usePreferences();
  const { data } = preferences;
  
  const [apiEndpoint, setApiEndpoint] = useState(data.customApiEndpoint || '');
  const [showEndpointField, setShowEndpointField] = useState(data.dataSource === 'custom');
  
  const handleRefreshRateChange = (event: SelectChangeEvent<string>) => {
    updatePreference('data', {
      refreshRate: event.target.value as any
    });
  };
  
  const handleDataRangeChange = (event: SelectChangeEvent<string>) => {
    updatePreference('data', {
      historicalDataRange: event.target.value as any
    });
  };
  
  const handleDataSourceChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newDataSource = event.target.value as 'default' | 'custom';
    updatePreference('data', {
      dataSource: newDataSource
    });
    setShowEndpointField(newDataSource === 'custom');
  };
  
  const handleApiEndpointChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setApiEndpoint(event.target.value);
  };
  
  const saveApiEndpoint = () => {
    updatePreference('data', {
      customApiEndpoint: apiEndpoint
    });
  };
  
  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Data Settings
      </Typography>
      <Divider sx={{ mb: 3 }} />
      
      <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", sm: "repeat(2, 1fr)", md: "repeat(3, 1fr)" }, gap: 3 }}>
        {/* Refresh Rate */}
        <Box>
          <FormControl fullWidth>
            <FormLabel id="refresh-rate-label">Data Refresh Rate</FormLabel>
            <Select
              labelId="refresh-rate-label"
              id="refresh-rate"
              value={data.refreshRate}
              onChange={handleRefreshRateChange}
              size="small"
            >
              <MenuItem value="realtime">Real-Time (WebSocket)</MenuItem>
              <MenuItem value="5s">Every 5 seconds</MenuItem>
              <MenuItem value="10s">Every 10 seconds</MenuItem>
              <MenuItem value="30s">Every 30 seconds</MenuItem>
              <MenuItem value="1m">Every minute</MenuItem>
              <MenuItem value="manual">Manual Refresh Only</MenuItem>
            </Select>
            <FormHelperText>
              How frequently to update dashboard data
            </FormHelperText>
          </FormControl>
        </Box>
        
        {/* Historical Data Range */}
        <Box>
          <FormControl fullWidth>
            <FormLabel id="data-range-label">Historical Data Range</FormLabel>
            <Select
              labelId="data-range-label"
              id="data-range"
              value={data.historicalDataRange}
              onChange={handleDataRangeChange}
              size="small"
            >
              <MenuItem value="1d">Last 24 hours</MenuItem>
              <MenuItem value="7d">Last 7 days</MenuItem>
              <MenuItem value="30d">Last 30 days</MenuItem>
              <MenuItem value="90d">Last 90 days</MenuItem>
              <MenuItem value="1y">Last year</MenuItem>
              <MenuItem value="all">All available data</MenuItem>
            </Select>
            <FormHelperText>
              Default time range for charts and metrics
            </FormHelperText>
          </FormControl>
        </Box>
        
        {/* Data Source */}
        <Box>
          <FormControl component="fieldset">
            <FormLabel component="legend">Data Source</FormLabel>
            <RadioGroup
              aria-label="data-source"
              name="data-source"
              value={data.dataSource}
              onChange={handleDataSourceChange}
            >
              <FormControlLabel value="default" control={<Radio />} label="Default API Endpoint" />
              <FormControlLabel value="custom" control={<Radio />} label="Custom API Endpoint" />
            </RadioGroup>
          </FormControl>
        </Box>
        
        {/* Custom API Endpoint */}
        {showEndpointField && (
          <Box>
            <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
              <TextField
                fullWidth
                label="Custom API Endpoint"
                value={apiEndpoint}
                onChange={handleApiEndpointChange}
                placeholder="https://api.example.com/v1"
                variant="outlined"
                size="small"
                helperText="Enter your custom API endpoint URL"
              />
              <Button 
                variant="contained" 
                onClick={saveApiEndpoint}
                sx={{ mt: 1 }}
              >
                Save
              </Button>
            </Box>
            <Alert severity="info" sx={{ mt: 2 }}>
              Using a custom API endpoint may require additional authentication. 
              Make sure your endpoint supports all required data formats.
            </Alert>
          </Box>
        )}
      </Box>
    </Box>
  );
};

export default DataSettings; 