/**
 * Create Bot Dialog Component
 * Form for creating new Delta Exchange trading bots
 */

'use client';

import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
  Typography,
  Box,
  Chip,
  Alert,
  InputAdornment,
} from '@mui/material';
import {
  SmartToy as BotIcon,
  TrendingUp as TrendingUpIcon,
} from '@mui/icons-material';

interface CreateBotDialogProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (config: any) => Promise<void>;
}

export function CreateBotDialog({ open, onClose, onSubmit }: CreateBotDialogProps) {
  const [formData, setFormData] = useState({
    name: '',
    symbol: 'BTCUSD',
    strategy: 'momentum',
    capital: 1000,
    leverage: 3,
    riskPerTrade: 2,
    maxPositions: 3,
    stopLoss: 5,
    takeProfit: 10,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const symbols = [
    { value: 'BTCUSD', label: 'BTC/USD', description: 'Bitcoin Perpetual' },
    { value: 'ETHUSD', label: 'ETH/USD', description: 'Ethereum Perpetual' },
  ];

  const strategies = [
    { value: 'momentum', label: 'Momentum', description: 'Trend-following strategy' },
    { value: 'mean_reversion', label: 'Mean Reversion', description: 'Counter-trend strategy' },
    { value: 'scalping', label: 'Scalping', description: 'High-frequency trading' },
    { value: 'swing', label: 'Swing Trading', description: 'Medium-term positions' },
  ];

  const handleInputChange = (field: string, value: any) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
    setError(null);
  };

  const validateForm = () => {
    if (!formData.name.trim()) {
      setError('Bot name is required');
      return false;
    }
    if (formData.capital <= 0) {
      setError('Capital must be greater than 0');
      return false;
    }
    if (formData.leverage < 1 || formData.leverage > 100) {
      setError('Leverage must be between 1 and 100');
      return false;
    }
    if (formData.riskPerTrade <= 0 || formData.riskPerTrade > 100) {
      setError('Risk per trade must be between 0 and 100');
      return false;
    }
    return true;
  };

  const handleSubmit = async () => {
    if (!validateForm()) return;

    try {
      setLoading(true);
      setError(null);

      const config = {
        ...formData,
        id: `bot-${Date.now()}`, // Generate unique ID
        enabled: true,
        testnet: true,
      };

      await onSubmit(config);
      
      // Reset form
      setFormData({
        name: '',
        symbol: 'BTCUSD',
        strategy: 'momentum',
        capital: 1000,
        leverage: 3,
        riskPerTrade: 2,
        maxPositions: 3,
        stopLoss: 5,
        takeProfit: 10,
      });
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to create bot');
    } finally {
      setLoading(false);
    }
  };

  const calculateMaxLoss = () => {
    return (formData.capital * formData.riskPerTrade) / 100;
  };

  const calculatePositionSize = () => {
    return (formData.capital * formData.leverage * formData.riskPerTrade) / 100;
  };

  return (
    <Dialog 
      open={open} 
      onClose={onClose} 
      maxWidth="md" 
      fullWidth
      PaperProps={{
        sx: { bgcolor: 'background.paper' }
      }}
    >
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <BotIcon sx={{ color: 'primary.main' }} />
          <Box>
            <Typography variant="h6">Create Trading Bot</Typography>
            <Typography variant="caption" color="text.secondary">
              Configure your automated Delta Exchange trading bot
            </Typography>
          </Box>
        </Box>
      </DialogTitle>

      <DialogContent>
        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        <Grid container spacing={3}>
          {/* Basic Configuration */}
          <Grid item xs={12}>
            <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 600 }}>
              Basic Configuration
            </Typography>
          </Grid>

          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label="Bot Name"
              value={formData.name}
              onChange={(e) => handleInputChange('name', e.target.value)}
              placeholder="e.g., BTC Momentum Bot"
              required
            />
          </Grid>

          <Grid item xs={12} md={6}>
            <FormControl fullWidth>
              <InputLabel>Trading Symbol</InputLabel>
              <Select
                value={formData.symbol}
                onChange={(e) => handleInputChange('symbol', e.target.value)}
                label="Trading Symbol"
              >
                {symbols.map((symbol) => (
                  <MenuItem key={symbol.value} value={symbol.value}>
                    <Box>
                      <Typography variant="body2">{symbol.label}</Typography>
                      <Typography variant="caption" color="text.secondary">
                        {symbol.description}
                      </Typography>
                    </Box>
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12}>
            <FormControl fullWidth>
              <InputLabel>Trading Strategy</InputLabel>
              <Select
                value={formData.strategy}
                onChange={(e) => handleInputChange('strategy', e.target.value)}
                label="Trading Strategy"
              >
                {strategies.map((strategy) => (
                  <MenuItem key={strategy.value} value={strategy.value}>
                    <Box>
                      <Typography variant="body2">{strategy.label}</Typography>
                      <Typography variant="caption" color="text.secondary">
                        {strategy.description}
                      </Typography>
                    </Box>
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>

          {/* Risk Management */}
          <Grid item xs={12}>
            <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 600 }}>
              Risk Management
            </Typography>
          </Grid>

          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label="Capital"
              type="number"
              value={formData.capital}
              onChange={(e) => handleInputChange('capital', Number(e.target.value))}
              InputProps={{
                startAdornment: <InputAdornment position="start">₹</InputAdornment>,
              }}
              inputProps={{ min: 100, max: 1000000 }}
            />
          </Grid>

          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label="Leverage"
              type="number"
              value={formData.leverage}
              onChange={(e) => handleInputChange('leverage', Number(e.target.value))}
              InputProps={{
                endAdornment: <InputAdornment position="end">x</InputAdornment>,
              }}
              inputProps={{ min: 1, max: 100 }}
            />
          </Grid>

          <Grid item xs={12} md={4}>
            <TextField
              fullWidth
              label="Risk Per Trade"
              type="number"
              value={formData.riskPerTrade}
              onChange={(e) => handleInputChange('riskPerTrade', Number(e.target.value))}
              InputProps={{
                endAdornment: <InputAdornment position="end">%</InputAdornment>,
              }}
              inputProps={{ min: 0.1, max: 100, step: 0.1 }}
            />
          </Grid>

          <Grid item xs={12} md={4}>
            <TextField
              fullWidth
              label="Stop Loss"
              type="number"
              value={formData.stopLoss}
              onChange={(e) => handleInputChange('stopLoss', Number(e.target.value))}
              InputProps={{
                endAdornment: <InputAdornment position="end">%</InputAdornment>,
              }}
              inputProps={{ min: 0.1, max: 50, step: 0.1 }}
            />
          </Grid>

          <Grid item xs={12} md={4}>
            <TextField
              fullWidth
              label="Take Profit"
              type="number"
              value={formData.takeProfit}
              onChange={(e) => handleInputChange('takeProfit', Number(e.target.value))}
              InputProps={{
                endAdornment: <InputAdornment position="end">%</InputAdornment>,
              }}
              inputProps={{ min: 0.1, max: 100, step: 0.1 }}
            />
          </Grid>

          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Max Positions"
              type="number"
              value={formData.maxPositions}
              onChange={(e) => handleInputChange('maxPositions', Number(e.target.value))}
              inputProps={{ min: 1, max: 10 }}
            />
          </Grid>

          {/* Risk Summary */}
          <Grid item xs={12}>
            <Box sx={{ p: 2, bgcolor: 'background.default', borderRadius: 2 }}>
              <Typography variant="subtitle2" sx={{ mb: 2 }}>
                Risk Summary
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">
                    Max Loss Per Trade
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>
                    ₹{calculateMaxLoss().toFixed(2)}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">
                    Position Size (with leverage)
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>
                    ₹{calculatePositionSize().toFixed(2)}
                  </Typography>
                </Grid>
              </Grid>
              <Box sx={{ mt: 2 }}>
                <Chip 
                  label="Delta Exchange India Testnet" 
                  color="success" 
                  size="small" 
                  variant="outlined"
                />
              </Box>
            </Box>
          </Grid>
        </Grid>
      </DialogContent>

      <DialogActions sx={{ p: 3 }}>
        <Button onClick={onClose} disabled={loading}>
          Cancel
        </Button>
        <Button
          variant="contained"
          onClick={handleSubmit}
          disabled={loading}
          startIcon={<TrendingUpIcon />}
          sx={{ textTransform: 'none' }}
        >
          {loading ? 'Creating...' : 'Create Bot'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}
