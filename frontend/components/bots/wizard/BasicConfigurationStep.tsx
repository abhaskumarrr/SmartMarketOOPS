/**
 * Basic Configuration Step
 * First step of the bot configuration wizard
 */

'use client';

import React from 'react';
import {
  Box,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Typography,
  Grid,
  Card,
  CardContent,
  Chip,
  Alert,
} from '@mui/material';
import { useFormContext, Controller } from 'react-hook-form';
import { TRADING_SYMBOLS, TIMEFRAMES } from '../../../types/bot';

const symbolInfo = {
  BTCUSD: { name: 'Bitcoin', volatility: 'High', liquidity: 'Excellent' },
  ETHUSD: { name: 'Ethereum', volatility: 'High', liquidity: 'Excellent' },
  ADAUSD: { name: 'Cardano', volatility: 'Very High', liquidity: 'Good' },
  SOLUSD: { name: 'Solana', volatility: 'Very High', liquidity: 'Good' },
  DOTUSD: { name: 'Polkadot', volatility: 'Very High', liquidity: 'Moderate' },
  LINKUSD: { name: 'Chainlink', volatility: 'Very High', liquidity: 'Good' },
};

const timeframeInfo = {
  '1m': { name: '1 Minute', frequency: 'Very High', signals: '1000+/day' },
  '5m': { name: '5 Minutes', frequency: 'High', signals: '200+/day' },
  '15m': { name: '15 Minutes', frequency: 'Moderate', signals: '50+/day' },
  '30m': { name: '30 Minutes', frequency: 'Moderate', signals: '25+/day' },
  '1h': { name: '1 Hour', frequency: 'Low', signals: '10+/day' },
  '4h': { name: '4 Hours', frequency: 'Very Low', signals: '3+/day' },
  '1d': { name: '1 Day', frequency: 'Ultra Low', signals: '1/day' },
};

export const BasicConfigurationStep: React.FC = () => {
  const { control, formState: { errors }, watch } = useFormContext();
  const selectedSymbol = watch('symbol');
  const selectedTimeframe = watch('timeframe');

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Basic Bot Configuration
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Configure the fundamental settings for your trading bot
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Controller
            name="name"
            control={control}
            render={({ field }) => (
              <TextField
                {...field}
                fullWidth
                label="Bot Name"
                placeholder="My Trading Bot"
                error={!!errors.name}
                helperText={errors.name?.message as string}
                sx={{ mb: 2 }}
              />
            )}
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <Controller
            name="symbol"
            control={control}
            render={({ field }) => (
              <FormControl fullWidth error={!!errors.symbol}>
                <InputLabel>Trading Symbol</InputLabel>
                <Select {...field} label="Trading Symbol">
                  {TRADING_SYMBOLS.map((symbol) => (
                    <MenuItem key={symbol} value={symbol}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography>{symbol}</Typography>
                        <Chip 
                          label={symbolInfo[symbol as keyof typeof symbolInfo]?.name} 
                          size="small" 
                          variant="outlined" 
                        />
                      </Box>
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            )}
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <Controller
            name="timeframe"
            control={control}
            render={({ field }) => (
              <FormControl fullWidth error={!!errors.timeframe}>
                <InputLabel>Timeframe</InputLabel>
                <Select {...field} label="Timeframe">
                  {TIMEFRAMES.map((timeframe) => (
                    <MenuItem key={timeframe} value={timeframe}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography>{timeframe}</Typography>
                        <Chip 
                          label={timeframeInfo[timeframe as keyof typeof timeframeInfo]?.name} 
                          size="small" 
                          variant="outlined" 
                        />
                      </Box>
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            )}
          />
        </Grid>

        {selectedSymbol && (
          <Grid item xs={12} md={6}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="subtitle2" gutterBottom>
                  Symbol Information
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2">Name:</Typography>
                    <Typography variant="body2" fontWeight="bold">
                      {symbolInfo[selectedSymbol as keyof typeof symbolInfo]?.name}
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2">Volatility:</Typography>
                    <Chip 
                      label={symbolInfo[selectedSymbol as keyof typeof symbolInfo]?.volatility}
                      size="small"
                      color={
                        symbolInfo[selectedSymbol as keyof typeof symbolInfo]?.volatility === 'High' ? 'warning' :
                        symbolInfo[selectedSymbol as keyof typeof symbolInfo]?.volatility === 'Very High' ? 'error' : 'success'
                      }
                    />
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2">Liquidity:</Typography>
                    <Chip 
                      label={symbolInfo[selectedSymbol as keyof typeof symbolInfo]?.liquidity}
                      size="small"
                      color={
                        symbolInfo[selectedSymbol as keyof typeof symbolInfo]?.liquidity === 'Excellent' ? 'success' :
                        symbolInfo[selectedSymbol as keyof typeof symbolInfo]?.liquidity === 'Good' ? 'primary' : 'default'
                      }
                    />
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        )}

        {selectedTimeframe && (
          <Grid item xs={12} md={6}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="subtitle2" gutterBottom>
                  Timeframe Information
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2">Name:</Typography>
                    <Typography variant="body2" fontWeight="bold">
                      {timeframeInfo[selectedTimeframe as keyof typeof timeframeInfo]?.name}
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2">Frequency:</Typography>
                    <Chip 
                      label={timeframeInfo[selectedTimeframe as keyof typeof timeframeInfo]?.frequency}
                      size="small"
                      color={
                        timeframeInfo[selectedTimeframe as keyof typeof timeframeInfo]?.frequency.includes('High') ? 'error' :
                        timeframeInfo[selectedTimeframe as keyof typeof timeframeInfo]?.frequency === 'Moderate' ? 'warning' : 'success'
                      }
                    />
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2">Expected Signals:</Typography>
                    <Typography variant="body2" fontWeight="bold">
                      {timeframeInfo[selectedTimeframe as keyof typeof timeframeInfo]?.signals}
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        )}

        <Grid item xs={12}>
          <Alert severity="info">
            <Typography variant="body2">
              <strong>Tip:</strong> For beginners, we recommend starting with BTCUSD or ETHUSD on 1h or 4h timeframes 
              for more stable and manageable trading signals.
            </Typography>
          </Alert>
        </Grid>
      </Grid>
    </Box>
  );
};

export default BasicConfigurationStep;
