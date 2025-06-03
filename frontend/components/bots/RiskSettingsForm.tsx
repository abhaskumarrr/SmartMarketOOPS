/**
 * Risk Settings Form Component
 * Form for configuring bot risk management settings
 */

'use client';

import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Box,
  Typography,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  
  Switch,
  FormControlLabel,
  Slider,
  InputAdornment,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Security as SecurityIcon,
  TrendingDown as TrendingDownIcon,
  TrendingUp as TrendingUpIcon,
  Warning as WarningIcon,
} from '@mui/icons-material';
import { Bot, RiskSettings } from '../../types/bot';
import { botService } from '../../lib/services/botService';

interface RiskSettingsFormProps {
  open: boolean;
  onClose: () => void;
  onSuccess: () => void;
  bot: Bot;
}

const defaultRiskSettings: Partial<RiskSettings> = {
  name: '',
  description: '',
  isActive: true,
  positionSizingMethod: 'FIXED_FRACTIONAL',
  riskPercentage: 2.0,
  maxPositionSize: 1000,
  stopLossType: 'PERCENTAGE',
  stopLossValue: 2.0,
  takeProfitType: 'PERCENTAGE',
  takeProfitValue: 4.0,
  maxRiskPerTrade: 2.0,
  maxRiskPerSymbol: 10.0,
  maxRiskPerDirection: 15.0,
  maxTotalRisk: 20.0,
  maxDrawdown: 10.0,
  maxPositions: 5,
  maxDailyLoss: 5.0,
  cooldownPeriod: 300,
  volatilityLookback: 20,
  circuitBreakerEnabled: true,
  maxDailyLossBreaker: 10.0,
  maxDrawdownBreaker: 15.0,
  volatilityMultiplier: 2.0,
  consecutiveLossesBreaker: 5,
  tradingPause: 3600,
  marketWideEnabled: true,
  enableManualOverride: true,
};

export const RiskSettingsForm: React.FC<RiskSettingsFormProps> = ({
  open,
  onClose,
  onSuccess,
  bot,
}) => {
  const [formData, setFormData] = useState<Partial<RiskSettings>>(defaultRiskSettings);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);

  useEffect(() => {
    if (open && bot) {
      fetchRiskSettings();
    }
  }, [open, bot]);

  const fetchRiskSettings = async () => {
    try {
      const response = await botService.getBotRiskSettings(bot.id);
      if (response.success) {
        setFormData(response.data);
      } else {
        // Use default settings with bot name
        setFormData({
          ...defaultRiskSettings,
          name: `${bot.name} Risk Settings`,
          description: `Risk management settings for ${bot.name}`,
        });
      }
    } catch (error) {
      console.error('Error fetching risk settings:', error);
      setFormData({
        ...defaultRiskSettings,
        name: `${bot.name} Risk Settings`,
        description: `Risk management settings for ${bot.name}`,
      });
    }
  };

  const handleInputChange = (field: keyof RiskSettings, value: any) => {
    setFormData(prev => ({
      ...prev,
      [field]: value,
    }));
  };

  const validateForm = (): boolean => {
    const errors: string[] = [];

    if (!formData.name?.trim()) {
      errors.push('Risk settings name is required');
    }

    if ((formData.riskPercentage || 0) <= 0 || (formData.riskPercentage || 0) > 100) {
      errors.push('Risk percentage must be between 0 and 100');
    }

    if ((formData.maxPositionSize || 0) <= 0) {
      errors.push('Maximum position size must be greater than 0');
    }

    if ((formData.stopLossValue || 0) <= 0) {
      errors.push('Stop loss value must be greater than 0');
    }

    if ((formData.takeProfitValue || 0) <= 0) {
      errors.push('Take profit value must be greater than 0');
    }

    setValidationErrors(errors);
    return errors.length === 0;
  };

  const handleSubmit = async () => {
    if (!validateForm()) {
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const response = await botService.updateBotRiskSettings(bot.id, formData);
      if (response.success) {
        onSuccess();
        onClose();
      } else {
        setError(response.message || 'Failed to update risk settings');
      }
    } catch (error) {
      console.error('Error saving risk settings:', error);
      setError(error instanceof Error ? error.message : 'Failed to save risk settings');
    } finally {
      setLoading(false);
    }
  };

  const getRiskLevel = (percentage: number): { level: string; color: string } => {
    if (percentage <= 1) return { level: 'Conservative', color: 'success.main' };
    if (percentage <= 3) return { level: 'Moderate', color: 'warning.main' };
    return { level: 'Aggressive', color: 'error.main' };
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <SecurityIcon sx={{ mr: 1 }} />
          Risk Management Settings - {bot.name}
        </Box>
      </DialogTitle>
      
      <DialogContent>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        
        {validationErrors.length > 0 && (
          <Alert severity="warning" sx={{ mb: 2 }}>
            <Typography variant="body2" gutterBottom>
              Please fix the following errors:
            </Typography>
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {validationErrors.map((error, index) => (
                <li key={index}>{error}</li>
              ))}
            </ul>
          </Alert>
        )}

        <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", sm: "repeat(2, 1fr)", md: "repeat(3, 1fr)" }, gap: 3 }}>
          <Box>
            <TextField
              fullWidth
              label="Settings Name"
              value={formData.name || ''}
              onChange={(e) => handleInputChange('name', e.target.value)}
              required
            />
          </Box>

          <Box>
            <TextField
              fullWidth
              label="Description"
              value={formData.description || ''}
              onChange={(e) => handleInputChange('description', e.target.value)}
              multiline
              rows={2}
            />
          </Box>

          {/* Position Sizing */}
          <Box>
            <Accordion defaultExpanded>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="h6">Position Sizing</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", sm: "repeat(2, 1fr)", md: "repeat(3, 1fr)" }, gap: 2 }}>
                  <Box>
                    <FormControl fullWidth>
                      <InputLabel>Position Sizing Method</InputLabel>
                      <Select
                        value={formData.positionSizingMethod || 'FIXED_FRACTIONAL'}
                        onChange={(e) => handleInputChange('positionSizingMethod', e.target.value)}
                        label="Position Sizing Method"
                      >
                        <MenuItem value="FIXED_FRACTIONAL">Fixed Fractional</MenuItem>
                        <MenuItem value="KELLY_CRITERION">Kelly Criterion</MenuItem>
                        <MenuItem value="FIXED_AMOUNT">Fixed Amount</MenuItem>
                        <MenuItem value="VOLATILITY_BASED">Volatility Based</MenuItem>
                      </Select>
                    </FormControl>
                  </Box>
                  
                  <Box>
                    <TextField
                      fullWidth
                      label="Risk Per Trade"
                      type="number"
                      value={formData.riskPercentage || 2}
                      onChange={(e) => handleInputChange('riskPercentage', parseFloat(e.target.value))}
                      InputProps={{
                        endAdornment: <InputAdornment position="end">%</InputAdornment>,
                      }}
                      helperText={`Risk Level: ${getRiskLevel(formData.riskPercentage || 2).level}`}
                    />
                  </Box>

                  <Box>
                    <Typography gutterBottom>Risk Per Trade: {formData.riskPercentage || 2}%</Typography>
                    <Slider
                      value={formData.riskPercentage || 2}
                      onChange={(_, value) => handleInputChange('riskPercentage', value)}
                      min={0.1}
                      max={10}
                      step={0.1}
                      marks={[
                        { value: 1, label: '1%' },
                        { value: 2, label: '2%' },
                        { value: 5, label: '5%' },
                        { value: 10, label: '10%' },
                      ]}
                      sx={{
                        color: getRiskLevel(formData.riskPercentage || 2).color,
                      }}
                    />
                  </Box>

                  <Box>
                    <TextField
                      fullWidth
                      label="Maximum Position Size"
                      type="number"
                      value={formData.maxPositionSize || 1000}
                      onChange={(e) => handleInputChange('maxPositionSize', parseFloat(e.target.value))}
                      InputProps={{
                        endAdornment: <InputAdornment position="end">USD</InputAdornment>,
                      }}
                    />
                  </Box>
                </Box>
              </AccordionDetails>
            </Accordion>
          </Box>

          {/* Stop Loss & Take Profit */}
          <Box>
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="h6">Stop Loss & Take Profit</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", sm: "repeat(2, 1fr)", md: "repeat(3, 1fr)" }, gap: 2 }}>
                  <Box>
                    <FormControl fullWidth>
                      <InputLabel>Stop Loss Type</InputLabel>
                      <Select
                        value={formData.stopLossType || 'PERCENTAGE'}
                        onChange={(e) => handleInputChange('stopLossType', e.target.value)}
                        label="Stop Loss Type"
                      >
                        <MenuItem value="PERCENTAGE">Percentage</MenuItem>
                        <MenuItem value="FIXED">Fixed Price</MenuItem>
                        <MenuItem value="ATR">ATR Based</MenuItem>
                        <MenuItem value="TRAILING">Trailing Stop</MenuItem>
                      </Select>
                    </FormControl>
                  </Box>
                  
                  <Box>
                    <TextField
                      fullWidth
                      label="Stop Loss Value"
                      type="number"
                      value={formData.stopLossValue || 2}
                      onChange={(e) => handleInputChange('stopLossValue', parseFloat(e.target.value))}
                      InputProps={{
                        endAdornment: <InputAdornment position="end">
                          {formData.stopLossType === 'PERCENTAGE' ? '%' : 'USD'}
                        </InputAdornment>,
                        startAdornment: <TrendingDownIcon color="error" />,
                      }}
                    />
                  </Box>

                  <Box>
                    <FormControl fullWidth>
                      <InputLabel>Take Profit Type</InputLabel>
                      <Select
                        value={formData.takeProfitType || 'PERCENTAGE'}
                        onChange={(e) => handleInputChange('takeProfitType', e.target.value)}
                        label="Take Profit Type"
                      >
                        <MenuItem value="PERCENTAGE">Percentage</MenuItem>
                        <MenuItem value="FIXED">Fixed Price</MenuItem>
                        <MenuItem value="RISK_REWARD">Risk/Reward Ratio</MenuItem>
                        <MenuItem value="TRAILING">Trailing Profit</MenuItem>
                      </Select>
                    </FormControl>
                  </Box>
                  
                  <Box>
                    <TextField
                      fullWidth
                      label="Take Profit Value"
                      type="number"
                      value={formData.takeProfitValue || 4}
                      onChange={(e) => handleInputChange('takeProfitValue', parseFloat(e.target.value))}
                      InputProps={{
                        endAdornment: <InputAdornment position="end">
                          {formData.takeProfitType === 'PERCENTAGE' ? '%' : 'USD'}
                        </InputAdornment>,
                        startAdornment: <TrendingUpIcon color="success" />,
                      }}
                    />
                  </Box>
                </Box>
              </AccordionDetails>
            </Accordion>
          </Box>

          {/* Risk Limits */}
          <Box>
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="h6">Risk Limits</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", sm: "repeat(2, 1fr)", md: "repeat(3, 1fr)" }, gap: 2 }}>
                  <Box>
                    <TextField
                      fullWidth
                      label="Max Risk Per Symbol"
                      type="number"
                      value={formData.maxRiskPerSymbol || 10}
                      onChange={(e) => handleInputChange('maxRiskPerSymbol', parseFloat(e.target.value))}
                      InputProps={{
                        endAdornment: <InputAdornment position="end">%</InputAdornment>,
                      }}
                    />
                  </Box>
                  
                  <Box>
                    <TextField
                      fullWidth
                      label="Max Total Risk"
                      type="number"
                      value={formData.maxTotalRisk || 20}
                      onChange={(e) => handleInputChange('maxTotalRisk', parseFloat(e.target.value))}
                      InputProps={{
                        endAdornment: <InputAdornment position="end">%</InputAdornment>,
                      }}
                    />
                  </Box>

                  <Box>
                    <TextField
                      fullWidth
                      label="Max Drawdown"
                      type="number"
                      value={formData.maxDrawdown || 10}
                      onChange={(e) => handleInputChange('maxDrawdown', parseFloat(e.target.value))}
                      InputProps={{
                        endAdornment: <InputAdornment position="end">%</InputAdornment>,
                      }}
                    />
                  </Box>

                  <Box>
                    <TextField
                      fullWidth
                      label="Max Daily Loss"
                      type="number"
                      value={formData.maxDailyLoss || 5}
                      onChange={(e) => handleInputChange('maxDailyLoss', parseFloat(e.target.value))}
                      InputProps={{
                        endAdornment: <InputAdornment position="end">%</InputAdornment>,
                      }}
                    />
                  </Box>

                  <Box>
                    <TextField
                      fullWidth
                      label="Max Positions"
                      type="number"
                      value={formData.maxPositions || 5}
                      onChange={(e) => handleInputChange('maxPositions', parseInt(e.target.value))}
                    />
                  </Box>

                  <Box>
                    <TextField
                      fullWidth
                      label="Cooldown Period"
                      type="number"
                      value={formData.cooldownPeriod || 300}
                      onChange={(e) => handleInputChange('cooldownPeriod', parseInt(e.target.value))}
                      InputProps={{
                        endAdornment: <InputAdornment position="end">seconds</InputAdornment>,
                      }}
                    />
                  </Box>
                </Box>
              </AccordionDetails>
            </Accordion>
          </Box>

          {/* Circuit Breaker */}
          <Box>
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <WarningIcon sx={{ mr: 1 }} />
                  <Typography variant="h6">Circuit Breaker</Typography>
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", sm: "repeat(2, 1fr)", md: "repeat(3, 1fr)" }, gap: 2 }}>
                  <Box>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={formData.circuitBreakerEnabled || false}
                          onChange={(e) => handleInputChange('circuitBreakerEnabled', e.target.checked)}
                        />
                      }
                      label="Enable Circuit Breaker"
                    />
                  </Box>

                  {formData.circuitBreakerEnabled && (
                    <>
                      <Box>
                        <TextField
                          fullWidth
                          label="Max Daily Loss Breaker"
                          type="number"
                          value={formData.maxDailyLossBreaker || 10}
                          onChange={(e) => handleInputChange('maxDailyLossBreaker', parseFloat(e.target.value))}
                          InputProps={{
                            endAdornment: <InputAdornment position="end">%</InputAdornment>,
                          }}
                        />
                      </Box>

                      <Box>
                        <TextField
                          fullWidth
                          label="Max Drawdown Breaker"
                          type="number"
                          value={formData.maxDrawdownBreaker || 15}
                          onChange={(e) => handleInputChange('maxDrawdownBreaker', parseFloat(e.target.value))}
                          InputProps={{
                            endAdornment: <InputAdornment position="end">%</InputAdornment>,
                          }}
                        />
                      </Box>

                      <Box>
                        <TextField
                          fullWidth
                          label="Consecutive Losses Breaker"
                          type="number"
                          value={formData.consecutiveLossesBreaker || 5}
                          onChange={(e) => handleInputChange('consecutiveLossesBreaker', parseInt(e.target.value))}
                        />
                      </Box>

                      <Box>
                        <TextField
                          fullWidth
                          label="Trading Pause Duration"
                          type="number"
                          value={formData.tradingPause || 3600}
                          onChange={(e) => handleInputChange('tradingPause', parseInt(e.target.value))}
                          InputProps={{
                            endAdornment: <InputAdornment position="end">seconds</InputAdornment>,
                          }}
                        />
                      </Box>

                      <Box>
                        <FormControlLabel
                          control={
                            <Switch
                              checked={formData.enableManualOverride || false}
                              onChange={(e) => handleInputChange('enableManualOverride', e.target.checked)}
                            />
                          }
                          label="Enable Manual Override"
                        />
                      </Box>
                    </>
                  )}
                </Box>
              </AccordionDetails>
            </Accordion>
          </Box>
        </Box>
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose} disabled={loading}>
          Cancel
        </Button>
        <Button
          onClick={handleSubmit}
          variant="contained"
          disabled={loading}
        >
          {loading ? 'Saving...' : 'Save Risk Settings'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default RiskSettingsForm;
