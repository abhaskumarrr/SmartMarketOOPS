/**
 * Bot Configuration Form Component
 * Form for creating and editing trading bot configurations
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

  Chip,
  IconButton,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
} from '@mui/icons-material';
import { Bot, BotCreateRequest, BotUpdateRequest, STRATEGY_TYPES, TIMEFRAMES, TRADING_SYMBOLS } from '../../types/bot';
import { botService } from '../../lib/services/botService';

interface BotConfigurationFormProps {
  open: boolean;
  onClose: () => void;
  onSuccess: () => void;
  bot?: Bot | null; // If provided, we're editing; otherwise creating
}

interface ParameterField {
  key: string;
  value: string;
  type: 'string' | 'number' | 'boolean';
}

export const BotConfigurationForm: React.FC<BotConfigurationFormProps> = ({
  open,
  onClose,
  onSuccess,
  bot,
}) => {
  const [formData, setFormData] = useState({
    name: '',
    symbol: 'BTCUSD',
    strategy: 'ML_PREDICTION',
    timeframe: '1h',
  });
  
  const [parameters, setParameters] = useState<ParameterField[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);

  const isEditing = Boolean(bot);

  useEffect(() => {
    if (bot) {
      setFormData({
        name: bot.name,
        symbol: bot.symbol,
        strategy: bot.strategy,
        timeframe: bot.timeframe,
      });
      
      // Convert parameters object to array of fields
      const paramFields: ParameterField[] = Object.entries(bot.parameters || {}).map(([key, value]) => ({
        key,
        value: String(value),
        type: typeof value === 'number' ? 'number' : typeof value === 'boolean' ? 'boolean' : 'string',
      }));
      setParameters(paramFields);
    } else {
      // Reset form for new bot
      setFormData({
        name: '',
        symbol: 'BTCUSD',
        strategy: 'ML_PREDICTION',
        timeframe: '1h',
      });
      setParameters([]);
    }
    setError(null);
    setValidationErrors([]);
  }, [bot, open]);

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({
      ...prev,
      [field]: value,
    }));
  };

  const addParameter = () => {
    setParameters(prev => [
      ...prev,
      { key: '', value: '', type: 'string' },
    ]);
  };

  const updateParameter = (index: number, field: keyof ParameterField, value: string) => {
    setParameters(prev => prev.map((param, i) => 
      i === index ? { ...param, [field]: value } : param
    ));
  };

  const removeParameter = (index: number) => {
    setParameters(prev => prev.filter((_, i) => i !== index));
  };

  const validateForm = (): boolean => {
    const errors: string[] = [];

    if (!formData.name.trim()) {
      errors.push('Bot name is required');
    }

    if (!formData.symbol) {
      errors.push('Trading symbol is required');
    }

    if (!formData.strategy) {
      errors.push('Strategy is required');
    }

    if (!formData.timeframe) {
      errors.push('Timeframe is required');
    }

    // Validate parameters
    const paramKeys = new Set<string>();
    parameters.forEach((param, index) => {
      if (!param.key.trim()) {
        errors.push(`Parameter ${index + 1}: Key is required`);
      } else if (paramKeys.has(param.key)) {
        errors.push(`Parameter ${index + 1}: Duplicate key "${param.key}"`);
      } else {
        paramKeys.add(param.key);
      }

      if (param.type === 'number' && param.value && isNaN(Number(param.value))) {
        errors.push(`Parameter "${param.key}": Invalid number value`);
      }
    });

    setValidationErrors(errors);
    return errors.length === 0;
  };

  const convertParametersToObject = (): Record<string, any> => {
    const result: Record<string, any> = {};
    parameters.forEach(param => {
      if (param.key.trim()) {
        let value: any = param.value;
        if (param.type === 'number') {
          value = Number(param.value);
        } else if (param.type === 'boolean') {
          value = param.value.toLowerCase() === 'true';
        }
        result[param.key] = value;
      }
    });
    return result;
  };

  const handleSubmit = async () => {
    if (!validateForm()) {
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const requestData = {
        ...formData,
        parameters: convertParametersToObject(),
      };

      if (isEditing && bot) {
        const response = await botService.updateBot(bot.id, requestData as BotUpdateRequest);
        if (response.success) {
          onSuccess();
          onClose();
        } else {
          setError(response.message || 'Failed to update bot');
        }
      } else {
        const response = await botService.createBot(requestData as BotCreateRequest);
        if (response.success) {
          onSuccess();
          onClose();
        } else {
          setError(response.message || 'Failed to create bot');
        }
      }
    } catch (error) {
      console.error('Error saving bot:', error);
      setError(error instanceof Error ? error.message : 'Failed to save bot');
    } finally {
      setLoading(false);
    }
  };

  const getStrategyDescription = (strategy: string) => {
    const descriptions: Record<string, string> = {
      ML_PREDICTION: 'Uses machine learning models to predict price movements',
      TECHNICAL_ANALYSIS: 'Based on technical indicators and chart patterns',
      ARBITRAGE: 'Exploits price differences across exchanges',
      GRID_TRADING: 'Places buy and sell orders at regular intervals',
      DCA: 'Dollar Cost Averaging strategy',
      CUSTOM: 'Custom strategy with user-defined parameters',
    };
    return descriptions[strategy] || '';
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        {isEditing ? 'Edit Trading Bot' : 'Create New Trading Bot'}
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

        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
          <TextField
            fullWidth
            label="Bot Name"
            value={formData.name}
            onChange={(e) => handleInputChange('name', e.target.value)}
            placeholder="My Trading Bot"
            required
          />

          <Box sx={{ display: 'flex', gap: 2, flexDirection: { xs: 'column', sm: 'row' } }}>
            <FormControl fullWidth required>
              <InputLabel>Trading Symbol</InputLabel>
              <Select
                value={formData.symbol}
                onChange={(e) => handleInputChange('symbol', e.target.value)}
                label="Trading Symbol"
              >
                {TRADING_SYMBOLS.map((symbol) => (
                  <MenuItem key={symbol} value={symbol}>
                    {symbol}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <FormControl fullWidth required>
              <InputLabel>Timeframe</InputLabel>
              <Select
                value={formData.timeframe}
                onChange={(e) => handleInputChange('timeframe', e.target.value)}
                label="Timeframe"
              >
                {TIMEFRAMES.map((timeframe) => (
                  <MenuItem key={timeframe} value={timeframe}>
                    {timeframe}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Box>

          <FormControl fullWidth required>
            <InputLabel>Strategy</InputLabel>
            <Select
              value={formData.strategy}
              onChange={(e) => handleInputChange('strategy', e.target.value)}
              label="Strategy"
            >
              {STRATEGY_TYPES.map((strategy) => (
                <MenuItem key={strategy} value={strategy}>
                  <Box>
                    <Typography variant="body1">{strategy}</Typography>
                    <Typography variant="caption" color="text.secondary">
                      {getStrategyDescription(strategy)}
                    </Typography>
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <Box>
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="h6">
                  Strategy Parameters
                  {parameters.length > 0 && (
                    <Chip 
                      label={`${parameters.length} parameter${parameters.length !== 1 ? 's' : ''}`} 
                      size="small" 
                      sx={{ ml: 1 }} 
                    />
                  )}
                </Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Typography variant="body2" color="text.secondary">
                      Configure custom parameters for your trading strategy
                    </Typography>
                    <Button
                      startIcon={<AddIcon />}
                      onClick={addParameter}
                      size="small"
                    >
                      Add Parameter
                    </Button>
                  </Box>

                  {parameters.map((param, index) => (
                    <Box key={index} sx={{ display: 'flex', gap: 1, mb: 2, alignItems: 'center' }}>
                      <TextField
                        label="Key"
                        value={param.key}
                        onChange={(e) => updateParameter(index, 'key', e.target.value)}
                        size="small"
                        sx={{ flex: 1 }}
                      />
                      <FormControl size="small" sx={{ minWidth: 100 }}>
                        <InputLabel>Type</InputLabel>
                        <Select
                          value={param.type}
                          onChange={(e) => updateParameter(index, 'type', e.target.value)}
                          label="Type"
                        >
                          <MenuItem value="string">String</MenuItem>
                          <MenuItem value="number">Number</MenuItem>
                          <MenuItem value="boolean">Boolean</MenuItem>
                        </Select>
                      </FormControl>
                      <TextField
                        label="Value"
                        value={param.value}
                        onChange={(e) => updateParameter(index, 'value', e.target.value)}
                        size="small"
                        sx={{ flex: 1 }}
                        placeholder={param.type === 'boolean' ? 'true/false' : ''}
                      />
                      <IconButton
                        onClick={() => removeParameter(index)}
                        color="error"
                        size="small"
                      >
                        <DeleteIcon />
                      </IconButton>
                    </Box>
                  ))}

                  {parameters.length === 0 && (
                    <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 2 }}>
                      No parameters configured. Click "Add Parameter" to add custom strategy parameters.
                    </Typography>
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
          {loading ? 'Saving...' : (isEditing ? 'Update Bot' : 'Create Bot')}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default BotConfigurationForm;
