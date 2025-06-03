/**
 * Strategy Selection Step
 * Second step of the bot configuration wizard
 */

'use client';

import React, { useState } from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  CardActions,
  Button,
  Chip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Alert,
  IconButton,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  TrendingUp as TrendingUpIcon,
  Psychology as PsychologyIcon,
  GridOn as GridOnIcon,
  SwapHoriz as SwapHorizIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
} from '@mui/icons-material';
import { useFormContext, Controller } from 'react-hook-form';
import { STRATEGY_TYPES } from '../../../types/bot';

const strategyTemplates = {
  ML_PREDICTION: {
    name: 'ML Prediction',
    description: 'Uses advanced machine learning models for price prediction',
    icon: <PsychologyIcon />,
    parameters: {
      confidence_threshold: 0.7,
      prediction_horizon: 24,
      model_ensemble: true,
      feature_importance: 0.8,
    },
    riskProfile: 'moderate',
    expectedReturn: '15-25%',
    winRate: '75-85%',
    complexity: 'Advanced',
    color: 'primary',
  },
  TECHNICAL_ANALYSIS: {
    name: 'Technical Analysis',
    description: 'Based on technical indicators and chart patterns',
    icon: <TrendingUpIcon />,
    parameters: {
      rsi_period: 14,
      ma_fast: 12,
      ma_slow: 26,
      bollinger_period: 20,
      volume_threshold: 1.5,
    },
    riskProfile: 'conservative',
    expectedReturn: '8-15%',
    winRate: '65-75%',
    complexity: 'Intermediate',
    color: 'success',
  },
  GRID_TRADING: {
    name: 'Grid Trading',
    description: 'Places buy and sell orders at regular intervals',
    icon: <GridOnIcon />,
    parameters: {
      grid_size: 0.5,
      grid_levels: 10,
      profit_per_grid: 0.2,
      max_grids: 20,
    },
    riskProfile: 'moderate',
    expectedReturn: '10-20%',
    winRate: '80-90%',
    complexity: 'Beginner',
    color: 'info',
  },
  ARBITRAGE: {
    name: 'Arbitrage',
    description: 'Exploits price differences across exchanges',
    icon: <SwapHorizIcon />,
    parameters: {
      min_spread: 0.1,
      max_exposure: 0.3,
      execution_speed: 'fast',
      slippage_tolerance: 0.05,
    },
    riskProfile: 'low',
    expectedReturn: '5-12%',
    winRate: '90-95%',
    complexity: 'Expert',
    color: 'warning',
  },
  DCA: {
    name: 'Dollar Cost Averaging',
    description: 'Systematic investment strategy with regular purchases',
    icon: <TrendingUpIcon />,
    parameters: {
      investment_amount: 100,
      frequency: 'daily',
      price_threshold: 0.05,
      max_deviation: 0.1,
    },
    riskProfile: 'conservative',
    expectedReturn: '5-15%',
    winRate: '70-80%',
    complexity: 'Beginner',
    color: 'secondary',
  },
  CUSTOM: {
    name: 'Custom Strategy',
    description: 'Define your own custom trading strategy',
    icon: <PsychologyIcon />,
    parameters: {},
    riskProfile: 'variable',
    expectedReturn: 'Variable',
    winRate: 'Variable',
    complexity: 'Expert',
    color: 'default',
  },
};

interface ParameterField {
  key: string;
  value: string;
  type: 'string' | 'number' | 'boolean';
}

export const StrategySelectionStep: React.FC = () => {
  const { control, watch, setValue, formState: { errors } } = useFormContext();
  const [customParameters, setCustomParameters] = useState<ParameterField[]>([]);
  const selectedStrategy = watch('strategy');
  const currentParameters = watch('parameters');

  const handleStrategySelect = (strategy: string) => {
    setValue('strategy', strategy);
    const template = strategyTemplates[strategy as keyof typeof strategyTemplates];
    if (template && template.parameters) {
      setValue('parameters', template.parameters);
    }
  };

  const addCustomParameter = () => {
    setCustomParameters(prev => [
      ...prev,
      { key: '', value: '', type: 'string' },
    ]);
  };

  const updateCustomParameter = (index: number, field: keyof ParameterField, value: string) => {
    setCustomParameters(prev => prev.map((param, i) => 
      i === index ? { ...param, [field]: value } : param
    ));
    
    // Update form parameters
    const updatedParams = { ...currentParameters };
    const param = customParameters[index];
    if (param && param.key) {
      let convertedValue: any = value;
      if (field === 'value') {
        if (param.type === 'number') {
          convertedValue = Number(value);
        } else if (param.type === 'boolean') {
          convertedValue = value.toLowerCase() === 'true';
        }
      }
      updatedParams[param.key] = convertedValue;
      setValue('parameters', updatedParams);
    }
  };

  const removeCustomParameter = (index: number) => {
    const param = customParameters[index];
    if (param && param.key) {
      const updatedParams = { ...currentParameters };
      delete updatedParams[param.key];
      setValue('parameters', updatedParams);
    }
    setCustomParameters(prev => prev.filter((_, i) => i !== index));
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Strategy Selection
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Choose a trading strategy that matches your risk tolerance and trading goals
      </Typography>

      <Grid container spacing={2} sx={{ mb: 3 }}>
        {STRATEGY_TYPES.map((strategy) => {
          const template = strategyTemplates[strategy as keyof typeof strategyTemplates];
          const isSelected = selectedStrategy === strategy;
          
          return (
            <Grid item xs={12} md={6} key={strategy}>
              <Card 
                variant={isSelected ? "elevation" : "outlined"}
                sx={{ 
                  cursor: 'pointer',
                  border: isSelected ? 2 : 1,
                  borderColor: isSelected ? `${template.color}.main` : 'divider',
                  '&:hover': { 
                    boxShadow: 2,
                    borderColor: `${template.color}.main`,
                  }
                }}
                onClick={() => handleStrategySelect(strategy)}
              >
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Box sx={{ mr: 2, color: `${template.color}.main` }}>
                      {template.icon}
                    </Box>
                    <Box sx={{ flex: 1 }}>
                      <Typography variant="h6" component="div">
                        {template.name}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {template.description}
                      </Typography>
                    </Box>
                  </Box>
                  
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                    <Chip 
                      label={`Risk: ${template.riskProfile}`} 
                      size="small" 
                      color={template.riskProfile === 'low' ? 'success' : template.riskProfile === 'conservative' ? 'info' : 'warning'}
                    />
                    <Chip 
                      label={`Return: ${template.expectedReturn}`} 
                      size="small" 
                      variant="outlined"
                    />
                    <Chip 
                      label={`Win Rate: ${template.winRate}`} 
                      size="small" 
                      variant="outlined"
                    />
                  </Box>
                  
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Chip 
                      label={template.complexity} 
                      size="small"
                      color={
                        template.complexity === 'Beginner' ? 'success' :
                        template.complexity === 'Intermediate' ? 'info' :
                        template.complexity === 'Advanced' ? 'warning' : 'error'
                      }
                    />
                    {isSelected && (
                      <Chip label="Selected" color="primary" size="small" />
                    )}
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          );
        })}
      </Grid>

      {selectedStrategy && (
        <Accordion defaultExpanded>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="h6">
              Strategy Parameters
              {Object.keys(currentParameters || {}).length > 0 && (
                <Chip 
                  label={`${Object.keys(currentParameters).length} parameter${Object.keys(currentParameters).length !== 1 ? 's' : ''}`} 
                  size="small" 
                  sx={{ ml: 1 }} 
                />
              )}
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Box>
              {selectedStrategy !== 'CUSTOM' ? (
                <Grid container spacing={2}>
                  {Object.entries(currentParameters || {}).map(([key, value]) => (
                    <Grid item xs={12} sm={6} key={key}>
                      <TextField
                        fullWidth
                        label={key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                        value={value}
                        onChange={(e) => {
                          const updatedParams = { ...currentParameters };
                          updatedParams[key] = e.target.value;
                          setValue('parameters', updatedParams);
                        }}
                        size="small"
                      />
                    </Grid>
                  ))}
                </Grid>
              ) : (
                <Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Typography variant="body2" color="text.secondary">
                      Define custom parameters for your strategy
                    </Typography>
                    <Button
                      startIcon={<AddIcon />}
                      onClick={addCustomParameter}
                      size="small"
                    >
                      Add Parameter
                    </Button>
                  </Box>

                  {customParameters.map((param, index) => (
                    <Box key={index} sx={{ display: 'flex', gap: 1, mb: 2, alignItems: 'center' }}>
                      <TextField
                        label="Key"
                        value={param.key}
                        onChange={(e) => updateCustomParameter(index, 'key', e.target.value)}
                        size="small"
                        sx={{ flex: 1 }}
                      />
                      <FormControl size="small" sx={{ minWidth: 100 }}>
                        <InputLabel>Type</InputLabel>
                        <Select
                          value={param.type}
                          onChange={(e) => updateCustomParameter(index, 'type', e.target.value)}
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
                        onChange={(e) => updateCustomParameter(index, 'value', e.target.value)}
                        size="small"
                        sx={{ flex: 1 }}
                        placeholder={param.type === 'boolean' ? 'true/false' : ''}
                      />
                      <IconButton
                        onClick={() => removeCustomParameter(index)}
                        color="error"
                        size="small"
                      >
                        <DeleteIcon />
                      </IconButton>
                    </Box>
                  ))}

                  {customParameters.length === 0 && (
                    <Alert severity="info">
                      No custom parameters defined. Click "Add Parameter" to add strategy-specific parameters.
                    </Alert>
                  )}
                </Box>
              )}
            </Box>
          </AccordionDetails>
        </Accordion>
      )}
    </Box>
  );
};

export default StrategySelectionStep;
