/**
 * Enhanced Bot Configuration Wizard
 * Multi-step wizard for comprehensive bot configuration with risk management
 */

'use client';

import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Stepper,
  Step,
  StepLabel,
  Box,
  Typography,
  Alert,
  Card,
  CardContent,
  Grid,
  Chip,
  LinearProgress,
} from '@mui/material';
import { useForm, FormProvider } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';
import { Bot, BotCreateRequest, STRATEGY_TYPES, TIMEFRAMES, TRADING_SYMBOLS } from '../../types/bot';
import { botService } from '../../lib/services/botService';

// Validation schema using Zod
const botConfigSchema = z.object({
  name: z.string().min(1, 'Bot name is required').max(50, 'Name too long'),
  symbol: z.enum(TRADING_SYMBOLS as [string, ...string[]]),
  strategy: z.enum(STRATEGY_TYPES as [string, ...string[]]),
  timeframe: z.enum(TIMEFRAMES as [string, ...string[]]),
  parameters: z.record(z.any()).optional(),
  riskSettings: z.object({
    riskPercentage: z.number().min(0.1).max(10),
    maxPositionSize: z.number().min(100).max(100000),
    stopLossType: z.enum(['percentage', 'atr', 'fixed']),
    stopLossValue: z.number().min(0.1).max(20),
    takeProfitType: z.enum(['percentage', 'ratio', 'fixed']),
    takeProfitValue: z.number().min(0.1).max(50),
    maxDailyLoss: z.number().min(1).max(50),
  }),
});

type BotConfigFormData = z.infer<typeof botConfigSchema>;

interface BotConfigurationWizardProps {
  open: boolean;
  onClose: () => void;
  onSuccess: () => void;
  bot?: Bot | null;
}

const steps = [
  'Basic Configuration',
  'Strategy Selection',
  'Risk Management',
  'Review & Create'
];

const strategyTemplates = {
  ML_PREDICTION: {
    name: 'ML Prediction',
    description: 'Uses advanced machine learning models for price prediction',
    parameters: {
      confidence_threshold: 0.7,
      prediction_horizon: 24,
      model_ensemble: true,
      feature_importance: 0.8,
    },
    riskProfile: 'moderate',
    expectedReturn: '15-25%',
    winRate: '75-85%',
  },
  TECHNICAL_ANALYSIS: {
    name: 'Technical Analysis',
    description: 'Based on technical indicators and chart patterns',
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
  },
  GRID_TRADING: {
    name: 'Grid Trading',
    description: 'Places buy and sell orders at regular intervals',
    parameters: {
      grid_size: 0.5,
      grid_levels: 10,
      profit_per_grid: 0.2,
      max_grids: 20,
    },
    riskProfile: 'moderate',
    expectedReturn: '10-20%',
    winRate: '80-90%',
  },
  ARBITRAGE: {
    name: 'Arbitrage',
    description: 'Exploits price differences across exchanges',
    parameters: {
      min_spread: 0.1,
      max_exposure: 0.3,
      execution_speed: 'fast',
      slippage_tolerance: 0.05,
    },
    riskProfile: 'low',
    expectedReturn: '5-12%',
    winRate: '90-95%',
  },
};

export const BotConfigurationWizard: React.FC<BotConfigurationWizardProps> = ({
  open,
  onClose,
  onSuccess,
  bot,
}) => {
  const [activeStep, setActiveStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const methods = useForm<BotConfigFormData>({
    resolver: zodResolver(botConfigSchema),
    defaultValues: {
      name: '',
      symbol: 'BTCUSD',
      strategy: 'ML_PREDICTION',
      timeframe: '1h',
      parameters: {},
      riskSettings: {
        riskPercentage: 2,
        maxPositionSize: 1000,
        stopLossType: 'percentage',
        stopLossValue: 2,
        takeProfitType: 'ratio',
        takeProfitValue: 2,
        maxDailyLoss: 5,
      },
    },
  });

  const { handleSubmit, watch, setValue, formState: { errors } } = methods;
  const watchedStrategy = watch('strategy');

  useEffect(() => {
    if (bot) {
      // Populate form with existing bot data
      setValue('name', bot.name);
      setValue('symbol', bot.symbol as any);
      setValue('strategy', bot.strategy as any);
      setValue('timeframe', bot.timeframe as any);
      setValue('parameters', bot.parameters || {});
    }
  }, [bot, setValue]);

  useEffect(() => {
    // Auto-populate strategy parameters when strategy changes
    if (watchedStrategy && strategyTemplates[watchedStrategy as keyof typeof strategyTemplates]) {
      const template = strategyTemplates[watchedStrategy as keyof typeof strategyTemplates];
      setValue('parameters', template.parameters);
    }
  }, [watchedStrategy, setValue]);

  const handleNext = () => {
    setActiveStep((prevActiveStep) => prevActiveStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
  };

  const onSubmit = async (data: BotConfigFormData) => {
    try {
      setLoading(true);
      setError(null);

      const requestData: BotCreateRequest = {
        name: data.name,
        symbol: data.symbol,
        strategy: data.strategy,
        timeframe: data.timeframe,
        parameters: data.parameters,
      };

      if (bot) {
        const response = await botService.updateBot(bot.id, requestData);
        if (response.success) {
          onSuccess();
          onClose();
        } else {
          setError(response.message || 'Failed to update bot');
        }
      } else {
        const response = await botService.createBot(requestData);
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

  const renderStepContent = (step: number) => {
    switch (step) {
      case 0:
        return <BasicConfigurationStep />;
      case 1:
        return <StrategySelectionStep />;
      case 2:
        return <RiskManagementStep />;
      case 3:
        return <ReviewStep />;
      default:
        return null;
    }
  };

  const isStepValid = (step: number): boolean => {
    const values = methods.getValues();
    switch (step) {
      case 0:
        return !!values.name && !!values.symbol && !!values.timeframe;
      case 1:
        return !!values.strategy;
      case 2:
        return !!values.riskSettings;
      case 3:
        return true;
      default:
        return false;
    }
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Typography variant="h5">
            {bot ? 'Edit Trading Bot' : 'Create New Trading Bot'}
          </Typography>
          <Chip 
            label={`Step ${activeStep + 1} of ${steps.length}`} 
            color="primary" 
            variant="outlined" 
          />
        </Box>
      </DialogTitle>

      <DialogContent>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        <Box sx={{ mb: 3 }}>
          <Stepper activeStep={activeStep} alternativeLabel>
            {steps.map((label, index) => (
              <Step key={label} completed={index < activeStep || isStepValid(index)}>
                <StepLabel>{label}</StepLabel>
              </Step>
            ))}
          </Stepper>
        </Box>

        <Box sx={{ minHeight: 400 }}>
          <FormProvider {...methods}>
            {renderStepContent(activeStep)}
          </FormProvider>
        </Box>

        {loading && (
          <Box sx={{ mt: 2 }}>
            <LinearProgress />
            <Typography variant="body2" sx={{ mt: 1, textAlign: 'center' }}>
              {bot ? 'Updating bot...' : 'Creating bot...'}
            </Typography>
          </Box>
        )}
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose} disabled={loading}>
          Cancel
        </Button>
        <Box sx={{ flex: '1 1 auto' }} />
        <Button
          disabled={activeStep === 0 || loading}
          onClick={handleBack}
        >
          Back
        </Button>
        {activeStep === steps.length - 1 ? (
          <Button
            variant="contained"
            onClick={handleSubmit(onSubmit)}
            disabled={loading || !isStepValid(activeStep)}
          >
            {loading ? 'Creating...' : (bot ? 'Update Bot' : 'Create Bot')}
          </Button>
        ) : (
          <Button
            variant="contained"
            onClick={handleNext}
            disabled={!isStepValid(activeStep)}
          >
            Next
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
};

// Step Components (to be implemented in separate files)
const BasicConfigurationStep = () => <div>Basic Configuration Step</div>;
const StrategySelectionStep = () => <div>Strategy Selection Step</div>;
const RiskManagementStep = () => <div>Risk Management Step</div>;
const ReviewStep = () => <div>Review Step</div>;

export default BotConfigurationWizard;
