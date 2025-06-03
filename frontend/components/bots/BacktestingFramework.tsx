/**
 * Backtesting Framework Component
 * Comprehensive backtesting interface with historical data analysis
 */

'use client';

import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  LinearProgress,
  Alert,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Assessment as AssessmentIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Timeline as TimelineIcon,
  Download as DownloadIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { Bot, TRADING_SYMBOLS, TIMEFRAMES } from '../../types/bot';

interface BacktestConfig {
  symbol: string;
  timeframe: string;
  startDate: Date;
  endDate: Date;
  initialCapital: number;
  leverage: number;
  riskPerTrade: number;
  commission: number;
}

interface BacktestResult {
  id: string;
  config: BacktestConfig;
  performance: {
    totalReturn: number;
    totalReturnPercent: number;
    annualizedReturn: number;
    sharpeRatio: number;
    maxDrawdown: number;
    maxDrawdownPercent: number;
    winRate: number;
    profitFactor: number;
    totalTrades: number;
    winningTrades: number;
    losingTrades: number;
    averageWin: number;
    averageLoss: number;
    largestWin: number;
    largestLoss: number;
  };
  trades: any[];
  startTime: number;
  endTime: number;
  duration: number;
}

interface BacktestingFrameworkProps {
  bot: Bot;
  onBacktestComplete?: (result: BacktestResult) => void;
}

export const BacktestingFramework: React.FC<BacktestingFrameworkProps> = ({
  bot,
  onBacktestComplete,
}) => {
  const [config, setConfig] = useState<BacktestConfig>({
    symbol: bot.symbol,
    timeframe: bot.timeframe,
    startDate: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), // 30 days ago
    endDate: new Date(),
    initialCapital: 10000,
    leverage: 1,
    riskPerTrade: 2,
    commission: 0.1,
  });

  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentResult, setCurrentResult] = useState<BacktestResult | null>(null);
  const [historicalResults, setHistoricalResults] = useState<BacktestResult[]>([]);
  const [activeTab, setActiveTab] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Load historical backtest results for this bot
    loadHistoricalResults();
  }, [bot.id]);

  const loadHistoricalResults = async () => {
    try {
      // TODO: Implement API call to load historical backtest results
      // const response = await backtestService.getHistoricalResults(bot.id);
      // setHistoricalResults(response.data);
    } catch (error) {
      console.error('Failed to load historical results:', error);
    }
  };

  const runBacktest = async () => {
    try {
      setIsRunning(true);
      setProgress(0);
      setError(null);

      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 500);

      // TODO: Implement actual backtest API call
      // const response = await backtestService.runBacktest(bot.id, config);
      
      // Simulate backtest result
      const mockResult: BacktestResult = {
        id: `backtest_${Date.now()}`,
        config,
        performance: {
          totalReturn: 1250.75,
          totalReturnPercent: 12.51,
          annualizedReturn: 45.2,
          sharpeRatio: 1.85,
          maxDrawdown: -850.25,
          maxDrawdownPercent: -8.5,
          winRate: 68.5,
          profitFactor: 2.15,
          totalTrades: 127,
          winningTrades: 87,
          losingTrades: 40,
          averageWin: 95.25,
          averageLoss: -42.15,
          largestWin: 285.50,
          largestLoss: -125.75,
        },
        trades: [], // TODO: Add mock trades
        startTime: Date.now() - 5000,
        endTime: Date.now(),
        duration: 5000,
      };

      clearInterval(progressInterval);
      setProgress(100);
      setCurrentResult(mockResult);
      setHistoricalResults(prev => [mockResult, ...prev]);
      
      if (onBacktestComplete) {
        onBacktestComplete(mockResult);
      }

    } catch (error) {
      console.error('Backtest failed:', error);
      setError(error instanceof Error ? error.message : 'Backtest failed');
    } finally {
      setIsRunning(false);
    }
  };

  const stopBacktest = () => {
    setIsRunning(false);
    setProgress(0);
  };

  const exportResults = () => {
    if (currentResult) {
      const dataStr = JSON.stringify(currentResult, null, 2);
      const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
      
      const exportFileDefaultName = `backtest_${bot.name}_${new Date().toISOString().split('T')[0]}.json`;
      
      const linkElement = document.createElement('a');
      linkElement.setAttribute('href', dataUri);
      linkElement.setAttribute('download', exportFileDefaultName);
      linkElement.click();
    }
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  const formatPercentage = (value: number) => {
    return `${value.toFixed(2)}%`;
  };

  return (
    <LocalizationProvider dateAdapter={AdapterDateFns}>
      <Box>
        <Typography variant="h5" gutterBottom>
          Backtesting Framework
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Test your bot strategy against historical data to evaluate performance
        </Typography>

        <Grid container spacing={3}>
          {/* Configuration Panel */}
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Backtest Configuration
                </Typography>

                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Symbol</InputLabel>
                    <Select
                      value={config.symbol}
                      onChange={(e) => setConfig(prev => ({ ...prev, symbol: e.target.value }))}
                      label="Symbol"
                    >
                      {TRADING_SYMBOLS.map((symbol) => (
                        <MenuItem key={symbol} value={symbol}>
                          {symbol}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>

                  <FormControl fullWidth size="small">
                    <InputLabel>Timeframe</InputLabel>
                    <Select
                      value={config.timeframe}
                      onChange={(e) => setConfig(prev => ({ ...prev, timeframe: e.target.value }))}
                      label="Timeframe"
                    >
                      {TIMEFRAMES.map((timeframe) => (
                        <MenuItem key={timeframe} value={timeframe}>
                          {timeframe}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>

                  <DatePicker
                    label="Start Date"
                    value={config.startDate}
                    onChange={(date) => date && setConfig(prev => ({ ...prev, startDate: date }))}
                    slotProps={{ textField: { size: 'small', fullWidth: true } }}
                  />

                  <DatePicker
                    label="End Date"
                    value={config.endDate}
                    onChange={(date) => date && setConfig(prev => ({ ...prev, endDate: date }))}
                    slotProps={{ textField: { size: 'small', fullWidth: true } }}
                  />

                  <TextField
                    label="Initial Capital"
                    type="number"
                    value={config.initialCapital}
                    onChange={(e) => setConfig(prev => ({ ...prev, initialCapital: Number(e.target.value) }))}
                    size="small"
                    fullWidth
                  />

                  <TextField
                    label="Leverage"
                    type="number"
                    value={config.leverage}
                    onChange={(e) => setConfig(prev => ({ ...prev, leverage: Number(e.target.value) }))}
                    size="small"
                    fullWidth
                    inputProps={{ min: 1, max: 100 }}
                  />

                  <TextField
                    label="Risk per Trade (%)"
                    type="number"
                    value={config.riskPerTrade}
                    onChange={(e) => setConfig(prev => ({ ...prev, riskPerTrade: Number(e.target.value) }))}
                    size="small"
                    fullWidth
                    inputProps={{ min: 0.1, max: 10, step: 0.1 }}
                  />

                  <TextField
                    label="Commission (%)"
                    type="number"
                    value={config.commission}
                    onChange={(e) => setConfig(prev => ({ ...prev, commission: Number(e.target.value) }))}
                    size="small"
                    fullWidth
                    inputProps={{ min: 0, max: 1, step: 0.01 }}
                  />

                  <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
                    <Button
                      variant="contained"
                      startIcon={isRunning ? <StopIcon /> : <PlayIcon />}
                      onClick={isRunning ? stopBacktest : runBacktest}
                      disabled={isRunning}
                      fullWidth
                    >
                      {isRunning ? 'Stop' : 'Run Backtest'}
                    </Button>
                  </Box>

                  {isRunning && (
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="body2" gutterBottom>
                        Progress: {progress}%
                      </Typography>
                      <LinearProgress variant="determinate" value={progress} />
                    </Box>
                  )}

                  {error && (
                    <Alert severity="error" sx={{ mt: 2 }}>
                      {error}
                    </Alert>
                  )}
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Results Panel */}
          <Grid item xs={12} md={8}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6">
                    Backtest Results
                  </Typography>
                  {currentResult && (
                    <Box sx={{ display: 'flex', gap: 1 }}>
                      <Tooltip title="Export Results">
                        <IconButton onClick={exportResults} size="small">
                          <DownloadIcon />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Refresh">
                        <IconButton onClick={loadHistoricalResults} size="small">
                          <RefreshIcon />
                        </IconButton>
                      </Tooltip>
                    </Box>
                  )}
                </Box>

                <Tabs value={activeTab} onChange={(_, newValue) => setActiveTab(newValue)}>
                  <Tab label="Current Result" />
                  <Tab label="Historical Results" />
                  <Tab label="Performance Chart" />
                </Tabs>

                <Box sx={{ mt: 2 }}>
                  {activeTab === 0 && currentResult && (
                    <PerformanceMetrics result={currentResult} />
                  )}
                  {activeTab === 1 && (
                    <HistoricalResults results={historicalResults} />
                  )}
                  {activeTab === 2 && currentResult && (
                    <PerformanceChart result={currentResult} />
                  )}
                </Box>

                {!currentResult && !isRunning && (
                  <Box sx={{ textAlign: 'center', py: 4 }}>
                    <AssessmentIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                    <Typography variant="h6" color="text.secondary">
                      No backtest results yet
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Configure your backtest parameters and click "Run Backtest" to get started
                    </Typography>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Box>
    </LocalizationProvider>
  );
};

// Performance Metrics Component
const PerformanceMetrics: React.FC<{ result: BacktestResult }> = ({ result }) => {
  const { performance } = result;

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  const formatPercentage = (value: number) => {
    return `${value.toFixed(2)}%`;
  };

  return (
    <Grid container spacing={2}>
      <Grid item xs={6} sm={4}>
        <Card variant="outlined">
          <CardContent sx={{ textAlign: 'center' }}>
            <Typography variant="h4" color={performance.totalReturn >= 0 ? 'success.main' : 'error.main'}>
              {formatCurrency(performance.totalReturn)}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Total Return ({formatPercentage(performance.totalReturnPercent)})
            </Typography>
          </CardContent>
        </Card>
      </Grid>
      
      <Grid item xs={6} sm={4}>
        <Card variant="outlined">
          <CardContent sx={{ textAlign: 'center' }}>
            <Typography variant="h4" color="primary">
              {formatPercentage(performance.winRate)}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Win Rate
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={6} sm={4}>
        <Card variant="outlined">
          <CardContent sx={{ textAlign: 'center' }}>
            <Typography variant="h4">
              {performance.sharpeRatio.toFixed(2)}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Sharpe Ratio
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={6} sm={4}>
        <Card variant="outlined">
          <CardContent sx={{ textAlign: 'center' }}>
            <Typography variant="h4" color="error">
              {formatPercentage(performance.maxDrawdownPercent)}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Max Drawdown
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={6} sm={4}>
        <Card variant="outlined">
          <CardContent sx={{ textAlign: 'center' }}>
            <Typography variant="h4">
              {performance.totalTrades}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Total Trades
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={6} sm={4}>
        <Card variant="outlined">
          <CardContent sx={{ textAlign: 'center' }}>
            <Typography variant="h4">
              {performance.profitFactor.toFixed(2)}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Profit Factor
            </Typography>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

// Historical Results Component
const HistoricalResults: React.FC<{ results: BacktestResult[] }> = ({ results }) => {
  if (results.length === 0) {
    return (
      <Box sx={{ textAlign: 'center', py: 4 }}>
        <Typography variant="body1" color="text.secondary">
          No historical backtest results found
        </Typography>
      </Box>
    );
  }

  return (
    <TableContainer component={Paper} variant="outlined">
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Date</TableCell>
            <TableCell>Symbol</TableCell>
            <TableCell>Timeframe</TableCell>
            <TableCell align="right">Return</TableCell>
            <TableCell align="right">Win Rate</TableCell>
            <TableCell align="right">Trades</TableCell>
            <TableCell align="right">Sharpe</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {results.map((result) => (
            <TableRow key={result.id}>
              <TableCell>
                {new Date(result.startTime).toLocaleDateString()}
              </TableCell>
              <TableCell>{result.config.symbol}</TableCell>
              <TableCell>{result.config.timeframe}</TableCell>
              <TableCell align="right">
                <Chip
                  label={`${result.performance.totalReturnPercent.toFixed(2)}%`}
                  color={result.performance.totalReturn >= 0 ? 'success' : 'error'}
                  size="small"
                />
              </TableCell>
              <TableCell align="right">
                {result.performance.winRate.toFixed(1)}%
              </TableCell>
              <TableCell align="right">
                {result.performance.totalTrades}
              </TableCell>
              <TableCell align="right">
                {result.performance.sharpeRatio.toFixed(2)}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

// Performance Chart Component (placeholder)
const PerformanceChart: React.FC<{ result: BacktestResult }> = ({ result }) => {
  return (
    <Box sx={{ textAlign: 'center', py: 4 }}>
      <TimelineIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
      <Typography variant="h6" color="text.secondary">
        Performance Chart
      </Typography>
      <Typography variant="body2" color="text.secondary">
        Chart visualization will be implemented here
      </Typography>
    </Box>
  );
};

export default BacktestingFramework;
