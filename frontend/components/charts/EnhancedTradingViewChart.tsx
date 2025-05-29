import React, { useState, useEffect } from 'react';
import { TradingViewChartContainer } from './TradingViewChartContainer';
import { ChartControls } from './ChartControls';
import { formatPredictionData, formatSignalData, defaultIndicators } from './TradingViewConfig';
import { Box, CircularProgress, Typography } from '@mui/material';

interface EnhancedTradingViewChartProps {
  initialSymbol?: string;
  initialInterval?: string;
  initialDarkMode?: boolean;
  height?: number;
  predictionsData?: any[];
  signalsData?: any[];
  isLoading?: boolean;
}

export const EnhancedTradingViewChart: React.FC<EnhancedTradingViewChartProps> = ({
  initialSymbol = 'BTCUSD',
  initialInterval = '60', // 1h
  initialDarkMode = true,
  height = 600,
  predictionsData = [],
  signalsData = [],
  isLoading = false,
}) => {
  // State
  const [symbol, setSymbol] = useState(initialSymbol);
  const [interval, setInterval] = useState(initialInterval);
  const [darkMode, setDarkMode] = useState(initialDarkMode);
  const [showPredictions, setShowPredictions] = useState(true);
  const [showIndicators, setShowIndicators] = useState(true);
  const [activeIndicators, setActiveIndicators] = useState(defaultIndicators);
  const [predictions, setPredictions] = useState(formatPredictionData(predictionsData));
  const [signals, setSignals] = useState(formatSignalData(signalsData));
  
  // Handle prediction data updates
  useEffect(() => {
    setPredictions(formatPredictionData(predictionsData));
  }, [predictionsData]);
  
  // Handle signal data updates
  useEffect(() => {
    setSignals(formatSignalData(signalsData));
  }, [signalsData]);
  
  // Handle symbol change
  const handleSymbolChange = (newSymbol: string) => {
    setSymbol(newSymbol);
    // In a real app, this would fetch new data for the selected symbol
  };
  
  // Handle interval change
  const handleIntervalChange = (newInterval: string) => {
    setInterval(newInterval);
    // In a real app, this would fetch new data for the selected timeframe
  };
  
  // Toggle dark mode
  const handleToggleDarkMode = () => {
    setDarkMode(!darkMode);
  };
  
  // Toggle prediction overlay
  const handleTogglePredictions = () => {
    setShowPredictions(!showPredictions);
  };
  
  // Toggle technical indicators
  const handleToggleIndicators = () => {
    setShowIndicators(!showIndicators);
  };
  
  // Add a technical indicator
  const handleAddIndicator = (indicatorId: string) => {
    // Find the indicator in the available indicators list
    const indicator = defaultIndicators.find(ind => ind.id === indicatorId);
    
    if (indicator) {
      // Check if already added to avoid duplicates
      if (!activeIndicators.some(ind => ind.id === indicatorId)) {
        setActiveIndicators([...activeIndicators, indicator]);
      }
    }
  };
  
  return (
    <div className="enhanced-trading-view-chart" style={{ 
      width: '100%',
      backgroundColor: darkMode ? '#181A20' : '#FFFFFF',
      borderRadius: '8px',
      overflow: 'hidden',
      boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
      height: height
    }}>
      <ChartControls 
        symbol={symbol}
        interval={interval}
        darkMode={darkMode}
        showPredictions={showPredictions}
        showIndicators={showIndicators}
        onSymbolChange={handleSymbolChange}
        onIntervalChange={handleIntervalChange}
        onToggleDarkMode={handleToggleDarkMode}
        onTogglePredictions={handleTogglePredictions}
        onToggleIndicators={handleToggleIndicators}
        onAddIndicator={handleAddIndicator}
      />
      
      {isLoading ? (
        <Box 
          display="flex" 
          justifyContent="center" 
          alignItems="center" 
          height={height - 100} // Adjust for controls and footer
        >
          <CircularProgress />
        </Box>
      ) : predictionsData.length === 0 && signalsData.length === 0 ? (
        <Box 
          display="flex" 
          justifyContent="center" 
          alignItems="center" 
          height={height - 100} // Adjust for controls and footer
          flexDirection="column"
        >
          <Typography 
            variant="body1" 
            color={darkMode ? "#B0B0C0" : "#5F5F7A"}
            sx={{ mb: 2 }}
          >
            No prediction or signal data available
          </Typography>
          <Typography 
            variant="body2" 
            color={darkMode ? "#B0B0C0" : "#5F5F7A"}
          >
            The chart will update when data becomes available
          </Typography>
        </Box>
      ) : (
        <div style={{ height: height - 100 }}>
          <TradingViewChartContainer 
            symbol={symbol}
            interval={interval}
            darkMode={darkMode}
            predictionsData={predictions}
            signals={signals}
            showPredictions={showPredictions}
            showIndicators={showIndicators}
          />
        </div>
      )}
      
      <div style={{ 
        padding: '8px 16px',
        fontSize: '12px',
        color: darkMode ? '#999' : '#666',
        borderTop: darkMode ? '1px solid #363A45' : '1px solid #E0E3EB',
        textAlign: 'center'
      }}>
        Predictions are based on machine learning models and should not be considered as financial advice.
      </div>
    </div>
  );
}; 