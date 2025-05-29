import React from 'react';
import { timeIntervals, technicalIndicators } from './TradingViewConfig';

interface ChartControlsProps {
  symbol: string;
  interval: string;
  darkMode: boolean;
  showPredictions: boolean;
  showIndicators: boolean;
  onSymbolChange: (symbol: string) => void;
  onIntervalChange: (interval: string) => void;
  onToggleDarkMode: () => void;
  onTogglePredictions: () => void;
  onToggleIndicators: () => void;
  onAddIndicator?: (indicatorId: string) => void;
}

export const ChartControls: React.FC<ChartControlsProps> = ({
  symbol,
  interval,
  darkMode,
  showPredictions,
  showIndicators,
  onSymbolChange,
  onIntervalChange,
  onToggleDarkMode,
  onTogglePredictions,
  onToggleIndicators,
  onAddIndicator,
}) => {
  // Common styles
  const buttonStyle = {
    padding: '8px 12px',
    margin: '0 4px',
    borderRadius: '4px',
    border: 'none',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: 500,
    transition: 'all 0.2s ease',
    backgroundColor: darkMode ? '#2A2E39' : '#F0F3FA',
    color: darkMode ? '#D9D9D9' : '#131722',
  };

  const activeButtonStyle = {
    ...buttonStyle,
    backgroundColor: darkMode ? '#3949AB' : '#2962FF',
    color: '#FFFFFF',
  };

  const selectStyle = {
    padding: '8px 12px',
    margin: '0 4px',
    borderRadius: '4px',
    border: darkMode ? '1px solid #363A45' : '1px solid #D1D4DC',
    backgroundColor: darkMode ? '#2A2E39' : '#FFFFFF',
    color: darkMode ? '#D9D9D9' : '#131722',
    fontSize: '14px',
    cursor: 'pointer',
  };

  // Predefined crypto pairs
  const commonSymbols = ['BTCUSD', 'ETHUSD', 'XRPUSD', 'SOLUSD', 'BNBUSD', 'ADAUSD'];

  return (
    <div style={{ 
      display: 'flex', 
      flexWrap: 'wrap', 
      alignItems: 'center', 
      padding: '12px', 
      borderRadius: '8px',
      backgroundColor: darkMode ? '#181A20' : '#FFFFFF',
      border: darkMode ? '1px solid #363A45' : '1px solid #E0E3EB',
      marginBottom: '16px',
    }}>
      {/* Symbol selection */}
      <div style={{ margin: '8px', display: 'flex', alignItems: 'center' }}>
        <label style={{ 
          marginRight: '8px', 
          fontSize: '14px',
          color: darkMode ? '#D9D9D9' : '#131722',
        }}>
          Symbol:
        </label>
        <select 
          value={symbol} 
          onChange={(e) => onSymbolChange(e.target.value)}
          style={selectStyle}
        >
          {commonSymbols.map((sym) => (
            <option key={sym} value={sym}>{sym}</option>
          ))}
        </select>
      </div>
      
      {/* Time interval selection */}
      <div style={{ margin: '8px', display: 'flex', alignItems: 'center' }}>
        <label style={{ 
          marginRight: '8px', 
          fontSize: '14px',
          color: darkMode ? '#D9D9D9' : '#131722',
        }}>
          Interval:
        </label>
        <select 
          value={interval} 
          onChange={(e) => onIntervalChange(e.target.value)}
          style={selectStyle}
        >
          {timeIntervals.map((int) => (
            <option key={int.value} value={int.value}>{int.text}</option>
          ))}
        </select>
      </div>
      
      {/* Theme toggle */}
      <div style={{ margin: '8px' }}>
        <button 
          onClick={onToggleDarkMode}
          style={buttonStyle}
          title={darkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
        >
          {darkMode ? '☀️ Light' : '🌙 Dark'}
        </button>
      </div>
      
      {/* Prediction overlay toggle */}
      <div style={{ margin: '8px' }}>
        <button 
          onClick={onTogglePredictions}
          style={showPredictions ? activeButtonStyle : buttonStyle}
          title={showPredictions ? 'Hide Predictions' : 'Show Predictions'}
        >
          📊 Predictions
        </button>
      </div>
      
      {/* Technical indicators toggle */}
      <div style={{ margin: '8px' }}>
        <button 
          onClick={onToggleIndicators}
          style={showIndicators ? activeButtonStyle : buttonStyle}
          title={showIndicators ? 'Hide Indicators' : 'Show Indicators'}
        >
          📈 Indicators
        </button>
      </div>
      
      {/* Indicator selection dropdown */}
      {showIndicators && onAddIndicator && (
        <div style={{ margin: '8px', display: 'flex', alignItems: 'center' }}>
          <label style={{ 
            marginRight: '8px', 
            fontSize: '14px',
            color: darkMode ? '#D9D9D9' : '#131722',
          }}>
            Add:
          </label>
          <select 
            onChange={(e) => {
              onAddIndicator(e.target.value);
              e.target.value = '';
            }}
            defaultValue=""
            style={selectStyle}
          >
            <option value="" disabled>Select indicator</option>
            {technicalIndicators.map((indicator) => (
              <option key={indicator.id} value={indicator.id}>
                {indicator.name}
              </option>
            ))}
          </select>
        </div>
      )}
    </div>
  );
}; 