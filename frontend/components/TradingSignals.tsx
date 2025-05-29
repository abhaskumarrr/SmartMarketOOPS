import React, { useEffect, useState } from 'react';
import {
  Paper,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  Box,
  Chip,
  Alert,
  CircularProgress,
} from '@mui/material';
import {
  TrendingUp as BuyIcon,
  TrendingDown as SellIcon,
  RemoveCircleOutline as HoldIcon,
  ErrorOutline as ErrorIcon,
} from '@mui/icons-material';
import useWebSocket from '../lib/hooks/useWebSocket';
import { ConnectionStatus } from '../lib/websocketService';

interface SignalData {
  id: string;
  symbol: string;
  type: 'buy' | 'sell' | 'hold';
  price: number;
  timestamp: string;
  confidence: number;
  reason?: string;
}

interface TradingSignalsProps {
  symbol?: string;
  maxSignals?: number;
}

const TradingSignals: React.FC<TradingSignalsProps> = ({
  symbol = 'BTCUSD',
  maxSignals = 10,
}) => {
  const [signals, setSignals] = useState<SignalData[]>([]);
  
  // Connect to WebSocket for trading signals
  const { 
    data: signalData, 
    status, 
    error,
    isConnected,
    isConnecting 
  } = useWebSocket<SignalData>(
    'signal',
    'signal:new',
    symbol
  );
  
  // Add new signals to the list when they arrive
  useEffect(() => {
    if (signalData && signalData.symbol === symbol) {
      setSignals(prevSignals => {
        // Add new signal to the beginning of the array
        const newSignals = [signalData, ...prevSignals];
        // Limit the number of signals displayed
        return newSignals.slice(0, maxSignals);
      });
    }
  }, [signalData, symbol, maxSignals]);
  
  // Get icon and color based on signal type
  const getSignalInfo = (type: string) => {
    switch (type) {
      case 'buy':
        return {
          icon: <BuyIcon />,
          color: 'success.main',
          label: 'BUY',
        };
      case 'sell':
        return {
          icon: <SellIcon />,
          color: 'error.main',
          label: 'SELL',
        };
      case 'hold':
      default:
        return {
          icon: <HoldIcon />,
          color: 'warning.main',
          label: 'HOLD',
        };
    }
  };
  
  // Format date from timestamp
  const formatDate = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
  };
  
  // Format confidence percentage
  const formatConfidence = (confidence: number) => {
    return `${Math.round(confidence * 100)}%`;
  };
  
  return (
    <Paper 
      elevation={3} 
      sx={{ 
        p: 2, 
        height: '100%', 
        display: 'flex', 
        flexDirection: 'column',
        overflow: 'hidden'
      }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" component="h2">
          Trading Signals
        </Typography>
        
        {/* Connection status */}
        {status === ConnectionStatus.CONNECTING && (
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <CircularProgress size={16} sx={{ mr: 1 }} />
            <Typography variant="caption">Connecting...</Typography>
          </Box>
        )}
        
        {status === ConnectionStatus.CONNECTED && (
          <Chip 
            label="Live" 
            color="success" 
            size="small" 
            sx={{ height: 24 }}
          />
        )}
        
        {status === ConnectionStatus.ERROR && (
          <Chip 
            label="Connection Error" 
            color="error" 
            size="small" 
            icon={<ErrorIcon />}
            sx={{ height: 24 }}
          />
        )}
      </Box>
      
      {/* Error message */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error.message}
        </Alert>
      )}
      
      {/* Loading state */}
      {isConnecting && signals.length === 0 && (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress />
        </Box>
      )}
      
      {/* No signals message */}
      {isConnected && signals.length === 0 && (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <Typography variant="body2" color="text.secondary">
            No trading signals yet. Signals will appear here as they are generated.
          </Typography>
        </Box>
      )}
      
      {/* Signal list */}
      {signals.length > 0 && (
        <List sx={{ overflow: 'auto', flexGrow: 1 }}>
          {signals.map((signal, index) => {
            const { icon, color, label } = getSignalInfo(signal.type);
            
            return (
              <React.Fragment key={signal.id || index}>
                {index > 0 && <Divider component="li" />}
                <ListItem alignItems="flex-start">
                  <ListItemIcon sx={{ color }}>
                    {icon}
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Typography variant="subtitle2" component="span">
                          {label} @ ${signal.price.toLocaleString()}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {formatDate(signal.timestamp)}
                        </Typography>
                      </Box>
                    }
                    secondary={
                      <>
                        <Typography variant="body2" component="span">
                          {signal.reason || `${symbol} signal with ${formatConfidence(signal.confidence)} confidence`}
                        </Typography>
                        <Box sx={{ mt: 1, display: 'flex', alignItems: 'center' }}>
                          <Chip 
                            label={`Confidence: ${formatConfidence(signal.confidence)}`}
                            size="small"
                            sx={{ 
                              height: 20, 
                              fontSize: '0.7rem',
                              bgcolor: `${color}20`,
                              color: color
                            }}
                          />
                        </Box>
                      </>
                    }
                  />
                </ListItem>
              </React.Fragment>
            );
          })}
        </List>
      )}
    </Paper>
  );
};

export default TradingSignals; 