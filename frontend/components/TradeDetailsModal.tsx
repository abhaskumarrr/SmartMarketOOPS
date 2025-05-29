import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Grid,
  Divider,
  Box,
  Chip,
  IconButton,
} from '@mui/material';
import {
  Close as CloseIcon,
  Receipt as ReceiptIcon,
  AttachMoney as FeeIcon,
  AccessTime as TimeIcon,
  Notes as NotesIcon,
} from '@mui/icons-material';
import { Trade } from './TradeHistory';

interface TradeDetailsModalProps {
  trade: Trade | null;
  open: boolean;
  onClose: () => void;
  darkMode: boolean;
}

export const TradeDetailsModal: React.FC<TradeDetailsModalProps> = ({
  trade,
  open,
  onClose,
  darkMode,
}) => {
  if (!trade) return null;

  // Get styles based on dark mode
  const getStyles = () => {
    return {
      dialog: {
        backgroundColor: darkMode ? '#1E1E2D' : '#FFFFFF',
        color: darkMode ? '#E0E0E0' : '#121212',
      },
      title: {
        color: darkMode ? '#E0E0E0' : '#121212',
      },
      text: {
        color: darkMode ? '#E0E0E0' : '#121212',
      },
      subtitle: {
        color: darkMode ? '#B0B0C0' : '#5F5F7A',
      },
      divider: {
        backgroundColor: darkMode ? '#3A3A50' : '#E0E0E0',
      },
      buyChip: {
        backgroundColor: darkMode ? '#143C1E' : '#E6F4EA',
        color: darkMode ? '#4CAF50' : '#1E8E3E',
      },
      sellChip: {
        backgroundColor: darkMode ? '#5B2626' : '#FDEDED',
        color: darkMode ? '#FF5252' : '#D32F2F',
      },
      profitChip: {
        backgroundColor: darkMode ? '#143C1E' : '#E6F4EA',
        color: darkMode ? '#4CAF50' : '#1E8E3E',
      },
      lossChip: {
        backgroundColor: darkMode ? '#5B2626' : '#FDEDED',
        color: darkMode ? '#FF5252' : '#D32F2F',
      },
      pendingChip: {
        backgroundColor: darkMode ? '#494833' : '#FFF8E1',
        color: darkMode ? '#FFC107' : '#F57C00',
      },
      executedChip: {
        backgroundColor: darkMode ? '#143C1E' : '#E6F4EA',
        color: darkMode ? '#4CAF50' : '#1E8E3E',
      },
      cancelledChip: {
        backgroundColor: darkMode ? '#37374F' : '#ECECF1',
        color: darkMode ? '#B0B0C0' : '#5F5F7A',
      },
      closeButton: {
        color: darkMode ? '#B0B0C0' : '#5F5F7A',
      },
      sectionIcon: {
        color: darkMode ? '#90CAF9' : '#1976D2',
      },
      actions: {
        backgroundColor: darkMode ? '#1E1E2D' : '#FFFFFF',
        borderTop: `1px solid ${darkMode ? '#3A3A50' : '#E0E0E0'}`,
      },
    };
  };
  
  const styles = getStyles();

  // Format timestamp to readable date
  const formatDate = (timestamp: number) => {
    return new Date(timestamp).toLocaleString();
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      fullWidth
      maxWidth="md"
      PaperProps={{
        style: styles.dialog,
        sx: { borderRadius: '12px' },
      }}
    >
      <DialogTitle sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', pb: 1 }}>
        <Typography variant="h6" style={styles.title}>
          Trade Details
        </Typography>
        <IconButton onClick={onClose} size="small" style={styles.closeButton}>
          <CloseIcon />
        </IconButton>
      </DialogTitle>
      
      <DialogContent dividers sx={{ px: 3, py: 2 }}>
        <Grid container spacing={3}>
          {/* Trade Summary */}
          <Grid item xs={12}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Box>
                <Typography variant="h5" style={styles.text}>
                  {trade.symbol}
                </Typography>
                <Typography variant="body2" style={styles.subtitle}>
                  ID: {trade.id}
                </Typography>
              </Box>
              
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Chip 
                  label={trade.type.toUpperCase()} 
                  style={trade.type === 'buy' ? styles.buyChip : styles.sellChip}
                />
                <Chip 
                  label={trade.status.toUpperCase()} 
                  style={
                    trade.status === 'executed' ? styles.executedChip :
                    trade.status === 'pending' ? styles.pendingChip : styles.cancelledChip
                  }
                />
              </Box>
            </Box>
            <Divider style={styles.divider} sx={{ mb: 3 }} />
          </Grid>
          
          {/* Price Details */}
          <Grid item xs={12} md={6}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <ReceiptIcon style={styles.sectionIcon} sx={{ mr: 1 }} />
              <Typography variant="subtitle1" style={styles.text}>
                Price Details
              </Typography>
            </Box>
            
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Typography variant="body2" style={styles.subtitle}>
                  Price
                </Typography>
                <Typography variant="body1" style={styles.text}>
                  ${trade.price.toLocaleString()}
                </Typography>
              </Grid>
              
              <Grid item xs={6}>
                <Typography variant="body2" style={styles.subtitle}>
                  Amount
                </Typography>
                <Typography variant="body1" style={styles.text}>
                  {trade.amount.toLocaleString()} {trade.symbol.split('/')[0]}
                </Typography>
              </Grid>
              
              <Grid item xs={6}>
                <Typography variant="body2" style={styles.subtitle}>
                  Total Value
                </Typography>
                <Typography variant="body1" style={styles.text}>
                  ${trade.value.toLocaleString()}
                </Typography>
              </Grid>
              
              <Grid item xs={6}>
                <Typography variant="body2" style={styles.subtitle}>
                  Fee
                </Typography>
                <Typography variant="body1" style={styles.text}>
                  ${trade.fee.toLocaleString()}
                </Typography>
              </Grid>
              
              {trade.profit !== undefined && (
                <Grid item xs={12}>
                  <Typography variant="body2" style={styles.subtitle}>
                    Profit/Loss
                  </Typography>
                  <Chip 
                    label={`${trade.profit > 0 ? '+' : ''}$${trade.profit.toLocaleString()} (${trade.profitPercentage}%)`} 
                    style={trade.profit > 0 ? styles.profitChip : styles.lossChip}
                  />
                </Grid>
              )}
            </Grid>
          </Grid>
          
          {/* Time & Status */}
          <Grid item xs={12} md={6}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <TimeIcon style={styles.sectionIcon} sx={{ mr: 1 }} />
              <Typography variant="subtitle1" style={styles.text}>
                Time & Status
              </Typography>
            </Box>
            
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <Typography variant="body2" style={styles.subtitle}>
                  Trade Time
                </Typography>
                <Typography variant="body1" style={styles.text}>
                  {formatDate(trade.timestamp)}
                </Typography>
              </Grid>
              
              <Grid item xs={12}>
                <Typography variant="body2" style={styles.subtitle}>
                  Status
                </Typography>
                <Typography variant="body1" style={styles.text}>
                  {trade.status === 'executed' ? 'Executed' : 
                   trade.status === 'pending' ? 'Pending Execution' : 'Cancelled'}
                </Typography>
              </Grid>
              
              {/* Additional dummy data for demonstration */}
              <Grid item xs={12}>
                <Typography variant="body2" style={styles.subtitle}>
                  Exchange
                </Typography>
                <Typography variant="body1" style={styles.text}>
                  Delta Exchange
                </Typography>
              </Grid>
              
              <Grid item xs={12}>
                <Typography variant="body2" style={styles.subtitle}>
                  Order Type
                </Typography>
                <Typography variant="body1" style={styles.text}>
                  Market Order
                </Typography>
              </Grid>
            </Grid>
          </Grid>
          
          {/* Notes Section */}
          <Grid item xs={12}>
            <Divider style={styles.divider} sx={{ mt: 1, mb: 3 }} />
            
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <NotesIcon style={styles.sectionIcon} sx={{ mr: 1 }} />
              <Typography variant="subtitle1" style={styles.text}>
                Notes & Tags
              </Typography>
            </Box>
            
            <Typography variant="body2" style={styles.text} sx={{ mb: 2 }}>
              {/* Dummy notes for demonstration */}
              {trade.type === 'buy' 
                ? 'Entry based on MACD crossover and RSI oversold condition. Price action showing strong support at this level.' 
                : 'Exit triggered by take profit level. Market showing signs of resistance at upper Bollinger Band.'}
            </Typography>
            
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Chip label="Strategy A" size="small" sx={{ 
                backgroundColor: darkMode ? '#2A2A3C' : '#F5F5F5',
                color: darkMode ? '#E0E0E0' : '#121212',
              }} />
              <Chip label="Swing Trade" size="small" sx={{ 
                backgroundColor: darkMode ? '#2A2A3C' : '#F5F5F5',
                color: darkMode ? '#E0E0E0' : '#121212',
              }} />
              <Chip label="Technical" size="small" sx={{ 
                backgroundColor: darkMode ? '#2A2A3C' : '#F5F5F5',
                color: darkMode ? '#E0E0E0' : '#121212',
              }} />
            </Box>
          </Grid>
        </Grid>
      </DialogContent>
      
      <DialogActions style={styles.actions}>
        <Button 
          onClick={onClose}
          sx={{
            color: darkMode ? '#90CAF9' : '#1976D2',
            '&:hover': {
              backgroundColor: darkMode ? 'rgba(144, 202, 249, 0.08)' : 'rgba(25, 118, 210, 0.08)'
            }
          }}
        >
          Close
        </Button>
        {trade.status === 'pending' && (
          <Button 
            color="error"
            sx={{
              color: darkMode ? '#FF5252' : '#D32F2F',
              '&:hover': {
                backgroundColor: darkMode ? 'rgba(255, 82, 82, 0.08)' : 'rgba(211, 47, 47, 0.08)'
              }
            }}
          >
            Cancel Trade
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
};

export default TradeDetailsModal; 