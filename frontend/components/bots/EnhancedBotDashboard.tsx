/**
 * Enhanced Bot Management Dashboard
 * Comprehensive dashboard with performance monitoring and backtesting
 */

'use client';

import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Button,
  Grid,
  Card,
  CardContent,
  Chip,
  IconButton,
  Tooltip,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tabs,
  Tab,
  Badge,
  LinearProgress,
} from '@mui/material';
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Assessment as AssessmentIcon,
  Visibility as VisibilityIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
} from '@mui/icons-material';
import { Bot } from '../../types/bot';
import { botService } from '../../lib/services/botService';
import BotConfigurationWizard from './BotConfigurationWizard';
import BotPerformanceMonitor from './BotPerformanceMonitor';
import BacktestingFramework from './BacktestingFramework';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`bot-tabpanel-${index}`}
      aria-labelledby={`bot-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

export const EnhancedBotDashboard: React.FC = () => {
  const [bots, setBots] = useState<Bot[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedBot, setSelectedBot] = useState<Bot | null>(null);
  const [showConfigWizard, setShowConfigWizard] = useState(false);
  const [showPerformanceMonitor, setShowPerformanceMonitor] = useState(false);
  const [showBacktesting, setShowBacktesting] = useState(false);
  const [deleteConfirmBot, setDeleteConfirmBot] = useState<Bot | null>(null);
  const [activeTab, setActiveTab] = useState(0);

  useEffect(() => {
    loadBots();
    const interval = setInterval(loadBots, 30000);
    return () => clearInterval(interval);
  }, []);

  const loadBots = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await botService.getBots();
      
      if (response.success) {
        setBots(response.data);
      } else {
        setError(response.message || 'Failed to load bots');
      }
    } catch (error) {
      console.error('Error loading bots:', error);
      setError(error instanceof Error ? error.message : 'Failed to load bots');
    } finally {
      setLoading(false);
    }
  };

  const handleCreateBot = () => {
    setSelectedBot(null);
    setShowConfigWizard(true);
  };

  const handleEditBot = (bot: Bot) => {
    setSelectedBot(bot);
    setShowConfigWizard(true);
  };

  const handleViewPerformance = (bot: Bot) => {
    setSelectedBot(bot);
    setShowPerformanceMonitor(true);
  };

  const handleRunBacktest = (bot: Bot) => {
    setSelectedBot(bot);
    setShowBacktesting(true);
  };

  const handleDeleteBot = async (bot: Bot) => {
    try {
      const response = await botService.deleteBot(bot.id);
      
      if (response.success) {
        await loadBots();
        setDeleteConfirmBot(null);
      } else {
        setError(response.message || 'Failed to delete bot');
      }
    } catch (error) {
      console.error('Error deleting bot:', error);
      setError(error instanceof Error ? error.message : 'Failed to delete bot');
    }
  };

  const handleStartBot = async (bot: Bot) => {
    try {
      const response = await botService.startBot(bot.id);
      
      if (response.success) {
        await loadBots();
      } else {
        setError(response.message || 'Failed to start bot');
      }
    } catch (error) {
      console.error('Error starting bot:', error);
      setError(error instanceof Error ? error.message : 'Failed to start bot');
    }
  };

  const handleStopBot = async (bot: Bot) => {
    try {
      const response = await botService.stopBot(bot.id);
      
      if (response.success) {
        await loadBots();
      } else {
        setError(response.message || 'Failed to stop bot');
      }
    } catch (error) {
      console.error('Error stopping bot:', error);
      setError(error instanceof Error ? error.message : 'Failed to stop bot');
    }
  };

  const getRunningBotsCount = () => bots.filter(bot => bot.isActive).length;
  const getStrategyDistribution = () => {
    const distribution: Record<string, number> = {};
    bots.forEach(bot => {
      distribution[bot.strategy] = (distribution[bot.strategy] || 0) + 1;
    });
    return distribution;
  };

  if (loading && bots.length === 0) {
    return (
      <Box sx={{ p: 3 }}>
        <LinearProgress />
        <Typography sx={{ mt: 2 }}>Loading bots...</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1">
          Enhanced Bot Management
        </Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={handleCreateBot}
          size="large"
        >
          Create Bot
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Dashboard Overview */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h3" color="primary">
                {bots.length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Total Bots
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h3" color="success.main">
                {getRunningBotsCount()}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Running Bots
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h3" color="warning.main">
                {Object.keys(getStrategyDistribution()).length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Strategies
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h3" color="info.main">
                {getRunningBotsCount() > 0 ? '24/7' : 'Offline'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Status
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Tabs for different views */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={activeTab} onChange={(_, newValue) => setActiveTab(newValue)}>
          <Tab 
            label={
              <Badge badgeContent={bots.length} color="primary">
                All Bots
              </Badge>
            } 
          />
          <Tab 
            label={
              <Badge badgeContent={getRunningBotsCount()} color="success">
                Active Bots
              </Badge>
            } 
          />
        </Tabs>
      </Box>

      <TabPanel value={activeTab} index={0}>
        <BotGridView 
          bots={bots}
          onEdit={handleEditBot}
          onDelete={setDeleteConfirmBot}
          onStart={handleStartBot}
          onStop={handleStopBot}
          onViewPerformance={handleViewPerformance}
          onRunBacktest={handleRunBacktest}
        />
      </TabPanel>

      <TabPanel value={activeTab} index={1}>
        <BotGridView 
          bots={bots.filter(bot => bot.isActive)}
          onEdit={handleEditBot}
          onDelete={setDeleteConfirmBot}
          onStart={handleStartBot}
          onStop={handleStopBot}
          onViewPerformance={handleViewPerformance}
          onRunBacktest={handleRunBacktest}
        />
      </TabPanel>

      {/* Dialogs */}
      <BotConfigurationWizard
        open={showConfigWizard}
        onClose={() => setShowConfigWizard(false)}
        onSuccess={() => {
          setShowConfigWizard(false);
          loadBots();
        }}
        bot={selectedBot}
      />

      <Dialog
        open={showPerformanceMonitor}
        onClose={() => setShowPerformanceMonitor(false)}
        maxWidth="xl"
        fullWidth
      >
        <DialogTitle>Bot Performance Monitor</DialogTitle>
        <DialogContent>
          {selectedBot && (
            <BotPerformanceMonitor
              bot={selectedBot}
              onBotUpdate={(updatedBot) => {
                setBots(prev => prev.map(b => b.id === updatedBot.id ? updatedBot : b));
              }}
            />
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowPerformanceMonitor(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      <Dialog
        open={showBacktesting}
        onClose={() => setShowBacktesting(false)}
        maxWidth="xl"
        fullWidth
      >
        <DialogTitle>Backtesting Framework</DialogTitle>
        <DialogContent>
          {selectedBot && (
            <BacktestingFramework
              bot={selectedBot}
              onBacktestComplete={(result) => console.log('Backtest completed:', result)}
            />
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowBacktesting(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      <Dialog open={!!deleteConfirmBot} onClose={() => setDeleteConfirmBot(null)}>
        <DialogTitle>Delete Bot</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete "{deleteConfirmBot?.name}"? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteConfirmBot(null)}>Cancel</Button>
          <Button 
            onClick={() => deleteConfirmBot && handleDeleteBot(deleteConfirmBot)}
            color="error"
            variant="contained"
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

// Bot Grid View Component
interface BotGridViewProps {
  bots: Bot[];
  onEdit: (bot: Bot) => void;
  onDelete: (bot: Bot) => void;
  onStart: (bot: Bot) => void;
  onStop: (bot: Bot) => void;
  onViewPerformance: (bot: Bot) => void;
  onRunBacktest: (bot: Bot) => void;
}

const BotGridView: React.FC<BotGridViewProps> = ({
  bots,
  onEdit,
  onDelete,
  onStart,
  onStop,
  onViewPerformance,
  onRunBacktest,
}) => {
  if (bots.length === 0) {
    return (
      <Box sx={{ textAlign: 'center', py: 8 }}>
        <Typography variant="h6" color="text.secondary" gutterBottom>
          No bots found
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Create your first trading bot to get started
        </Typography>
      </Box>
    );
  }

  return (
    <Grid container spacing={3}>
      {bots.map((bot) => (
        <Grid item xs={12} sm={6} md={4} key={bot.id}>
          <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <CardContent sx={{ flexGrow: 1 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                <Typography variant="h6" component="h2">
                  {bot.name}
                </Typography>
                <Chip
                  label={bot.isActive ? 'Running' : 'Stopped'}
                  color={bot.isActive ? 'success' : 'default'}
                  size="small"
                />
              </Box>
              
              <Typography color="text.secondary" gutterBottom>
                {bot.symbol} • {bot.timeframe}
              </Typography>
              
              <Typography variant="body2" sx={{ mb: 2 }}>
                Strategy: {bot.strategy}
              </Typography>
              
              <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
                <Chip label={bot.symbol} size="small" variant="outlined" />
                <Chip label={bot.timeframe} size="small" variant="outlined" />
                <Chip label={bot.strategy} size="small" variant="outlined" />
              </Box>
            </CardContent>
            
            <Box sx={{ p: 2, pt: 0, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              <Tooltip title="Edit Bot">
                <IconButton size="small" onClick={() => onEdit(bot)}>
                  <EditIcon />
                </IconButton>
              </Tooltip>
              
              <Tooltip title="Delete Bot">
                <IconButton size="small" onClick={() => onDelete(bot)}>
                  <DeleteIcon />
                </IconButton>
              </Tooltip>
              
              <Tooltip title="View Performance">
                <IconButton size="small" onClick={() => onViewPerformance(bot)}>
                  <VisibilityIcon />
                </IconButton>
              </Tooltip>
              
              <Tooltip title="Run Backtest">
                <IconButton size="small" onClick={() => onRunBacktest(bot)}>
                  <AssessmentIcon />
                </IconButton>
              </Tooltip>
              
              <Box sx={{ flexGrow: 1 }} />
              
              {bot.isActive ? (
                <Tooltip title="Stop Bot">
                  <IconButton 
                    size="small" 
                    color="error"
                    onClick={() => onStop(bot)}
                  >
                    <StopIcon />
                  </IconButton>
                </Tooltip>
              ) : (
                <Tooltip title="Start Bot">
                  <IconButton 
                    size="small" 
                    color="success"
                    onClick={() => onStart(bot)}
                  >
                    <PlayIcon />
                  </IconButton>
                </Tooltip>
              )}
            </Box>
          </Card>
        </Grid>
      ))}
    </Grid>
  );
};

export default EnhancedBotDashboard;
