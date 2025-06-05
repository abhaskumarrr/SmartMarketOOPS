/**
 * Delta Exchange Trading Dashboard
 * Professional trading bot management interface with real-time updates
 */

'use client';

import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Button,
  Chip,
  IconButton,
  Alert,
  CircularProgress,
  useTheme,
} from '@mui/material';
import {
  Add as AddIcon,
  Refresh as RefreshIcon,
  Emergency as EmergencyIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  SmartToy as BotIcon,
} from '@mui/icons-material';
import deltaTradingApi, { BotManagerStatus, BotStatus } from '../../lib/api/deltaTradingApi';
import { DeltaTradingStats } from './DeltaTradingStats';
import { DeltaBotList } from './DeltaBotList';
import { CreateBotDialog } from './CreateBotDialog';

export function DeltaTradingDashboard() {
  const theme = useTheme();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<BotManagerStatus | null>(null);
  const [bots, setBots] = useState<BotStatus[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'connecting'>('connecting');
  const [createBotOpen, setCreateBotOpen] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  // Initialize dashboard
  useEffect(() => {
    initializeDashboard();
    
    // Set up auto-refresh
    const interval = setInterval(() => {
      refreshData();
    }, 5000); // Refresh every 5 seconds

    return () => clearInterval(interval);
  }, []);

  const initializeDashboard = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Test connection first
      await deltaTradingApi.testConnection();
      setConnectionStatus('connected');
      
      // Load initial data
      await loadData();
      
    } catch (error) {
      console.error('Failed to initialize dashboard:', error);
      setError(error instanceof Error ? error.message : 'Failed to initialize dashboard');
      setConnectionStatus('disconnected');
    } finally {
      setLoading(false);
    }
  };

  const loadData = async () => {
    try {
      const [statusData, botsData] = await Promise.all([
        deltaTradingApi.getStatus(),
        deltaTradingApi.getBots()
      ]);
      
      setStatus(statusData);
      setBots(botsData);
    } catch (error) {
      console.error('Failed to load data:', error);
      throw error;
    }
  };

  const refreshData = async () => {
    try {
      setRefreshing(true);
      await loadData();
    } catch (error) {
      console.error('Failed to refresh data:', error);
      setConnectionStatus('disconnected');
    } finally {
      setRefreshing(false);
    }
  };

  const handleEmergencyStop = async () => {
    try {
      await deltaTradingApi.emergencyStopAll();
      await refreshData();
    } catch (error) {
      console.error('Emergency stop failed:', error);
      setError(error instanceof Error ? error.message : 'Emergency stop failed');
    }
  };

  const handleBotAction = async (action: string, botId: string) => {
    try {
      switch (action) {
        case 'start':
          await deltaTradingApi.startBot(botId);
          break;
        case 'stop':
          await deltaTradingApi.stopBot(botId);
          break;
        case 'pause':
          await deltaTradingApi.pauseBot(botId);
          break;
        case 'resume':
          await deltaTradingApi.resumeBot(botId);
          break;
        case 'remove':
          await deltaTradingApi.removeBot(botId);
          break;
      }
      await refreshData();
    } catch (error) {
      console.error(`Bot action ${action} failed:`, error);
      setError(error instanceof Error ? error.message : `Bot action ${action} failed`);
    }
  };

  const handleCreateBot = async (config: any) => {
    try {
      await deltaTradingApi.createBot(config);
      await refreshData();
      setCreateBotOpen(false);
    } catch (error) {
      console.error('Failed to create bot:', error);
      setError(error instanceof Error ? error.message : 'Failed to create bot');
    }
  };

  if (loading) {
    return (
      <Box 
        sx={{ 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center', 
          height: '100vh',
          flexDirection: 'column',
          gap: 2
        }}
      >
        <CircularProgress size={60} />
        <Typography variant="h6" color="text.secondary">
          Initializing Delta Exchange Trading...
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3, height: '100%', overflow: 'auto' }}>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'between', alignItems: 'center', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <BotIcon sx={{ fontSize: 32, color: 'primary.main' }} />
            <Box>
              <Typography variant="h4" sx={{ fontWeight: 700, color: 'text.primary' }}>
                Delta Exchange Trading
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
                <Chip 
                  label="India Testnet" 
                  color="success" 
                  size="small" 
                  variant="outlined"
                />
                <Chip 
                  label={connectionStatus === 'connected' ? 'Connected' : 'Disconnected'} 
                  color={connectionStatus === 'connected' ? 'success' : 'error'} 
                  size="small"
                />
              </Box>
            </Box>
          </Box>
          
          <Box sx={{ display: 'flex', gap: 1 }}>
            <IconButton 
              onClick={refreshData} 
              disabled={refreshing}
              sx={{ bgcolor: 'background.paper' }}
            >
              <RefreshIcon />
            </IconButton>
            <Button
              variant="contained"
              startIcon={<AddIcon />}
              onClick={() => setCreateBotOpen(true)}
              sx={{ textTransform: 'none' }}
            >
              Create Bot
            </Button>
            <Button
              variant="outlined"
              color="error"
              startIcon={<EmergencyIcon />}
              onClick={handleEmergencyStop}
              sx={{ textTransform: 'none' }}
            >
              Emergency Stop
            </Button>
          </Box>
        </Box>

        {error && (
          <Alert 
            severity="error" 
            onClose={() => setError(null)}
            sx={{ mb: 2 }}
          >
            {error}
          </Alert>
        )}
      </Box>

      {/* Stats Overview */}
      {status && (
        <DeltaTradingStats 
          status={status} 
          sx={{ mb: 3 }}
        />
      )}

      {/* Bot Management */}
      <DeltaBotList 
        bots={bots}
        onBotAction={handleBotAction}
        refreshing={refreshing}
      />

      {/* Create Bot Dialog */}
      <CreateBotDialog
        open={createBotOpen}
        onClose={() => setCreateBotOpen(false)}
        onSubmit={handleCreateBot}
      />
    </Box>
  );
}
