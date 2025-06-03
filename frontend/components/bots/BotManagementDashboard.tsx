/**
 * Bot Management Dashboard Component
 * Main dashboard for managing trading bots
 */

'use client';

import React, { useState, useCallback } from 'react';
import {
  Box,
  Typography,
  Button,
  Card,
  CardContent,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Fab,
  Tooltip,
  Snackbar,
} from '@mui/material';
import {
  Add as AddIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import { Bot } from '../../types/bot';
import { botService } from '../../lib/services/botService';
import BotList from './BotList';
import BotConfigurationForm from './BotConfigurationForm';
import RiskSettingsForm from './RiskSettingsForm';
import BotStatusMonitor from './BotStatusMonitor';

export const BotManagementDashboard: React.FC = () => {
  const [selectedBot, setSelectedBot] = useState<Bot | null>(null);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [showEditForm, setShowEditForm] = useState(false);
  const [showRiskForm, setShowRiskForm] = useState(false);
  const [showStatusMonitor, setShowStatusMonitor] = useState(false);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [snackbar, setSnackbar] = useState<{
    open: boolean;
    message: string;
    severity: 'success' | 'error' | 'warning' | 'info';
  }>({
    open: false,
    message: '',
    severity: 'success',
  });

  const showSnackbar = useCallback((message: string, severity: 'success' | 'error' | 'warning' | 'info' = 'success') => {
    setSnackbar({ open: true, message, severity });
  }, []);

  const handleCloseSnackbar = () => {
    setSnackbar(prev => ({ ...prev, open: false }));
  };

  const handleRefresh = () => {
    setRefreshTrigger(prev => prev + 1);
    showSnackbar('Bot list refreshed', 'info');
  };

  const handleCreateBot = () => {
    setSelectedBot(null);
    setShowCreateForm(true);
  };

  const handleEditBot = (bot: Bot) => {
    setSelectedBot(bot);
    setShowEditForm(true);
  };

  const handleConfigureRisk = (bot: Bot) => {
    setSelectedBot(bot);
    setShowRiskForm(true);
  };

  const handleViewStatus = (bot: Bot) => {
    setSelectedBot(bot);
    setShowStatusMonitor(true);
  };

  const handleDeleteBot = (bot: Bot) => {
    setSelectedBot(bot);
    setShowDeleteDialog(true);
  };

  const confirmDeleteBot = async () => {
    if (!selectedBot) return;

    try {
      const response = await botService.deleteBot(selectedBot.id);
      if (response.success) {
        showSnackbar(`Bot "${selectedBot.name}" deleted successfully`);
        setRefreshTrigger(prev => prev + 1);
      } else {
        showSnackbar(response.message || 'Failed to delete bot', 'error');
      }
    } catch (error) {
      console.error('Error deleting bot:', error);
      showSnackbar(error instanceof Error ? error.message : 'Failed to delete bot', 'error');
    } finally {
      setShowDeleteDialog(false);
      setSelectedBot(null);
    }
  };

  const handleFormSuccess = () => {
    setRefreshTrigger(prev => prev + 1);
    showSnackbar(
      showCreateForm ? 'Bot created successfully' : 'Bot updated successfully'
    );
  };

  const handleRiskSettingsSuccess = () => {
    showSnackbar('Risk settings updated successfully');
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4" component="h1" gutterBottom>
            Bot Management
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Create, configure, and monitor your automated trading bots
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={handleRefresh}
          >
            Refresh
          </Button>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={handleCreateBot}
          >
            Create Bot
          </Button>
        </Box>
      </Box>

      {/* Quick Stats */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Quick Overview
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Manage your trading bots from this dashboard. Create new bots, configure risk settings, 
            monitor performance, and control bot operations all in one place.
          </Typography>
        </CardContent>
      </Card>

      {/* Bot List */}
      <BotList
        onEditBot={handleEditBot}
        onDeleteBot={handleDeleteBot}
        onConfigureRisk={handleConfigureRisk}
        refreshTrigger={refreshTrigger}
      />

      {/* Floating Action Button for Quick Actions */}
      <Tooltip title="Create New Bot">
        <Fab
          color="primary"
          aria-label="create bot"
          onClick={handleCreateBot}
          sx={{
            position: 'fixed',
            bottom: 16,
            right: 16,
          }}
        >
          <AddIcon />
        </Fab>
      </Tooltip>

      {/* Create Bot Form */}
      <BotConfigurationForm
        open={showCreateForm}
        onClose={() => setShowCreateForm(false)}
        onSuccess={handleFormSuccess}
      />

      {/* Edit Bot Form */}
      <BotConfigurationForm
        open={showEditForm}
        onClose={() => setShowEditForm(false)}
        onSuccess={handleFormSuccess}
        bot={selectedBot}
      />

      {/* Risk Settings Form */}
      {selectedBot && (
        <RiskSettingsForm
          open={showRiskForm}
          onClose={() => setShowRiskForm(false)}
          onSuccess={handleRiskSettingsSuccess}
          bot={selectedBot}
        />
      )}

      {/* Bot Status Monitor */}
      <Dialog
        open={showStatusMonitor}
        onClose={() => setShowStatusMonitor(false)}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>
          Bot Status Monitor
        </DialogTitle>
        <DialogContent>
          {selectedBot && (
            <BotStatusMonitor
              bot={selectedBot}
              autoRefresh={true}
            />
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowStatusMonitor(false)}>
            Close
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={showDeleteDialog}
        onClose={() => setShowDeleteDialog(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          Confirm Bot Deletion
        </DialogTitle>
        <DialogContent>
          <Alert severity="warning" sx={{ mb: 2 }}>
            This action cannot be undone. All bot data, including trading history and configurations, will be permanently deleted.
          </Alert>
          <Typography>
            Are you sure you want to delete the bot "{selectedBot?.name}"?
          </Typography>
          {selectedBot?.isActive && (
            <Alert severity="error" sx={{ mt: 2 }}>
              Warning: This bot is currently active. Deleting it will stop all trading activities immediately.
            </Alert>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowDeleteDialog(false)}>
            Cancel
          </Button>
          <Button
            onClick={confirmDeleteBot}
            color="error"
            variant="contained"
          >
            Delete Bot
          </Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
      >
        <Alert
          onClose={handleCloseSnackbar}
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default BotManagementDashboard;
