/**
 * Bot Management Dashboard Tests
 * Tests for the BotManagementDashboard component
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import BotManagementDashboard from '../../../components/bots/BotManagementDashboard';
import { botService } from '../../../lib/services/botService';

// Mock the bot service
jest.mock('../../../lib/services/botService', () => ({
  botService: {
    getBots: jest.fn(),
    createBot: jest.fn(),
    updateBot: jest.fn(),
    deleteBot: jest.fn(),
    startBot: jest.fn(),
    stopBot: jest.fn(),
  },
}));

// Mock the configuration wizard
jest.mock('../../../components/bots/BotConfigurationForm', () => {
  return function MockBotConfigurationForm({ open, onClose, onSuccess }: any) {
    return open ? (
      <div data-testid="bot-config-form">
        <button onClick={onSuccess}>Save Bot</button>
        <button onClick={onClose}>Cancel</button>
      </div>
    ) : null;
  };
});

const theme = createTheme();

const renderWithTheme = (component: React.ReactElement) => {
  return render(
    <ThemeProvider theme={theme}>
      {component}
    </ThemeProvider>
  );
};

const mockBots = [
  {
    id: 'bot-1',
    name: 'Test Bot 1',
    symbol: 'BTCUSD',
    strategy: 'ML_PREDICTION',
    timeframe: '1h',
    isActive: true,
    parameters: {},
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  },
  {
    id: 'bot-2',
    name: 'Test Bot 2',
    symbol: 'ETHUSD',
    strategy: 'TECHNICAL_ANALYSIS',
    timeframe: '4h',
    isActive: false,
    parameters: {},
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  },
];

describe('BotManagementDashboard', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (botService.getBots as jest.Mock).mockResolvedValue({
      success: true,
      data: mockBots,
    });
  });

  it('should render loading state initially', () => {
    renderWithTheme(<BotManagementDashboard />);
    expect(screen.getByText('Loading bots...')).toBeInTheDocument();
  });

  it('should render bots after loading', async () => {
    renderWithTheme(<BotManagementDashboard />);

    await waitFor(() => {
      expect(screen.getByText('Test Bot 1')).toBeInTheDocument();
      expect(screen.getByText('Test Bot 2')).toBeInTheDocument();
    });

    expect(screen.getByText('BTCUSD • 1h')).toBeInTheDocument();
    expect(screen.getByText('ETHUSD • 4h')).toBeInTheDocument();
  });

  it('should show running status for active bots', async () => {
    renderWithTheme(<BotManagementDashboard />);

    await waitFor(() => {
      const runningChips = screen.getAllByText('Running');
      expect(runningChips).toHaveLength(1);
      
      const stoppedChips = screen.getAllByText('Stopped');
      expect(stoppedChips).toHaveLength(1);
    });
  });

  it('should open configuration form when create button is clicked', async () => {
    renderWithTheme(<BotManagementDashboard />);

    await waitFor(() => {
      expect(screen.getByText('Test Bot 1')).toBeInTheDocument();
    });

    const createButton = screen.getByText('Create Bot');
    fireEvent.click(createButton);

    expect(screen.getByTestId('bot-config-form')).toBeInTheDocument();
  });

  it('should handle bot creation', async () => {
    (botService.createBot as jest.Mock).mockResolvedValue({
      success: true,
      data: { id: 'new-bot', name: 'New Bot' },
    });

    renderWithTheme(<BotManagementDashboard />);

    await waitFor(() => {
      expect(screen.getByText('Test Bot 1')).toBeInTheDocument();
    });

    // Open create form
    const createButton = screen.getByText('Create Bot');
    fireEvent.click(createButton);

    // Save bot
    const saveButton = screen.getByText('Save Bot');
    fireEvent.click(saveButton);

    await waitFor(() => {
      expect(botService.getBots).toHaveBeenCalledTimes(2); // Initial load + after create
    });
  });

  it('should handle bot editing', async () => {
    renderWithTheme(<BotManagementDashboard />);

    await waitFor(() => {
      expect(screen.getByText('Test Bot 1')).toBeInTheDocument();
    });

    // Click edit button for first bot
    const editButtons = screen.getAllByLabelText('Edit Bot');
    fireEvent.click(editButtons[0]);

    expect(screen.getByTestId('bot-config-form')).toBeInTheDocument();
  });

  it('should handle bot deletion', async () => {
    (botService.deleteBot as jest.Mock).mockResolvedValue({
      success: true,
    });

    renderWithTheme(<BotManagementDashboard />);

    await waitFor(() => {
      expect(screen.getByText('Test Bot 1')).toBeInTheDocument();
    });

    // Click delete button for first bot
    const deleteButtons = screen.getAllByLabelText('Delete Bot');
    fireEvent.click(deleteButtons[0]);

    // Confirm deletion
    const confirmButton = screen.getByText('Delete');
    fireEvent.click(confirmButton);

    await waitFor(() => {
      expect(botService.deleteBot).toHaveBeenCalledWith('bot-1');
      expect(botService.getBots).toHaveBeenCalledTimes(2); // Initial load + after delete
    });
  });

  it('should handle starting a bot', async () => {
    (botService.startBot as jest.Mock).mockResolvedValue({
      success: true,
    });

    renderWithTheme(<BotManagementDashboard />);

    await waitFor(() => {
      expect(screen.getByText('Test Bot 2')).toBeInTheDocument();
    });

    // Find and click start button for stopped bot
    const startButtons = screen.getAllByLabelText('Start Bot');
    fireEvent.click(startButtons[0]);

    await waitFor(() => {
      expect(botService.startBot).toHaveBeenCalledWith('bot-2');
      expect(botService.getBots).toHaveBeenCalledTimes(2); // Initial load + after start
    });
  });

  it('should handle stopping a bot', async () => {
    (botService.stopBot as jest.Mock).mockResolvedValue({
      success: true,
    });

    renderWithTheme(<BotManagementDashboard />);

    await waitFor(() => {
      expect(screen.getByText('Test Bot 1')).toBeInTheDocument();
    });

    // Find and click stop button for running bot
    const stopButtons = screen.getAllByLabelText('Stop Bot');
    fireEvent.click(stopButtons[0]);

    await waitFor(() => {
      expect(botService.stopBot).toHaveBeenCalledWith('bot-1');
      expect(botService.getBots).toHaveBeenCalledTimes(2); // Initial load + after stop
    });
  });

  it('should display error message when API fails', async () => {
    (botService.getBots as jest.Mock).mockResolvedValue({
      success: false,
      message: 'Failed to load bots',
    });

    renderWithTheme(<BotManagementDashboard />);

    await waitFor(() => {
      expect(screen.getByText('Failed to load bots')).toBeInTheDocument();
    });
  });

  it('should show empty state when no bots exist', async () => {
    (botService.getBots as jest.Mock).mockResolvedValue({
      success: true,
      data: [],
    });

    renderWithTheme(<BotManagementDashboard />);

    await waitFor(() => {
      expect(screen.getByText('No trading bots found')).toBeInTheDocument();
      expect(screen.getByText('Create your first trading bot to get started')).toBeInTheDocument();
    });
  });

  it('should handle network errors gracefully', async () => {
    (botService.getBots as jest.Mock).mockRejectedValue(new Error('Network error'));

    renderWithTheme(<BotManagementDashboard />);

    await waitFor(() => {
      expect(screen.getByText('Network error')).toBeInTheDocument();
    });
  });

  it('should display bot strategies correctly', async () => {
    renderWithTheme(<BotManagementDashboard />);

    await waitFor(() => {
      expect(screen.getByText('Strategy: ML_PREDICTION')).toBeInTheDocument();
      expect(screen.getByText('Strategy: TECHNICAL_ANALYSIS')).toBeInTheDocument();
    });
  });

  it('should show correct number of bots in header', async () => {
    renderWithTheme(<BotManagementDashboard />);

    await waitFor(() => {
      expect(screen.getByText('Trading Bots')).toBeInTheDocument();
      // Should show both bots
      expect(screen.getByText('Test Bot 1')).toBeInTheDocument();
      expect(screen.getByText('Test Bot 2')).toBeInTheDocument();
    });
  });

  it('should handle concurrent operations', async () => {
    (botService.startBot as jest.Mock).mockResolvedValue({ success: true });
    (botService.stopBot as jest.Mock).mockResolvedValue({ success: true });

    renderWithTheme(<BotManagementDashboard />);

    await waitFor(() => {
      expect(screen.getByText('Test Bot 1')).toBeInTheDocument();
    });

    // Try to start and stop bots simultaneously
    const startButtons = screen.getAllByLabelText('Start Bot');
    const stopButtons = screen.getAllByLabelText('Stop Bot');

    fireEvent.click(startButtons[0]);
    fireEvent.click(stopButtons[0]);

    await waitFor(() => {
      expect(botService.startBot).toHaveBeenCalled();
      expect(botService.stopBot).toHaveBeenCalled();
    });
  });
});
