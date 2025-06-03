/**
 * Bot Service
 * API service functions for bot management
 */

import { Bot, BotCreateRequest, BotUpdateRequest, BotStatus, RiskSettings, BotHealthData } from '../../types/bot';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';

class BotService {
  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const token = localStorage.getItem('token');
    
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...(token && { Authorization: `Bearer ${token}` }),
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  // Bot CRUD operations
  async createBot(botData: BotCreateRequest): Promise<{ success: boolean; data: Bot; message: string }> {
    return this.request<{ success: boolean; data: Bot; message: string }>('/api/bots', {
      method: 'POST',
      body: JSON.stringify(botData),
    });
  }

  async getBots(): Promise<{ success: boolean; data: Bot[]; message: string }> {
    return this.request<{ success: boolean; data: Bot[]; message: string }>('/api/bots');
  }

  async getBot(botId: string): Promise<{ success: boolean; data: Bot; message: string }> {
    return this.request<{ success: boolean; data: Bot; message: string }>(`/api/bots/${botId}`);
  }

  async updateBot(botId: string, botData: BotUpdateRequest): Promise<{ success: boolean; data: Bot; message: string }> {
    return this.request<{ success: boolean; data: Bot; message: string }>(`/api/bots/${botId}`, {
      method: 'PUT',
      body: JSON.stringify(botData),
    });
  }

  async deleteBot(botId: string): Promise<{ success: boolean; message: string }> {
    return this.request<{ success: boolean; message: string }>(`/api/bots/${botId}`, {
      method: 'DELETE',
    });
  }

  // Bot control operations
  async startBot(botId: string): Promise<{ success: boolean; message: string }> {
    return this.request<{ success: boolean; message: string }>(`/api/bots/${botId}/start`, {
      method: 'POST',
    });
  }

  async stopBot(botId: string): Promise<{ success: boolean; message: string }> {
    return this.request<{ success: boolean; message: string }>(`/api/bots/${botId}/stop`, {
      method: 'POST',
    });
  }

  async pauseBot(botId: string, duration?: number): Promise<{ success: boolean; message: string }> {
    return this.request<{ success: boolean; message: string }>(`/api/bots/${botId}/pause`, {
      method: 'POST',
      body: JSON.stringify({ duration }),
    });
  }

  async resumeBot(botId: string): Promise<{ success: boolean; message: string }> {
    return this.request<{ success: boolean; message: string }>(`/api/bots/${botId}/resume`, {
      method: 'POST',
    });
  }

  // Bot status and monitoring
  async getBotStatus(botId: string): Promise<{ success: boolean; data: BotStatus; message: string }> {
    return this.request<{ success: boolean; data: BotStatus; message: string }>(`/api/bots/${botId}/status`);
  }

  async updateBotHealth(botId: string, healthData: BotHealthData): Promise<{ success: boolean; message: string }> {
    return this.request<{ success: boolean; message: string }>(`/api/bots/${botId}/health`, {
      method: 'POST',
      body: JSON.stringify(healthData),
    });
  }

  // Risk settings operations
  async getBotRiskSettings(botId: string): Promise<{ success: boolean; data: RiskSettings; message: string }> {
    return this.request<{ success: boolean; data: RiskSettings; message: string }>(`/api/bots/${botId}/risk-settings`);
  }

  async updateBotRiskSettings(
    botId: string, 
    riskSettings: Partial<RiskSettings>
  ): Promise<{ success: boolean; data: RiskSettings; message: string }> {
    return this.request<{ success: boolean; data: RiskSettings; message: string }>(`/api/bots/${botId}/risk-settings`, {
      method: 'PUT',
      body: JSON.stringify(riskSettings),
    });
  }

  // Utility methods
  async validateBotConfiguration(botData: BotCreateRequest): Promise<{ valid: boolean; errors: string[] }> {
    const errors: string[] = [];

    if (!botData.name || botData.name.trim().length === 0) {
      errors.push('Bot name is required');
    }

    if (!botData.symbol) {
      errors.push('Trading symbol is required');
    }

    if (!botData.strategy) {
      errors.push('Trading strategy is required');
    }

    if (!botData.timeframe) {
      errors.push('Timeframe is required');
    }

    return {
      valid: errors.length === 0,
      errors
    };
  }

  // WebSocket connection for real-time updates
  connectToBot(botId: string, onUpdate: (status: BotStatus) => void): WebSocket | null {
    try {
      const token = localStorage.getItem('token');
      const wsUrl = `${API_BASE_URL.replace('http', 'ws')}/ws/bots/${botId}?token=${token}`;
      
      const ws = new WebSocket(wsUrl);
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'bot_status_update') {
            onUpdate(data.payload);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      return ws;
    } catch (error) {
      console.error('Error connecting to bot WebSocket:', error);
      return null;
    }
  }
}

export const botService = new BotService();
export default botService;
