/**
 * Delta Exchange Trading API Client
 * Provides methods to interact with our Delta Exchange trading bot system
 */

import axios from 'axios';

export interface BotConfig {
  id: string;
  name: string;
  symbol: string;
  strategy: string;
  capital: number;
  leverage: number;
  riskPerTrade: number;
  maxPositions: number;
  stopLoss: number;
  takeProfit: number;
  enabled: boolean;
  testnet: boolean;
}

export interface BotStatus {
  id: string;
  name: string;
  symbol: string;
  status: 'running' | 'stopped' | 'paused' | 'error';
  totalTrades: number;
  totalPnL: number;
  winRate: string;
  lastActivity: string;
  config: BotConfig;
}

export interface BotManagerStatus {
  totalBots: number;
  runningBots: number;
  pausedBots: number;
  stoppedBots: number;
  errorBots: number;
  totalTrades: number;
  totalPnL: number;
  exchange: string;
  environment: string;
  timestamp: number;
}

export interface DeltaProduct {
  id: number;
  symbol: string;
  description: string;
  contract_type: string;
  state: string;
  underlying_asset: { symbol: string };
  quoting_asset: { symbol: string };
  settling_asset: { symbol: string };
}

class DeltaTradingApi {
  private baseUrl: string;

  constructor() {
    // Use environment variable for backend URL, fallback to relative path
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || '';
    this.baseUrl = backendUrl ? `${backendUrl}/api/delta-trading` : '/api/delta-trading';

    // Debug logging
    console.log('DeltaTradingApi initialized with baseUrl:', this.baseUrl);
    console.log('NEXT_PUBLIC_BACKEND_URL:', process.env.NEXT_PUBLIC_BACKEND_URL);
  }
  
  /**
   * Get service health status (no auth required)
   */
  async getHealth() {
    try {
      const response = await axios.get(`${this.baseUrl}/health`);
      return response.data;
    } catch (error) {
      this.handleError(error, 'Failed to get health status');
    }
  }
  
  /**
   * Test Delta Exchange connection (no auth required)
   */
  async testConnection() {
    try {
      const response = await axios.get(`${this.baseUrl}/test-connection`);
      return response.data;
    } catch (error) {
      this.handleError(error, 'Failed to test connection');
    }
  }
  
  /**
   * Get Delta Exchange trading service status
   */
  async getStatus(): Promise<BotManagerStatus> {
    try {
      const response = await axios.get(`${this.baseUrl}/status`);
      return response.data.data;
    } catch (error) {
      this.handleError(error, 'Failed to get trading status');
    }
  }
  
  /**
   * Get all trading bots
   */
  async getBots(): Promise<BotStatus[]> {
    try {
      const response = await axios.get(`${this.baseUrl}/bots`);
      return response.data.data;
    } catch (error) {
      this.handleError(error, 'Failed to get bots');
    }
  }
  
  /**
   * Get specific bot status
   */
  async getBot(botId: string): Promise<BotStatus> {
    try {
      const response = await axios.get(`${this.baseUrl}/bots/${botId}`);
      return response.data.data;
    } catch (error) {
      this.handleError(error, `Failed to get bot ${botId}`);
    }
  }
  
  /**
   * Create a new trading bot
   */
  async createBot(config: Partial<BotConfig>): Promise<{ botId: string; config: BotConfig }> {
    try {
      const response = await axios.post(`${this.baseUrl}/bots`, config);
      return response.data.data;
    } catch (error) {
      this.handleError(error, 'Failed to create bot');
    }
  }
  
  /**
   * Start a trading bot
   */
  async startBot(botId: string): Promise<void> {
    try {
      await axios.post(`${this.baseUrl}/bots/${botId}/start`);
    } catch (error) {
      this.handleError(error, `Failed to start bot ${botId}`);
    }
  }
  
  /**
   * Stop a trading bot
   */
  async stopBot(botId: string): Promise<void> {
    try {
      await axios.post(`${this.baseUrl}/bots/${botId}/stop`);
    } catch (error) {
      this.handleError(error, `Failed to stop bot ${botId}`);
    }
  }
  
  /**
   * Pause a trading bot
   */
  async pauseBot(botId: string): Promise<void> {
    try {
      await axios.post(`${this.baseUrl}/bots/${botId}/pause`);
    } catch (error) {
      this.handleError(error, `Failed to pause bot ${botId}`);
    }
  }
  
  /**
   * Resume a trading bot
   */
  async resumeBot(botId: string): Promise<void> {
    try {
      await axios.post(`${this.baseUrl}/bots/${botId}/resume`);
    } catch (error) {
      this.handleError(error, `Failed to resume bot ${botId}`);
    }
  }
  
  /**
   * Remove a trading bot
   */
  async removeBot(botId: string): Promise<void> {
    try {
      await axios.delete(`${this.baseUrl}/bots/${botId}`);
    } catch (error) {
      this.handleError(error, `Failed to remove bot ${botId}`);
    }
  }
  
  /**
   * Update bot configuration
   */
  async updateBotConfig(botId: string, config: Partial<BotConfig>): Promise<void> {
    try {
      await axios.put(`${this.baseUrl}/bots/${botId}/config`, config);
    } catch (error) {
      this.handleError(error, `Failed to update bot ${botId} config`);
    }
  }
  
  /**
   * Get bot performance metrics
   */
  async getBotPerformance(botId: string): Promise<any> {
    try {
      const response = await axios.get(`${this.baseUrl}/bots/${botId}/performance`);
      return response.data.data;
    } catch (error) {
      this.handleError(error, `Failed to get bot ${botId} performance`);
    }
  }
  
  /**
   * Get overall performance summary
   */
  async getOverallPerformance(): Promise<any> {
    try {
      const response = await axios.get(`${this.baseUrl}/performance`);
      return response.data.data;
    } catch (error) {
      this.handleError(error, 'Failed to get overall performance');
    }
  }
  
  /**
   * Get available Delta Exchange products
   */
  async getProducts(): Promise<DeltaProduct[]> {
    try {
      const response = await axios.get(`${this.baseUrl}/products`);
      return response.data.data;
    } catch (error) {
      this.handleError(error, 'Failed to get products');
    }
  }
  
  /**
   * Emergency stop all bots
   */
  async emergencyStopAll(): Promise<void> {
    try {
      await axios.post(`${this.baseUrl}/emergency-stop`);
    } catch (error) {
      this.handleError(error, 'Failed to execute emergency stop');
    }
  }
  
  /**
   * Error handling helper
   */
  private handleError(error: any, defaultMessage: string): never {
    const errorMsg = error.response?.data?.message || error.message || defaultMessage;
    console.error(errorMsg, error);
    throw new Error(errorMsg);
  }
}

// Create singleton instance
const deltaTradingApi = new DeltaTradingApi();

export default deltaTradingApi;
