export interface Position {
  id: string;
  symbol: string;
  side: 'long' | 'short';
  size: number;
  entryPrice: number;
  currentPrice: number;
  pnl: number;
  pnlPercentage: number;
  status: 'open' | 'closed' | 'pending';
  timestamp: string;
}

export interface Trade {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  size: number;
  price: number;
  timestamp: string;
  status: 'executed' | 'pending' | 'cancelled';
  pnl?: number;
}

export interface Portfolio {
  totalBalance: number;
  availableBalance: number;
  totalPnl: number;
  totalPnlPercentage: number;
  positions: Position[];
  dailyPnl: number;
  dailyPnlPercentage: number;
}

export interface MarketData {
  symbol: string;
  price: number;
  change24h: number;
  changePercentage24h: number;
  volume24h: number;
  high24h: number;
  low24h: number;
  timestamp: string;
}

export interface CandlestickData {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface AIModelPrediction {
  symbol: string;
  prediction: 'bullish' | 'bearish' | 'neutral';
  confidence: number;
  targetPrice: number;
  timeframe: string;
  timestamp: string;
}

export interface ModelPerformance {
  accuracy: number;
  totalPredictions: number;
  correctPredictions: number;
  profitableTrades: number;
  totalTrades: number;
  winRate: number;
  averageReturn: number;
}
