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

export interface MarketTick {
  symbol: string;
  price: number;
  changePercentage24h: number;
  volume: number;
  timestamp: number;
}

export interface TradeSignal {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  price: number;
  size: number;
  timestamp: number;
  source: 'algorithm' | 'manual' | 'bot';
  strategy?: string;
  confidence?: number;
}

export interface PortfolioUpdate {
  totalBalance: number;
  availableBalance: number;
  totalPnL: number;
  totalPnLPercentage: number;
  positions: {
    symbol: string;
    side: 'long' | 'short';
    size: number;
    entryPrice: number;
    currentPrice: number;
    pnl: number;
    pnlPercentage: number;
  }[];
  timestamp: number;
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
