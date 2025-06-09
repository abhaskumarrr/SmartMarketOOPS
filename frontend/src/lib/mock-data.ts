/**
 * Mock Data Provider
 * Generates realistic mock data for testing and development
 */
import { MarketTick, TradeSignal, PortfolioUpdate } from './realtime-data';

// Constants
const CRYPTO_SYMBOLS = ['BTCUSD', 'ETHUSD', 'SOLUSD', 'BNBUSD', 'DOTUSD', 'ADAUSD', 'LINKUSD', 'XRPUSD'];
const BASE_PRICES = {
  'BTCUSD': 48250,
  'ETHUSD': 2870,
  'SOLUSD': 106,
  'BNBUSD': 570,
  'DOTUSD': 7.8,
  'ADAUSD': 0.45,
  'LINKUSD': 14.20,
  'XRPUSD': 0.52,
};

// Helper functions
export function randomBetween(min: number, max: number): number {
  return Math.random() * (max - min) + min;
}

export function randomIntBetween(min: number, max: number): number {
  return Math.floor(randomBetween(min, max));
}

export function randomChoice<T>(array: T[]): T {
  return array[Math.floor(Math.random() * array.length)];
}

// Mock data generators
export function generateMockMarketTick(symbol: string, prevPrice?: number): MarketTick {
  const basePrice = BASE_PRICES[symbol as keyof typeof BASE_PRICES] || 100;
  const price = prevPrice 
    ? prevPrice * (1 + (Math.random() - 0.5) * 0.002) // Small change from previous
    : basePrice * (1 + (Math.random() - 0.5) * 0.1); // Initial ±5% from base
  
  const change24h = randomBetween(-5, 5);
  
  return {
    symbol,
    price: +price.toFixed(2),
    volume: randomIntBetween(1000000, 500000000),
    timestamp: Date.now(),
    change24h: +change24h.toFixed(2),
    high24h: +(price * (1 + Math.random() * 0.03)).toFixed(2),
    low24h: +(price * (1 - Math.random() * 0.03)).toFixed(2),
  };
}

export function generateMockMarketData(): Record<string, MarketTick> {
  const result: Record<string, MarketTick> = {};
  CRYPTO_SYMBOLS.forEach(symbol => {
    result[symbol] = generateMockMarketTick(symbol);
  });
  return result;
}

export function generateMockTradeSignal(): TradeSignal {
  const symbol = randomChoice(CRYPTO_SYMBOLS);
  const type = Math.random() > 0.5 ? 'buy' : 'sell';
  const confidence = randomBetween(0.6, 0.95);
  const price = BASE_PRICES[symbol as keyof typeof BASE_PRICES] || 100;
  
  return {
    id: `signal-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
    symbol,
    type,
    price: price * (1 + (Math.random() - 0.5) * 0.05),
    confidence,
    timestamp: Date.now(),
    indicators: [
      {
        name: 'RSI',
        value: type === 'buy' ? randomBetween(30, 40) : randomBetween(70, 80),
        signal: type,
      },
      {
        name: 'MACD',
        value: type === 'buy' ? randomBetween(0.1, 0.5) : randomBetween(-0.5, -0.1),
        signal: type,
      },
      {
        name: 'Moving Average',
        value: 0,
        signal: type,
      },
    ],
  };
}

export function generateMockPortfolioUpdate(prevUpdate?: PortfolioUpdate): PortfolioUpdate {
  // Start with a base portfolio or use the previous one
  const totalBalance = prevUpdate?.totalBalance || 100000;
  const dayStartBalance = totalBalance * 0.99; // Assume slightly lower at day start
  
  // Generate some random daily movement
  const dayPnLPercentage = randomBetween(-2, 3);
  const dayPnL = totalBalance * (dayPnLPercentage / 100);
  
  // Calculate total P&L (assume starting with $80k)
  const initialInvestment = 80000;
  const totalPnLPercentage = ((totalBalance - initialInvestment) / initialInvestment) * 100;
  const totalPnL = totalBalance - initialInvestment;
  
  // Generate positions
  const positions = [];
  const numPositions = randomIntBetween(2, 5);
  let totalPositionValue = 0;
  
  for (let i = 0; i < numPositions; i++) {
    const symbol = CRYPTO_SYMBOLS[i];
    const basePrice = BASE_PRICES[symbol as keyof typeof BASE_PRICES] || 100;
    const entryPrice = basePrice * (1 - randomBetween(0.01, 0.1));
    const currentPrice = basePrice;
    const size = randomBetween(0.5, 5) * (basePrice < 10 ? 100 : basePrice < 100 ? 10 : 1);
    const positionValue = size * currentPrice;
    const pnl = size * (currentPrice - entryPrice);
    const pnlPercentage = (pnl / (size * entryPrice)) * 100;
    
    totalPositionValue += positionValue;
    
    positions.push({
      symbol,
      size: +size.toFixed(4),
      entryPrice: +entryPrice.toFixed(2),
      currentPrice: +currentPrice.toFixed(2),
      pnl: +pnl.toFixed(2),
      pnlPercentage: +pnlPercentage.toFixed(2)
    });
  }
  
  // Available balance is what's not in positions
  const availableBalance = totalBalance - totalPositionValue;
  
  return {
    totalBalance,
    availableBalance: +availableBalance.toFixed(2),
    totalPnL: +totalPnL.toFixed(2),
    totalPnLPercentage: +totalPnLPercentage.toFixed(2),
    dayPnL: +dayPnL.toFixed(2),
    dayPnLPercentage: +dayPnLPercentage.toFixed(2),
    positions,
    timestamp: Date.now(),
  };
}

// Set up mock data for testing
let mockMarketData: Record<string, MarketTick> = {};
let mockPortfolio: PortfolioUpdate | null = null;
let mockTradeSignals: TradeSignal[] = [];

// Initialize data
export function initializeMockData() {
  mockMarketData = generateMockMarketData();
  mockPortfolio = generateMockPortfolioUpdate();
  mockTradeSignals = [];
  
  // Generate a few initial signals
  for (let i = 0; i < 3; i++) {
    mockTradeSignals.push(generateMockTradeSignal());
  }
  
  return {
    mockMarketData,
    mockPortfolio,
    mockTradeSignals,
  };
}

// Update mock data with realistic changes
export function updateMockData() {
  // Update market data with small changes
  for (const symbol of CRYPTO_SYMBOLS) {
    if (mockMarketData[symbol]) {
      mockMarketData[symbol] = generateMockMarketTick(symbol, mockMarketData[symbol].price);
    }
  }
  
  // Update portfolio based on market changes
  if (mockPortfolio) {
    mockPortfolio = generateMockPortfolioUpdate(mockPortfolio);
  }
  
  // 10% chance to generate a new trade signal
  if (Math.random() < 0.10) {
    const signal = generateMockTradeSignal();
    mockTradeSignals.unshift(signal);
    
    // Keep only the last 20 signals
    if (mockTradeSignals.length > 20) {
      mockTradeSignals = mockTradeSignals.slice(0, 20);
    }
  }
  
  return {
    mockMarketData,
    mockPortfolio,
    mockTradeSignals,
  };
}

// Get current mock data
export function getMockData() {
  return {
    marketData: mockMarketData,
    portfolio: mockPortfolio,
    tradeSignals: mockTradeSignals,
  };
} 