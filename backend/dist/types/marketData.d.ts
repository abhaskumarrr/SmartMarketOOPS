/**
 * Market Data Types and Interfaces
 */
export interface OHLCV {
    timestamp: number;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
}
export interface MarketDataPoint extends OHLCV {
    symbol: string;
    exchange: string;
    timeframe: string;
}
export interface TechnicalIndicators {
    sma_20?: number;
    sma_50?: number;
    ema_12?: number;
    ema_26?: number;
    rsi?: number;
    macd?: number;
    macd_signal?: number;
    macd_histogram?: number;
    bollinger_upper?: number;
    bollinger_middle?: number;
    bollinger_lower?: number;
    volume_sma?: number;
}
export interface EnhancedMarketData extends MarketDataPoint {
    indicators: TechnicalIndicators;
}
export interface MarketDataRequest {
    symbol: string;
    timeframe: string;
    startDate: Date;
    endDate: Date;
    exchange?: string;
}
export interface MarketDataResponse {
    symbol: string;
    timeframe: string;
    data: MarketDataPoint[];
    count: number;
    startDate: Date;
    endDate: Date;
    source?: string;
}
export interface TradingSignal {
    id: string;
    timestamp: number;
    symbol: string;
    type: 'BUY' | 'SELL' | 'HOLD';
    price: number;
    quantity: number;
    confidence: number;
    strategy: string;
    reason: string;
    stopLoss?: number;
    takeProfit?: number;
    riskReward?: number;
}
export interface Position {
    symbol: string;
    side: 'LONG' | 'SHORT';
    size: number;
    entryPrice: number;
    entryTime: number;
    currentPrice: number;
    unrealizedPnl: number;
    leverage: number;
    stopLoss?: number;
    takeProfitLevels?: TakeProfitLevel[];
    originalSize?: number;
    partialExits?: PartialExit[];
}
export interface TakeProfitLevel {
    percentage: number;
    priceTarget: number;
    riskRewardRatio: number;
    trailingDistance?: number;
    executed?: boolean;
    executedAt?: number;
}
export interface PartialExit {
    percentage: number;
    price: number;
    timestamp: number;
    pnl: number;
    reason: string;
}
export interface Trade {
    id: string;
    symbol: string;
    side: 'LONG' | 'SHORT';
    entryPrice: number;
    exitPrice: number;
    quantity: number;
    entryTime: number;
    exitTime: number;
    pnl: number;
    pnlPercent: number;
    commission: number;
    strategy: string;
    reason: string;
    duration: number;
}
export interface PortfolioSnapshot {
    timestamp: number;
    totalValue: number;
    cash: number;
    positions: Position[];
    totalPnl: number;
    totalPnlPercent: number;
    drawdown: number;
    maxDrawdown: number;
    leverage: number;
}
export interface BacktestConfig {
    symbol: string;
    timeframe: string;
    startDate: Date;
    endDate: Date;
    initialCapital: number;
    leverage: number;
    riskPerTrade: number;
    commission: number;
    slippage: number;
    strategy: string;
    parameters: Record<string, any>;
}
export interface PerformanceMetrics {
    totalReturn: number;
    totalReturnPercent: number;
    annualizedReturn: number;
    sharpeRatio: number;
    sortinoRatio: number;
    maxDrawdown: number;
    maxDrawdownPercent: number;
    winRate: number;
    profitFactor: number;
    averageWin: number;
    averageLoss: number;
    averageWinPercent: number;
    averageLossPercent: number;
    totalTrades: number;
    winningTrades: number;
    losingTrades: number;
    largestWin: number;
    largestLoss: number;
    averageTradeDuration: number;
    volatility: number;
    calmarRatio: number;
    recoveryFactor: number;
    payoffRatio: number;
    expectancy: number;
}
export interface BacktestResult {
    config: BacktestConfig;
    performance: PerformanceMetrics;
    trades: Trade[];
    portfolioHistory: PortfolioSnapshot[];
    finalPortfolio: PortfolioSnapshot;
    startTime: number;
    endTime: number;
    duration: number;
    dataPoints: number;
}
export interface MarketDataProvider {
    name: string;
    fetchHistoricalData(request: MarketDataRequest): Promise<MarketDataResponse>;
    isAvailable(): boolean;
}
export interface TechnicalAnalysis {
    calculateSMA(prices: number[], period: number): number[];
    calculateEMA(prices: number[], period: number): number[];
    calculateRSI(prices: number[], period: number): number[];
    calculateMACD(prices: number[], fastPeriod: number, slowPeriod: number, signalPeriod: number): {
        macd: number[];
        signal: number[];
        histogram: number[];
    };
    calculateBollingerBands(prices: number[], period: number, stdDev: number): {
        upper: number[];
        middle: number[];
        lower: number[];
    };
}
export interface TradingStrategy {
    name: string;
    parameters: Record<string, any>;
    generateSignal(data: EnhancedMarketData[], currentIndex: number): TradingSignal | null;
    initialize(config: BacktestConfig): void;
    getDescription(): string;
}
export interface MarketDataEvent {
    type: 'MARKET_DATA_RECEIVED';
    data: MarketDataPoint;
}
export interface TradingSignalEvent {
    type: 'TRADING_SIGNAL_GENERATED';
    data: TradingSignal;
}
export interface TradeExecutedEvent {
    type: 'TRADE_EXECUTED';
    data: Trade;
}
export interface PortfolioUpdateEvent {
    type: 'PORTFOLIO_UPDATED';
    data: PortfolioSnapshot;
}
export type TimeFrame = '1m' | '5m' | '15m' | '1h' | '4h' | '1d' | '1w';
export type Exchange = 'binance' | 'coinbase' | 'kraken' | 'mock';
export declare const TIMEFRAMES: Record<TimeFrame, number>;
export declare const SYMBOLS: {
    readonly BTCUSD: "BTCUSD";
    readonly ETHUSD: "ETHUSD";
    readonly ADAUSD: "ADAUSD";
};
export type Symbol = typeof SYMBOLS[keyof typeof SYMBOLS];
