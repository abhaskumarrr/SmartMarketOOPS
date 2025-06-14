import { 
  DeltaExchangeMarket, 
  DeltaExchangePosition, 
  DeltaExchangeOrder, 
  DeltaExchangeBalance,
  DeltaOrderRequest,
  ApiResponse,
  TradeSignal
} from './index';

// Auth API Types
export interface LoginRequest {
  email: string;
  password: string;
}

export interface LoginResponse {
  token: string;
  user: UserData;
}

export interface RegisterRequest {
  email: string;
  password: string;
  username: string;
}

export interface UserData {
  id: string;
  email: string;
  username: string;
  createdAt: string;
  role: string;
  settings?: UserSettings;
}

export interface UserSettings {
  defaultLeverage: number;
  theme: 'light' | 'dark' | 'system';
  tradingViewInterval: string;
  defaultSymbol: string;
  showPortfolioValue: boolean;
  defaultOrderType: string;
  notifications: NotificationSettings;
}

export interface NotificationSettings {
  email: boolean;
  browser: boolean;
  mobile: boolean;
  orderFills: boolean;
  priceAlerts: boolean;
  tradingSignals: boolean;
}

// API Requests Types
export interface APIKeyRequest {
  name: string;
  exchange: string;
  publicKey: string;
  secretKey: string;
  passphrase?: string;
}

export interface UpdateUserSettingsRequest {
  settings: Partial<UserSettings>;
}

export interface CreateOrderRequest extends DeltaOrderRequest {}

export interface CancelOrderRequest {
  orderId: number;
  symbol: string;
}

export interface UpdateLeverageRequest {
  symbol: string;
  leverage: number;
}

export interface CreatePriceAlertRequest {
  symbol: string;
  price: number;
  type: 'above' | 'below';
  expiresAt?: string;
  notificationChannels?: ('email' | 'browser' | 'mobile')[];
}

// API Response Types
export type MarketsResponse = ApiResponse<DeltaExchangeMarket[]>;
export type MarketResponse = ApiResponse<DeltaExchangeMarket>;
export type PositionsResponse = ApiResponse<DeltaExchangePosition[]>;
export type PositionResponse = ApiResponse<DeltaExchangePosition>;
export type OrdersResponse = ApiResponse<DeltaExchangeOrder[]>;
export type OrderResponse = ApiResponse<DeltaExchangeOrder>;
export type BalancesResponse = ApiResponse<DeltaExchangeBalance[]>;
export type BalanceResponse = ApiResponse<DeltaExchangeBalance>;
export type TradeHistoryResponse = ApiResponse<{
  trades: {
    id: string;
    symbol: string;
    price: number;
    size: number;
    side: 'buy' | 'sell';
    timestamp: string;
    fee: number;
    feeCurrency: string;
  }[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
  };
}>;

export type UserResponse = ApiResponse<UserData>;
export type APIKeysResponse = ApiResponse<{
  id: string;
  name: string;
  exchange: string;
  publicKey: string;
  createdAt: string;
  lastUsed?: string;
}[]>;
export type PriceAlertsResponse = ApiResponse<{
  id: string;
  symbol: string;
  price: number;
  type: 'above' | 'below';
  createdAt: string;
  expiresAt?: string;
  triggered: boolean;
  notificationChannels: string[];
}[]>;

export type TradeSignalsResponse = ApiResponse<TradeSignal[]>;

// Websocket API Types
export interface WebSocketMessage<T = any> {
  type: string;
  data: T;
  timestamp: number;
}

export type MarketUpdateMessage = WebSocketMessage<{
  symbol: string;
  lastPrice: number;
  markPrice: number;
  indexPrice?: number;
  fundingRate?: number;
  volume24h?: number;
  change24h?: number;
}>;

export type OrderBookUpdateMessage = WebSocketMessage<{
  symbol: string;
  asks: [number, number][];
  bids: [number, number][];
  lastUpdateId: number;
}>;

export type TradeUpdateMessage = WebSocketMessage<{
  symbol: string;
  trades: {
    id: string;
    price: number;
    size: number;
    side: 'buy' | 'sell';
    timestamp: string;
  }[];
}>;

export type OrderUpdateMessage = WebSocketMessage<DeltaExchangeOrder>;
export type PositionUpdateMessage = WebSocketMessage<DeltaExchangePosition>;
export type BalanceUpdateMessage = WebSocketMessage<DeltaExchangeBalance>;
export type SignalUpdateMessage = WebSocketMessage<TradeSignal>; 