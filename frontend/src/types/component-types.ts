import { ReactNode } from 'react';
import { OrderSide, OrderStatus, DeltaOrderType } from './index';

// Base Props
export interface BaseProps {
  className?: string;
  id?: string;
}

// Children Props
export interface ChildrenProps extends BaseProps {
  children: ReactNode;
}

// Dashboard Component Props
export interface DashboardCardProps extends ChildrenProps {
  title: string;
  subtitle?: string;
  isLoading?: boolean;
  error?: string | null;
  onRefresh?: () => void;
  fullHeight?: boolean;
  padding?: 'none' | 'small' | 'normal' | 'large';
  scrollable?: boolean;
  headerActions?: ReactNode;
}

export interface TabViewProps extends ChildrenProps {
  tabs: {
    id: string;
    label: string;
    content: ReactNode;
    icon?: ReactNode;
  }[];
  defaultTab?: string;
  onChange?: (tabId: string) => void;
}

// Trading Component Props
export interface OrderFormProps extends BaseProps {
  symbol: string;
  onSubmit: (order: OrderFormData) => void;
  isSubmitting?: boolean;
  defaultValues?: Partial<OrderFormData>;
  availableBalance?: number;
  currentPrice?: number;
  maxLeverage?: number;
  errors?: Record<string, string>;
}

export interface OrderFormData {
  symbol: string;
  side: OrderSide;
  type: DeltaOrderType;
  size: number;
  price?: number;
  stopPrice?: number;
  leverage: number;
  reduceOnly: boolean;
  postOnly: boolean;
}

export interface OrderBookProps extends BaseProps {
  symbol: string;
  asks: [number, number][]; // [price, size]
  bids: [number, number][]; // [price, size]
  lastPrice?: number;
  precision?: number;
  depth?: number;
  isLoading?: boolean;
  onPriceClick?: (price: number) => void;
}

export interface TradeHistoryProps extends BaseProps {
  symbol: string;
  trades: {
    id: string;
    price: number;
    size: number;
    side: OrderSide;
    timestamp: string;
  }[];
  isLoading?: boolean;
}

export interface PositionItemProps extends BaseProps {
  position: {
    symbol: string;
    side: OrderSide;
    size: number;
    entryPrice: number;
    markPrice: number;
    leverage: number;
    unrealizedPnl: number;
    unrealizedPnlPercentage: number;
    liquidationPrice?: number;
  };
  onClose?: () => void;
  onUpdateLeverage?: (leverage: number) => void;
  expanded?: boolean;
  onToggleExpand?: () => void;
}

// Chart Component Props
export interface TradingChartProps extends BaseProps {
  symbol: string;
  interval?: string;
  height?: number | string;
  autosize?: boolean;
  showToolbar?: boolean;
  theme?: 'light' | 'dark';
  indicators?: string[];
  trades?: {
    price: number;
    side: OrderSide;
    timestamp: string;
  }[];
  orders?: {
    price: number;
    side: OrderSide;
    status: OrderStatus;
    type: DeltaOrderType;
  }[];
}

// Status Indicator Props
export interface StatusIndicatorProps extends BaseProps {
  status: 'online' | 'offline' | 'warning' | 'error';
  size?: 'sm' | 'md' | 'lg';
  label?: string;
  showLabel?: boolean;
  pulse?: boolean;
}

// Tooltips and Helpers
export interface TooltipProps extends ChildrenProps {
  content: ReactNode;
  position?: 'top' | 'right' | 'bottom' | 'left';
  delay?: number;
  maxWidth?: number | string;
}

// Data Display Components
export interface StatCardProps extends BaseProps {
  title: string;
  value: string | number;
  change?: number;
  precision?: number;
  prefix?: string;
  suffix?: string;
  icon?: ReactNode;
  isLoading?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

export interface AssetSelectorProps extends BaseProps {
  assets: {
    symbol: string;
    name: string;
    icon?: string;
  }[];
  selectedAsset?: string;
  onChange: (symbol: string) => void;
  isLoading?: boolean;
  showSearch?: boolean;
  maxHeight?: number | string;
}

// Form Control Props
export interface ToggleSwitchProps extends BaseProps {
  label?: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

export interface NumericInputProps extends BaseProps {
  label?: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  precision?: number;
  prefix?: string;
  suffix?: string;
  error?: string;
  disabled?: boolean;
  placeholder?: string;
} 