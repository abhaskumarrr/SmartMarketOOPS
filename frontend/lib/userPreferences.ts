// User preferences types
export interface ChartPreferences {
  defaultTimeframe: '1m' | '5m' | '15m' | '1h' | '4h' | '1d' | '1w';
  defaultIndicators: string[];
  showPredictions: boolean;
  showTrades: boolean;
  chartStyle: 'candles' | 'line' | 'bars';
  chartTheme: 'light' | 'dark' | 'system';
}

export interface NotificationPreferences {
  enablePriceAlerts: boolean;
  enableTradeNotifications: boolean;
  enablePredictionAlerts: boolean;
  soundAlerts: boolean;
  minConfidenceThreshold: number; // 0-1
}

export interface LayoutPreferences {
  sidebarCollapsed: boolean;
  dashboardLayout: 'compact' | 'expanded' | 'custom';
  visibleWidgets: string[];
  widgetPositions?: Record<string, { x: number; y: number; w: number; h: number }>;
}

export interface DataPreferences {
  refreshRate: 'realtime' | '5s' | '10s' | '30s' | '1m' | 'manual';
  historicalDataRange: '1d' | '7d' | '30d' | '90d' | '1y' | 'all';
  dataSource: 'default' | 'custom';
  customApiEndpoint?: string;
}

export interface ThemePreferences {
  mode: 'light' | 'dark' | 'system';
  primaryColor: string;
  accentColor: string;
  fontScale: number;
}

export interface UserPreferences {
  version: string;
  lastUpdated: string;
  chart: ChartPreferences;
  notifications: NotificationPreferences;
  layout: LayoutPreferences;
  data: DataPreferences;
  theme: ThemePreferences;
}

// Default preferences
export const defaultPreferences: UserPreferences = {
  version: '1.0',
  lastUpdated: new Date().toISOString(),
  chart: {
    defaultTimeframe: '1h',
    defaultIndicators: ['SMA', 'EMA', 'Volume'],
    showPredictions: true,
    showTrades: true,
    chartStyle: 'candles',
    chartTheme: 'system',
  },
  notifications: {
    enablePriceAlerts: true,
    enableTradeNotifications: true,
    enablePredictionAlerts: true,
    soundAlerts: false,
    minConfidenceThreshold: 0.7,
  },
  layout: {
    sidebarCollapsed: false,
    dashboardLayout: 'expanded',
    visibleWidgets: [
      'chart', 
      'trades', 
      'predictions', 
      'performance', 
      'signals'
    ],
  },
  data: {
    refreshRate: 'realtime',
    historicalDataRange: '30d',
    dataSource: 'default',
  },
  theme: {
    mode: 'system',
    primaryColor: '#3f51b5', // Indigo
    accentColor: '#f50057', // Pink
    fontScale: 1,
  },
};

// Storage key
const STORAGE_KEY = 'smartmarket_user_preferences';

// Load preferences from localStorage
export const loadPreferences = (): UserPreferences => {
  if (typeof window === 'undefined') {
    return defaultPreferences;
  }
  
  try {
    const storedPrefs = localStorage.getItem(STORAGE_KEY);
    if (!storedPrefs) {
      return defaultPreferences;
    }
    
    const parsed = JSON.parse(storedPrefs) as UserPreferences;
    
    // Ensure all required fields exist by merging with defaults
    // This handles cases where the stored preferences might be from an older version
    return {
      ...defaultPreferences,
      ...parsed,
      chart: { ...defaultPreferences.chart, ...parsed.chart },
      notifications: { ...defaultPreferences.notifications, ...parsed.notifications },
      layout: { ...defaultPreferences.layout, ...parsed.layout },
      data: { ...defaultPreferences.data, ...parsed.data },
      theme: { ...defaultPreferences.theme, ...parsed.theme },
    };
  } catch (error) {
    console.error('Failed to load preferences:', error);
    return defaultPreferences;
  }
};

// Save preferences to localStorage
export const savePreferences = (preferences: UserPreferences): void => {
  if (typeof window === 'undefined') {
    return;
  }
  
  try {
    const updatedPrefs = {
      ...preferences,
      lastUpdated: new Date().toISOString(),
    };
    
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updatedPrefs));
  } catch (error) {
    console.error('Failed to save preferences:', error);
  }
};

// Update a section of the preferences
export const updatePreferences = <K extends keyof Omit<UserPreferences, 'version' | 'lastUpdated'>>(
  preferences: UserPreferences,
  section: K,
  updates: Partial<UserPreferences[K]>
): UserPreferences => {
  const updated = {
    ...preferences,
    [section]: {
      ...preferences[section],
      ...updates,
    },
    lastUpdated: new Date().toISOString(),
  };
  
  savePreferences(updated);
  return updated;
};

// Reset all preferences to defaults
export const resetPreferences = (): UserPreferences => {
  savePreferences(defaultPreferences);
  return defaultPreferences;
};

// Export preferences to JSON string
export const exportPreferences = (preferences: UserPreferences): string => {
  return JSON.stringify(preferences, null, 2);
};

// Import preferences from JSON string
export const importPreferences = (json: string): UserPreferences | null => {
  try {
    const imported = JSON.parse(json) as UserPreferences;
    
    // Validate that this is a valid preferences object
    if (!imported.version || !imported.chart || !imported.theme) {
      throw new Error('Invalid preferences format');
    }
    
    // Merge with defaults to ensure all required fields
    const merged = {
      ...defaultPreferences,
      ...imported,
      chart: { ...defaultPreferences.chart, ...imported.chart },
      notifications: { ...defaultPreferences.notifications, ...imported.notifications },
      layout: { ...defaultPreferences.layout, ...imported.layout },
      data: { ...defaultPreferences.data, ...imported.data },
      theme: { ...defaultPreferences.theme, ...imported.theme },
      lastUpdated: new Date().toISOString(),
    };
    
    savePreferences(merged);
    return merged;
  } catch (error) {
    console.error('Failed to import preferences:', error);
    return null;
  }
}; 