import {
  BarChart3,
  Bot,
  ChartCandlestick,
  Home,
  Settings,
  TrendingUp,
  Wallet,
  Activity,
  Brain,
} from "lucide-react"

// Menu items for navigation
export const menuItems = [
  {
    title: "Overview",
    url: "/",
    icon: Home,
    description: "Dashboard overview"
  },
  {
    title: "Dashboard",
    url: "/dashboard",
    icon: BarChart3,
    description: "Trading dashboard"
  },
  {
    title: "Live Charts",
    url: "/charts",
    icon: ChartCandlestick,
    description: "Real-time charts"
  },
  {
    title: "AI Analytics",
    url: "/analytics",
    icon: Brain,
    description: "AI model insights"
  },
]

export const tradingItems = [
  {
    title: "Portfolio",
    url: "/portfolio",
    icon: Wallet,
    description: "Portfolio management"
  },
  {
    title: "Positions",
    url: "/positions",
    icon: TrendingUp,
    description: "Active positions"
  },
  {
    title: "Trading Bot",
    url: "/bot",
    icon: Bot,
    description: "Automated trading"
  },
  {
    title: "Performance",
    url: "/performance",
    icon: Activity,
    description: "Trading performance"
  },
]

export const settingsItems = [
  {
    title: "Settings",
    url: "/settings",
    icon: Settings,
    description: "App settings"
  },
]

// Helper function to get breadcrumb segments based on pathname
export function getBreadcrumbSegments(pathname: string) {
  const segments = pathname.split('/').filter(Boolean);
  
  return segments.map((segment, index) => {
    const href = `/${segments.slice(0, index + 1).join('/')}`;
    
    // Convert path segment to a more readable name
    let name = segment.charAt(0).toUpperCase() + segment.slice(1);
    
    // Special cases for common abbreviations
    if (segment === 'ai') name = 'AI';
    
    return { name, href };
  });
}

// Get page title based on the current route
export function getPageTitle(pathname: string) {
  // Remove trailing slash if present
  const path = pathname.endsWith('/') ? pathname.slice(0, -1) : pathname;
  
  // First check exact matches
  if (path === '/') return 'Dashboard';
  
  // Then check all navigation items
  const allItems = [...menuItems, ...tradingItems, ...settingsItems];
  const matchingItem = allItems.find(item => item.url === path);
  
  if (matchingItem) return matchingItem.title;
  
  // Fallback: get the last segment and capitalize it
  const segments = path.split('/').filter(Boolean);
  if (segments.length > 0) {
    const lastSegment = segments[segments.length - 1];
    return lastSegment.charAt(0).toUpperCase() + lastSegment.slice(1);
  }
  
  return 'SmartMarketOOPS';
} 