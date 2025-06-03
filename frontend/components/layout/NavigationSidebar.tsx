'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  LayoutDashboard,
  TrendingUp,
  Bot,
  BarChart3,
  Settings,
  Activity,
  Wallet,
  Target,
  ChevronLeft,
  ChevronRight,
  Zap,
  Brain
} from 'lucide-react';

interface NavItem {
  name: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
  badge?: string;
  description: string;
}

const navigation: NavItem[] = [
  {
    name: 'Dashboard',
    href: '/dashboard',
    icon: LayoutDashboard,
    description: 'Overview and analytics'
  },
  {
    name: 'Paper Trading',
    href: '/paper-trading',
    icon: TrendingUp,
    badge: 'Live',
    description: 'Live paper trading with Delta Exchange'
  },
  {
    name: 'Trading Bots',
    href: '/bots',
    icon: Bot,
    description: 'AI-powered trading automation'
  },
  {
    name: 'Market Analysis',
    href: '/analysis',
    icon: BarChart3,
    description: 'Technical and fundamental analysis'
  },
  {
    name: 'Portfolio',
    href: '/portfolio',
    icon: Wallet,
    description: 'Portfolio management and tracking'
  },
  {
    name: 'Signals',
    href: '/signals',
    icon: Target,
    description: 'Trading signals and alerts'
  },
  {
    name: 'AI Intelligence',
    href: '/intelligence',
    icon: Brain,
    description: 'ML-powered market insights'
  },
  {
    name: 'Performance',
    href: '/performance',
    icon: Activity,
    description: 'Trading performance metrics'
  }
];

const bottomNavigation: NavItem[] = [
  {
    name: 'Settings',
    href: '/settings',
    icon: Settings,
    description: 'Platform configuration'
  }
];

export function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);
  const pathname = usePathname();

  const isActive = (href: string) => {
    if (!pathname) return false;
    if (href === '/dashboard') {
      return pathname === '/' || pathname === '/dashboard';
    }
    return pathname.startsWith(href);
  };

  return (
    <div className={`${collapsed ? 'w-16' : 'w-64'} bg-slate-900/95 backdrop-blur-sm border-r border-slate-800 flex flex-col transition-all duration-300 ease-in-out`}>
      {/* Header */}
      <div className="p-4 border-b border-slate-800">
        <div className="flex items-center justify-between">
          {!collapsed && (
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <Zap className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-lg font-bold text-white">SmartMarket</h1>
                <p className="text-xs text-slate-400">Trading Platform</p>
              </div>
            </div>
          )}
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors"
          >
            {collapsed ? (
              <ChevronRight className="w-4 h-4" />
            ) : (
              <ChevronLeft className="w-4 h-4" />
            )}
          </button>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-2">
        {navigation.map((item) => {
          const active = isActive(item.href);
          return (
            <Link
              key={item.name}
              href={item.href}
              className={`group flex items-center px-3 py-2.5 text-sm font-medium rounded-lg transition-all duration-200 ${
                active
                  ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/25'
                  : 'text-slate-300 hover:text-white hover:bg-slate-800'
              }`}
            >
              <item.icon className={`${collapsed ? 'w-5 h-5' : 'w-5 h-5 mr-3'} flex-shrink-0`} />
              {!collapsed && (
                <>
                  <span className="flex-1">{item.name}</span>
                  {item.badge && (
                    <span className="ml-2 px-2 py-0.5 text-xs font-medium bg-emerald-500/20 text-emerald-400 rounded-full">
                      {item.badge}
                    </span>
                  )}
                </>
              )}
              
              {/* Tooltip for collapsed state */}
              {collapsed && (
                <div className="absolute left-16 ml-2 px-3 py-2 bg-slate-800 text-white text-sm rounded-lg shadow-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-50 whitespace-nowrap">
                  <div className="font-medium">{item.name}</div>
                  <div className="text-xs text-slate-400">{item.description}</div>
                  {item.badge && (
                    <div className="mt-1">
                      <span className="px-2 py-0.5 text-xs font-medium bg-emerald-500/20 text-emerald-400 rounded-full">
                        {item.badge}
                      </span>
                    </div>
                  )}
                </div>
              )}
            </Link>
          );
        })}
      </nav>

      {/* Bottom Navigation */}
      <div className="p-4 border-t border-slate-800 space-y-2">
        {bottomNavigation.map((item) => {
          const active = isActive(item.href);
          return (
            <Link
              key={item.name}
              href={item.href}
              className={`group flex items-center px-3 py-2.5 text-sm font-medium rounded-lg transition-all duration-200 ${
                active
                  ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/25'
                  : 'text-slate-300 hover:text-white hover:bg-slate-800'
              }`}
            >
              <item.icon className={`${collapsed ? 'w-5 h-5' : 'w-5 h-5 mr-3'} flex-shrink-0`} />
              {!collapsed && <span className="flex-1">{item.name}</span>}
              
              {/* Tooltip for collapsed state */}
              {collapsed && (
                <div className="absolute left-16 ml-2 px-3 py-2 bg-slate-800 text-white text-sm rounded-lg shadow-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-50 whitespace-nowrap">
                  <div className="font-medium">{item.name}</div>
                  <div className="text-xs text-slate-400">{item.description}</div>
                </div>
              )}
            </Link>
          );
        })}
      </div>

      {/* Status Indicator */}
      <div className="p-4 border-t border-slate-800">
        <div className={`flex items-center ${collapsed ? 'justify-center' : 'space-x-3'}`}>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></div>
            {!collapsed && (
              <div>
                <p className="text-xs font-medium text-slate-300">System Status</p>
                <p className="text-xs text-emerald-400">All systems operational</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
