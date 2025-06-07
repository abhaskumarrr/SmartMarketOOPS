'use client';

import { PortfolioDashboard } from '@/components/dashboard/PortfolioDashboard'
import { TradingChart } from '@/components/charts/TradingChart'
import { OrderManagement } from '@/components/trading/OrderManagement'

export default function DashboardPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Trading Dashboard</h1>
        <p className="text-muted-foreground">
          Monitor your portfolio, manage positions, and execute trades
        </p>
      </div>

      {/* Portfolio Overview */}
      <PortfolioDashboard />

      {/* Trading Chart and Order Management */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="xl:col-span-2">
          <TradingChart symbol="BTCUSD" height={500} />
        </div>
        <div className="xl:col-span-1">
          <OrderManagement />
        </div>
      </div>
    </div>
  )
}


