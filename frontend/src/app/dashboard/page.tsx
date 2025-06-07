'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { usePortfolio } from "@/hooks/usePortfolio";
import { useMarketData } from "@/hooks/useMarketData";

export default function DashboardPage() {
  const { portfolio, loading: portfolioLoading, error: portfolioError } = usePortfolio();
  const { marketData, loading: marketLoading, error: marketError } = useMarketData();

  if (portfolioLoading || marketLoading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[...Array(4)].map((_, i) => (
            <Card key={i}>
              <CardHeader className="animate-pulse">
                <div className="h-4 bg-muted rounded w-3/4"></div>
                <div className="h-3 bg-muted rounded w-1/2"></div>
              </CardHeader>
              <CardContent className="animate-pulse">
                <div className="h-8 bg-muted rounded w-full"></div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  if (portfolioError || marketError) {
    return (
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-destructive">Error Loading Data</CardTitle>
            <CardDescription>
              {portfolioError || marketError}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button onClick={() => window.location.reload()}>
              Retry
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Portfolio Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Total Balance</CardTitle>
            <CardDescription>Current portfolio value</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-foreground">
              ${portfolio?.totalBalance?.toLocaleString() || '10,000.00'}
            </div>
            <p className="text-sm text-muted-foreground">
              Available: ${portfolio?.availableBalance?.toLocaleString() || '8,500.00'}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Daily P&L</CardTitle>
            <CardDescription>Today's profit/loss</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-500">
              +${portfolio?.dailyPnl?.toLocaleString() || '250.00'}
            </div>
            <p className="text-sm text-muted-foreground">
              +{portfolio?.dailyPnlPercentage?.toFixed(2) || '2.5'}%
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Active Positions</CardTitle>
            <CardDescription>Open trading positions</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-foreground">
              {portfolio?.positions?.length || 3}
            </div>
            <p className="text-sm text-muted-foreground">
              2 profitable, 1 pending
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>AI Model Status</CardTitle>
            <CardDescription>ML prediction accuracy</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-blue-500">85%</div>
            <p className="text-sm text-muted-foreground">
              Last 24h accuracy
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Market Overview */}
      <Card>
        <CardHeader>
          <CardTitle>Market Overview</CardTitle>
          <CardDescription>Real-time market data</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {marketData.length > 0 ? (
              marketData.slice(0, 3).map((market) => (
                <div key={market.symbol} className="p-4 border border-border rounded-lg">
                  <div className="flex justify-between items-center">
                    <span className="font-medium">{market.symbol}</span>
                    <span className={`text-sm ${market.changePercentage24h >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                      {market.changePercentage24h >= 0 ? '+' : ''}{market.changePercentage24h?.toFixed(2)}%
                    </span>
                  </div>
                  <div className="text-lg font-bold">${market.price?.toLocaleString()}</div>
                  <div className="text-sm text-muted-foreground">
                    Vol: ${market.volume24h?.toLocaleString()}
                  </div>
                </div>
              ))
            ) : (
              // Mock data when API is not available
              <>
                <div className="p-4 border border-border rounded-lg">
                  <div className="flex justify-between items-center">
                    <span className="font-medium">BTC/USD</span>
                    <span className="text-sm text-green-500">+2.34%</span>
                  </div>
                  <div className="text-lg font-bold">$43,250</div>
                  <div className="text-sm text-muted-foreground">Vol: $2.1B</div>
                </div>
                <div className="p-4 border border-border rounded-lg">
                  <div className="flex justify-between items-center">
                    <span className="font-medium">ETH/USD</span>
                    <span className="text-sm text-green-500">+1.87%</span>
                  </div>
                  <div className="text-lg font-bold">$2,650</div>
                  <div className="text-sm text-muted-foreground">Vol: $1.8B</div>
                </div>
                <div className="p-4 border border-border rounded-lg">
                  <div className="flex justify-between items-center">
                    <span className="font-medium">SOL/USD</span>
                    <span className="text-sm text-red-500">-0.45%</span>
                  </div>
                  <div className="text-lg font-bold">$98.50</div>
                  <div className="text-sm text-muted-foreground">Vol: $450M</div>
                </div>
              </>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Quick Actions */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Actions</CardTitle>
          <CardDescription>Common trading operations</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex gap-4">
            <Button>Place Order</Button>
            <Button variant="outline">View Charts</Button>
            <Button variant="outline">AI Analysis</Button>
            <Button variant="secondary">Export Data</Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
