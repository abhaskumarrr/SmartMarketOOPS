'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

export default function ChartsPage() {
  return (
    <div className="min-h-screen bg-background p-8">
      <div className="max-w-6xl mx-auto">
        <header className="mb-8">
          <h1 className="text-4xl font-bold text-foreground mb-2">
            Live Charts
          </h1>
          <p className="text-xl text-muted-foreground">
            Real-time trading charts and technical analysis
          </p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Chart Area */}
          <div className="lg:col-span-2">
            <Card className="h-[600px]">
              <CardHeader>
                <CardTitle>BTC/USD - Live Chart</CardTitle>
                <CardDescription>
                  TradingView Lightweight Charts integration coming soon
                </CardDescription>
              </CardHeader>
              <CardContent className="h-full flex items-center justify-center">
                <div className="text-center">
                  <div className="text-6xl mb-4">📈</div>
                  <h3 className="text-xl font-semibold mb-2">Professional Charts Coming Soon</h3>
                  <p className="text-muted-foreground mb-4">
                    TradingView Lightweight Charts integration in progress
                  </p>
                  <div className="space-y-2 text-sm text-muted-foreground">
                    <p>✅ Real-time candlestick data</p>
                    <p>✅ Multiple timeframes (5m, 15m, 1h, 4h, 1d)</p>
                    <p>✅ Technical indicators</p>
                    <p>✅ AI model predictions overlay</p>
                    <p>✅ Fibonacci retracements</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Side Panel */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Market Data</CardTitle>
                <CardDescription>Live price information</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="font-medium">BTC/USD</span>
                    <div className="text-right">
                      <div className="font-bold">$43,250.00</div>
                      <div className="text-sm text-green-500">+2.34%</div>
                    </div>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="font-medium">ETH/USD</span>
                    <div className="text-right">
                      <div className="font-bold">$2,650.00</div>
                      <div className="text-sm text-green-500">+1.87%</div>
                    </div>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="font-medium">SOL/USD</span>
                    <div className="text-right">
                      <div className="font-bold">$98.50</div>
                      <div className="text-sm text-red-500">-0.45%</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>AI Predictions</CardTitle>
                <CardDescription>Model analysis</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="p-3 bg-green-500/10 border border-green-500/20 rounded-lg">
                    <div className="flex justify-between items-center mb-2">
                      <span className="font-medium">BTC/USD</span>
                      <span className="text-sm text-green-500">BULLISH</span>
                    </div>
                    <div className="text-sm text-muted-foreground">
                      Confidence: 87% | Target: $45,200
                    </div>
                  </div>
                  
                  <div className="p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg">
                    <div className="flex justify-between items-center mb-2">
                      <span className="font-medium">ETH/USD</span>
                      <span className="text-sm text-blue-500">NEUTRAL</span>
                    </div>
                    <div className="text-sm text-muted-foreground">
                      Confidence: 72% | Range: $2,600-$2,700
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Quick Actions</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <Button className="w-full">Place Order</Button>
                  <Button variant="outline" className="w-full">Set Alert</Button>
                  <Button variant="secondary" className="w-full">Export Data</Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
