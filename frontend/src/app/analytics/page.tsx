'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

export default function AnalyticsPage() {
  return (
    <div className="min-h-screen bg-background p-8">
      <div className="max-w-6xl mx-auto">
        <header className="mb-8">
          <h1 className="text-4xl font-bold text-foreground mb-2">
            AI Analytics
          </h1>
          <p className="text-xl text-muted-foreground">
            Machine learning model performance and insights
          </p>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          <Card>
            <CardHeader>
              <CardTitle>Model Accuracy</CardTitle>
              <CardDescription>Prediction success rate</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-blue-500 mb-2">85.7%</div>
              <p className="text-sm text-muted-foreground">Last 30 days</p>
              <div className="mt-4 text-sm">
                <div className="flex justify-between">
                  <span>Correct predictions:</span>
                  <span className="font-medium">342/399</span>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Win Rate</CardTitle>
              <CardDescription>Profitable trades percentage</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-green-500 mb-2">72.3%</div>
              <p className="text-sm text-muted-foreground">Based on executed trades</p>
              <div className="mt-4 text-sm">
                <div className="flex justify-between">
                  <span>Profitable trades:</span>
                  <span className="font-medium">89/123</span>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Average Return</CardTitle>
              <CardDescription>Per trade performance</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-green-500 mb-2">+3.2%</div>
              <p className="text-sm text-muted-foreground">Average per successful trade</p>
              <div className="mt-4 text-sm">
                <div className="flex justify-between">
                  <span>Best trade:</span>
                  <span className="font-medium text-green-500">+12.4%</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <Card>
            <CardHeader>
              <CardTitle>Model Performance by Asset</CardTitle>
              <CardDescription>Accuracy breakdown by trading pair</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="font-medium">BTC/USD</span>
                  <div className="text-right">
                    <div className="font-bold text-green-500">89.2%</div>
                    <div className="text-sm text-muted-foreground">156 predictions</div>
                  </div>
                </div>
                <div className="flex justify-between items-center">
                  <span className="font-medium">ETH/USD</span>
                  <div className="text-right">
                    <div className="font-bold text-blue-500">82.7%</div>
                    <div className="text-sm text-muted-foreground">127 predictions</div>
                  </div>
                </div>
                <div className="flex justify-between items-center">
                  <span className="font-medium">SOL/USD</span>
                  <div className="text-right">
                    <div className="font-bold text-yellow-500">78.4%</div>
                    <div className="text-sm text-muted-foreground">116 predictions</div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Market Regime Detection</CardTitle>
              <CardDescription>Current market conditions analysis</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="p-4 bg-green-500/10 border border-green-500/20 rounded-lg">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-medium">BTC Market</span>
                    <span className="text-sm text-green-500 font-medium">TRENDING</span>
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Strong upward momentum detected. Target ratio: 5:1-8:1
                  </div>
                </div>
                
                <div className="p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-medium">ETH Market</span>
                    <span className="text-sm text-blue-500 font-medium">RANGING</span>
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Sideways movement. Target ratio: 2:1-3:1
                  </div>
                </div>
                
                <div className="p-4 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-medium">SOL Market</span>
                    <span className="text-sm text-yellow-500 font-medium">UNCERTAIN</span>
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Mixed signals. Awaiting clearer direction.
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Recent Model Predictions</CardTitle>
            <CardDescription>Latest AI-generated trading signals</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between items-center p-4 border border-border rounded-lg">
                <div>
                  <div className="font-medium">BTC/USD Long Signal</div>
                  <div className="text-sm text-muted-foreground">Entry: $43,100 | Target: $45,200</div>
                </div>
                <div className="text-right">
                  <div className="text-sm text-green-500 font-medium">ACTIVE</div>
                  <div className="text-sm text-muted-foreground">Confidence: 87%</div>
                </div>
              </div>
              
              <div className="flex justify-between items-center p-4 border border-border rounded-lg">
                <div>
                  <div className="font-medium">ETH/USD Short Signal</div>
                  <div className="text-sm text-muted-foreground">Entry: $2,680 | Target: $2,580</div>
                </div>
                <div className="text-right">
                  <div className="text-sm text-blue-500 font-medium">PENDING</div>
                  <div className="text-sm text-muted-foreground">Confidence: 74%</div>
                </div>
              </div>
              
              <div className="flex justify-between items-center p-4 border border-border rounded-lg">
                <div>
                  <div className="font-medium">SOL/USD Long Signal</div>
                  <div className="text-sm text-muted-foreground">Entry: $97.50 | Target: $102.00</div>
                </div>
                <div className="text-right">
                  <div className="text-sm text-green-500 font-medium">EXECUTED</div>
                  <div className="text-sm text-muted-foreground">P&L: +2.1%</div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="mt-8 flex gap-4">
          <Button>Retrain Models</Button>
          <Button variant="outline">Export Report</Button>
          <Button variant="secondary">Model Settings</Button>
        </div>
      </div>
    </div>
  );
}
