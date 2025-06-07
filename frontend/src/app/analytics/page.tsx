'use client';

import { PositionTracking } from '@/components/trading/PositionTracking'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Brain, TrendingUp, Target, Activity, Zap, BarChart3 } from 'lucide-react'

export default function AnalyticsPage() {
  const aiMetrics = {
    modelAccuracy: 87.3,
    predictionsToday: 24,
    successfulTrades: 18,
    totalProfit: 2450,
    riskScore: 'Medium',
    lastUpdated: new Date().toLocaleTimeString(),
  }

  const modelPerformance = [
    { model: 'SMC Pattern Recognition', accuracy: 89.2, trades: 156, profit: 12450 },
    { model: 'Fibonacci Retracement', accuracy: 84.7, trades: 89, profit: 8920 },
    { model: 'Support/Resistance', accuracy: 91.1, trades: 203, profit: 15670 },
    { model: 'Volume Analysis', accuracy: 76.8, trades: 67, profit: 4320 },
  ]

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">AI Analytics</h1>
        <p className="text-muted-foreground">
          Advanced machine learning insights and trading performance analytics
        </p>
      </div>

      {/* AI Performance Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Model Accuracy</CardTitle>
            <Brain className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-blue-500">{aiMetrics.modelAccuracy}%</div>
            <p className="text-xs text-muted-foreground">
              Last 30 days performance
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Predictions Today</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{aiMetrics.predictionsToday}</div>
            <p className="text-xs text-muted-foreground">
              {aiMetrics.successfulTrades} successful
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">AI Profit</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-500">${aiMetrics.totalProfit}</div>
            <p className="text-xs text-muted-foreground">
              From AI trades this month
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Risk Level</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-yellow-500">{aiMetrics.riskScore}</div>
            <p className="text-xs text-muted-foreground">
              Updated {aiMetrics.lastUpdated}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Model Performance Table */}
      <Card>
        <CardHeader>
          <CardTitle>Model Performance Breakdown</CardTitle>
          <CardDescription>Detailed performance metrics for each AI model</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {modelPerformance.map((model, index) => (
              <div key={index} className="flex items-center justify-between p-4 border rounded-lg">
                <div className="flex items-center space-x-3">
                  <BarChart3 className="h-5 w-5 text-blue-500" />
                  <div>
                    <div className="font-medium">{model.model}</div>
                    <div className="text-sm text-muted-foreground">
                      {model.trades} trades executed
                    </div>
                  </div>
                </div>
                <div className="flex items-center space-x-6">
                  <div className="text-right">
                    <div className="text-sm text-muted-foreground">Accuracy</div>
                    <div className="font-bold text-green-500">{model.accuracy}%</div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm text-muted-foreground">Profit</div>
                    <div className="font-bold">${model.profit.toLocaleString()}</div>
                  </div>
                  <Badge variant={model.accuracy > 85 ? 'default' : model.accuracy > 75 ? 'secondary' : 'destructive'}>
                    {model.accuracy > 85 ? 'Excellent' : model.accuracy > 75 ? 'Good' : 'Needs Improvement'}
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Position Tracking */}
      <PositionTracking />

      {/* AI Insights and Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Market Regime Detection</CardTitle>
            <CardDescription>AI-powered market condition analysis</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="p-4 bg-green-500/10 border border-green-500/20 rounded-lg">
                <div className="flex justify-between items-center mb-2">
                  <span className="font-medium">BTC Market</span>
                  <Badge variant="default">TRENDING</Badge>
                </div>
                <div className="text-sm text-muted-foreground">
                  Strong upward momentum detected. Target ratio: 5:1-8:1
                </div>
              </div>

              <div className="p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
                <div className="flex justify-between items-center mb-2">
                  <span className="font-medium">ETH Market</span>
                  <Badge variant="secondary">RANGING</Badge>
                </div>
                <div className="text-sm text-muted-foreground">
                  Sideways movement. Target ratio: 2:1-3:1
                </div>
              </div>

              <div className="p-4 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
                <div className="flex justify-between items-center mb-2">
                  <span className="font-medium">SOL Market</span>
                  <Badge variant="outline">UNCERTAIN</Badge>
                </div>
                <div className="text-sm text-muted-foreground">
                  Mixed signals. Awaiting clearer direction.
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Recent AI Predictions</CardTitle>
            <CardDescription>Latest model-generated trading signals</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex justify-between items-center p-3 border rounded-lg">
                <div>
                  <div className="font-medium">BTC/USD Long</div>
                  <div className="text-sm text-muted-foreground">Entry: $50,100 | Target: $52,200</div>
                </div>
                <div className="text-right">
                  <Badge variant="default">ACTIVE</Badge>
                  <div className="text-sm text-muted-foreground">87% confidence</div>
                </div>
              </div>

              <div className="flex justify-between items-center p-3 border rounded-lg">
                <div>
                  <div className="font-medium">ETH/USD Short</div>
                  <div className="text-sm text-muted-foreground">Entry: $3,120 | Target: $3,020</div>
                </div>
                <div className="text-right">
                  <Badge variant="secondary">PENDING</Badge>
                  <div className="text-sm text-muted-foreground">74% confidence</div>
                </div>
              </div>

              <div className="flex justify-between items-center p-3 border rounded-lg">
                <div>
                  <div className="font-medium">SOL/USD Long</div>
                  <div className="text-sm text-muted-foreground">Entry: $97.50 | Target: $102.00</div>
                </div>
                <div className="text-right">
                  <Badge variant="default">EXECUTED</Badge>
                  <div className="text-sm text-green-500">+2.1% P&L</div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Action Buttons */}
      <div className="flex gap-4">
        <Button>
          <Brain className="mr-2 h-4 w-4" />
          Retrain Models
        </Button>
        <Button variant="outline">
          <BarChart3 className="mr-2 h-4 w-4" />
          Export Report
        </Button>
        <Button variant="secondary">
          <Target className="mr-2 h-4 w-4" />
          Model Settings
        </Button>
      </div>
    </div>
  )
}
