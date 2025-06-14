'use client';

import React, { useState } from 'react';
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardFooter, 
  CardHeader, 
  CardTitle 
} from '@/components/ui/card';
import { 
  Tabs, 
  TabsContent, 
  TabsList, 
  TabsTrigger 
} from '@/components/ui/tabs';
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { 
  ArrowUpRight, 
  ArrowDownRight, 
  TrendingUp,
  TrendingDown,
  BrainCircuit,
  BarChart2,
  LineChart,
  PieChart
} from 'lucide-react';

// Sample ML prediction data
const predictionData = [
  { symbol: 'BTCUSD', prediction: 'bullish', confidence: 78, priceChange: 3.2, timeframe: '24h' },
  { symbol: 'ETHUSD', prediction: 'bullish', confidence: 62, priceChange: 1.8, timeframe: '24h' },
  { symbol: 'SOLUSD', prediction: 'bearish', confidence: 56, priceChange: -1.2, timeframe: '24h' },
  { symbol: 'ADAUSD', prediction: 'neutral', confidence: 45, priceChange: 0.3, timeframe: '24h' },
  { symbol: 'DOTUSD', prediction: 'bearish', confidence: 67, priceChange: -2.1, timeframe: '24h' },
  { symbol: 'LINKUSD', prediction: 'bullish', confidence: 71, priceChange: 2.5, timeframe: '24h' },
];

// Sample historical performance data
const mlPerformanceData = [
  { date: 'Jun 01', accuracy: 68, profitFactor: 1.3 },
  { date: 'Jun 08', accuracy: 72, profitFactor: 1.5 },
  { date: 'Jun 15', accuracy: 65, profitFactor: 1.2 },
  { date: 'Jun 22', accuracy: 78, profitFactor: 1.7 },
  { date: 'Jun 29', accuracy: 75, profitFactor: 1.6 },
  { date: 'Jul 06', accuracy: 70, profitFactor: 1.4 },
  { date: 'Jul 13', accuracy: 82, profitFactor: 1.9 },
  { date: 'Jul 20', accuracy: 79, profitFactor: 1.8 },
];

// Sample feature importance data
const featureImportanceData = [
  { name: 'RSI', value: 24 },
  { name: 'MACD', value: 18 },
  { name: 'Volume', value: 15 },
  { name: 'Bollinger', value: 12 },
  { name: 'Price Action', value: 10 },
  { name: 'Moving Avg', value: 8 },
  { name: 'Other', value: 13 },
];

// Sample backtest results
const backtestResults = [
  { 
    name: 'LSTM Model',
    totalTrades: 142,
    winRate: 68.3,
    profitFactor: 1.72,
    maxDrawdown: 12.4,
    sharpeRatio: 1.45,
    strategy: 'Trend following using LSTM neural network',
    timeframe: '4h',
    lastUpdated: '2025-06-28',
  },
  { 
    name: 'XGBoost Classifier',
    totalTrades: 87,
    winRate: 72.1,
    profitFactor: 1.91,
    maxDrawdown: 8.2,
    sharpeRatio: 1.68,
    strategy: 'Mean reversion using XGBoost with custom feature engineering',
    timeframe: '1h',
    lastUpdated: '2025-06-30',
  },
  { 
    name: 'CNN-LSTM Hybrid',
    totalTrades: 104,
    winRate: 74.5,
    profitFactor: 2.05,
    maxDrawdown: 14.8,
    sharpeRatio: 1.72,
    strategy: 'Pattern recognition using CNN with LSTM for sequence modeling',
    timeframe: '1d',
    lastUpdated: '2025-07-01',
  },
];

export default function AnalyticsPage() {
  const [selectedTimeframe, setSelectedTimeframe] = useState('24h');
  
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">ML Analytics</h1>
          <p className="text-muted-foreground">
            Machine learning powered market insights and predictions
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Select
            value={selectedTimeframe}
            onValueChange={setSelectedTimeframe}
          >
            <SelectTrigger className="w-36">
              <SelectValue placeholder="Select timeframe" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="1h">1 Hour</SelectItem>
              <SelectItem value="4h">4 Hours</SelectItem>
              <SelectItem value="24h">24 Hours</SelectItem>
              <SelectItem value="7d">7 Days</SelectItem>
              <SelectItem value="30d">30 Days</SelectItem>
            </SelectContent>
          </Select>
          <Button>Refresh Data</Button>
        </div>
      </div>

      <Tabs defaultValue="predictions" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="predictions">ML Predictions</TabsTrigger>
          <TabsTrigger value="performance">Model Performance</TabsTrigger>
          <TabsTrigger value="features">Feature Analysis</TabsTrigger>
          <TabsTrigger value="backtests">Backtest Results</TabsTrigger>
        </TabsList>
        
        {/* ML Predictions Tab */}
        <TabsContent value="predictions" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* ML prediction cards */}
            {predictionData.map((item) => (
              <Card key={item.symbol}>
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <CardTitle>{item.symbol}</CardTitle>
                    <Badge 
                      variant={
                        item.prediction === 'bullish' 
                          ? 'default' 
                          : item.prediction === 'bearish' 
                            ? 'destructive' 
                            : 'secondary'
                      }
                    >
                      {item.prediction.toUpperCase()}
                    </Badge>
                  </div>
                  <CardDescription>
                    ML Confidence: {item.confidence}%
                  </CardDescription>
                </CardHeader>
                <CardContent className="pb-2">
                  <div className="flex items-center space-x-2">
                    <div className="text-2xl font-bold">
                      {item.priceChange >= 0 ? '+' : ''}{item.priceChange}%
                    </div>
                    {item.priceChange >= 0 ? (
                      <ArrowUpRight className="h-5 w-5 text-green-500" />
                    ) : (
                      <ArrowDownRight className="h-5 w-5 text-red-500" />
                    )}
                    <div className="text-sm text-muted-foreground">
                      Expected in {item.timeframe}
                    </div>
                  </div>
                  <Progress value={item.confidence} className="mt-2" />
                </CardContent>
                <CardFooter className="pt-2">
                  <Button variant="outline" size="sm" className="w-full">
                    View Details
                  </Button>
                </CardFooter>
              </Card>
            ))}
          </div>
          
          {/* ML Prediction Overview */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Market Sentiment Overview</CardTitle>
              <CardDescription>
                Aggregated market sentiment based on our ML models
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="flex flex-col items-center justify-center p-4 border rounded-lg">
                  <div className="text-sm font-medium text-muted-foreground mb-2">
                    Overall Market Prediction
                  </div>
                  <div className="flex items-center">
                    <TrendingUp className="h-8 w-8 text-green-500 mr-2" />
                    <span className="text-2xl font-bold">Bullish</span>
                  </div>
                  <div className="text-sm text-muted-foreground mt-2">
                    67% of assets show bullish signals
                  </div>
                </div>
                
                <div className="flex flex-col items-center justify-center p-4 border rounded-lg">
                  <div className="text-sm font-medium text-muted-foreground mb-2">
                    Average Confidence Score
                  </div>
                  <div className="text-3xl font-bold">72%</div>
                  <div className="text-sm text-muted-foreground mt-2">
                    Based on 6 different ML models
                  </div>
                </div>
                
                <div className="flex flex-col items-center justify-center p-4 border rounded-lg">
                  <div className="text-sm font-medium text-muted-foreground mb-2">
                    Volatility Forecast
                  </div>
                  <div className="text-3xl font-bold">Medium</div>
                  <div className="text-sm text-muted-foreground mt-2">
                    Expected 24h range: ±4.2%
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        {/* Model Performance Tab */}
        <TabsContent value="performance" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Model Accuracy Chart (simplified) */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">ML Model Accuracy</CardTitle>
                <CardDescription>
                  Historical prediction accuracy over time
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {mlPerformanceData.map((item) => (
                    <div key={item.date} className="flex items-center justify-between">
                      <div className="w-20">{item.date}</div>
                      <div className="flex-1 px-4">
                        <Progress value={item.accuracy} className="h-2" />
                      </div>
                      <div className="w-12 text-right">{item.accuracy}%</div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
            
            {/* Profit Factor Chart (simplified) */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Profit Factor</CardTitle>
                <CardDescription>
                  Ratio of gross profits to gross losses
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {mlPerformanceData.map((item) => (
                    <div key={item.date} className="flex items-center justify-between">
                      <div className="w-20">{item.date}</div>
                      <div className="flex-1 px-4">
                        <Progress 
                          value={item.profitFactor * 50} 
                          className="h-2 bg-muted"
                        />
                      </div>
                      <div className="w-12 text-right">{item.profitFactor}</div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
          
          {/* Performance Metrics Card */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Performance Metrics</CardTitle>
              <CardDescription>
                Key performance indicators for our ML trading models
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="flex flex-col p-4 border rounded-lg">
                  <div className="text-sm font-medium text-muted-foreground">
                    Win Rate
                  </div>
                  <div className="text-2xl font-bold mt-1">73.4%</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    <span className="text-green-500">↑2.1%</span> from last period
                  </div>
                </div>
                
                <div className="flex flex-col p-4 border rounded-lg">
                  <div className="text-sm font-medium text-muted-foreground">
                    Average Return
                  </div>
                  <div className="text-2xl font-bold mt-1">+4.2%</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    <span className="text-green-500">↑0.8%</span> from last period
                  </div>
                </div>
                
                <div className="flex flex-col p-4 border rounded-lg">
                  <div className="text-sm font-medium text-muted-foreground">
                    Max Drawdown
                  </div>
                  <div className="text-2xl font-bold mt-1">-12.6%</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    <span className="text-red-500">↓1.2%</span> from last period
                  </div>
                </div>
                
                <div className="flex flex-col p-4 border rounded-lg">
                  <div className="text-sm font-medium text-muted-foreground">
                    Sharpe Ratio
                  </div>
                  <div className="text-2xl font-bold mt-1">1.87</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    <span className="text-green-500">↑0.15</span> from last period
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        {/* Feature Analysis Tab */}
        <TabsContent value="features" className="space-y-4">
          {/* Feature Importance */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Feature Importance</CardTitle>
              <CardDescription>
                Key factors influencing ML model predictions
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {featureImportanceData.map((feature) => (
                  <div key={feature.name} className="space-y-1">
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">{feature.name}</span>
                      <span className="text-sm text-muted-foreground">{feature.value}%</span>
                    </div>
                    <Progress value={feature.value} className="h-2" />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
          
          {/* Feature Details Card */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Feature Details</CardTitle>
              <CardDescription>
                Technical analysis indicators used by our ML models
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="border rounded-lg p-4">
                  <div className="flex items-center mb-2">
                    <LineChart className="h-5 w-5 mr-2 text-primary" />
                    <h3 className="font-medium">Moving Averages</h3>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Trend indicators that smooth price data by calculating the average price over a specific period. Our models use combinations of SMA and EMA with different periods.
                  </p>
                </div>
                
                <div className="border rounded-lg p-4">
                  <div className="flex items-center mb-2">
                    <BarChart2 className="h-5 w-5 mr-2 text-primary" />
                    <h3 className="font-medium">Relative Strength Index (RSI)</h3>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Momentum oscillator that measures the speed and change of price movements on a scale from 0 to 100. Our models consider RSI values and divergences.
                  </p>
                </div>
                
                <div className="border rounded-lg p-4">
                  <div className="flex items-center mb-2">
                    <LineChart className="h-5 w-5 mr-2 text-primary" />
                    <h3 className="font-medium">MACD</h3>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Trend-following momentum indicator that shows the relationship between two moving averages of a security's price. Our models analyze MACD crossovers and divergences.
                  </p>
                </div>
                
                <div className="border rounded-lg p-4">
                  <div className="flex items-center mb-2">
                    <BarChart2 className="h-5 w-5 mr-2 text-primary" />
                    <h3 className="font-medium">Volume Indicators</h3>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Volume is a critical feature in our models, including OBV (On-Balance Volume), Volume Profile, and Volume-Weighted Average Price (VWAP).
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        {/* Backtest Results Tab */}
        <TabsContent value="backtests" className="space-y-4">
          {/* Backtest Results Cards */}
          {backtestResults.map((result, index) => (
            <Card key={index}>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="flex items-center">
                      <BrainCircuit className="h-5 w-5 mr-2" />
                      {result.name}
                    </CardTitle>
                    <CardDescription>{result.strategy}</CardDescription>
                  </div>
                  <Badge>
                    {result.timeframe} Timeframe
                  </Badge>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="flex flex-col">
                    <div className="text-sm font-medium text-muted-foreground">
                      Win Rate
                    </div>
                    <div className="text-2xl font-bold mt-1">
                      {result.winRate}%
                    </div>
                    <Progress value={result.winRate} className="h-1 mt-1" />
                  </div>
                  
                  <div className="flex flex-col">
                    <div className="text-sm font-medium text-muted-foreground">
                      Profit Factor
                    </div>
                    <div className="text-2xl font-bold mt-1">
                      {result.profitFactor}
                    </div>
                    <Progress value={result.profitFactor * 33} className="h-1 mt-1" />
                  </div>
                  
                  <div className="flex flex-col">
                    <div className="text-sm font-medium text-muted-foreground">
                      Max Drawdown
                    </div>
                    <div className="text-2xl font-bold mt-1">
                      {result.maxDrawdown}%
                    </div>
                    <Progress value={100 - result.maxDrawdown * 5} className="h-1 mt-1" />
                  </div>
                  
                  <div className="flex flex-col">
                    <div className="text-sm font-medium text-muted-foreground">
                      Sharpe Ratio
                    </div>
                    <div className="text-2xl font-bold mt-1">
                      {result.sharpeRatio}
                    </div>
                    <Progress value={result.sharpeRatio * 50} className="h-1 mt-1" />
                  </div>
                </div>
                
                <div className="mt-4 text-sm text-muted-foreground">
                  Total trades: {result.totalTrades} • Last updated: {result.lastUpdated}
                </div>
              </CardContent>
              <CardFooter>
                <Button variant="outline" className="w-full">View Detailed Report</Button>
              </CardFooter>
            </Card>
          ))}
        </TabsContent>
      </Tabs>
    </div>
  );
}
