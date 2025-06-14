'use client'

import { useState, useEffect } from 'react'
import { Bot, Activity, Play, Square, Plus, Trash2, Settings, BarChart3 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  Dialog, 
  DialogContent, 
  DialogDescription, 
  DialogFooter, 
  DialogHeader, 
  DialogTitle, 
  DialogTrigger 
} from '@/components/ui/dialog'
import { Label } from '@/components/ui/label'
import { Input } from '@/components/ui/input'
import { Separator } from '@/components/ui/separator'
import { 
  Table, 
  TableBody, 
  TableCaption, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from '@/components/ui/table'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'

interface BotMetrics {
  tradesExecuted: number;
  profitLoss: number;
  successRate: number;
  latency: number;
}

interface BotStatus {
  health: string;
  lastUpdate: string;
  metrics: BotMetrics;
  activePositions: number;
  errors: string[];
}

interface Bot {
  id: string;
  name: string;
  symbol: string;
  strategy: string;
  timeframe: string;
  isActive: boolean;
  status: BotStatus;
}

// Mock data for initial development
const mockBots: Bot[] = [
  {
    id: '1',
    name: 'BTC Trend Follower',
    symbol: 'BTC-USDT',
    strategy: 'Trend Following',
    timeframe: '1h',
    isActive: true,
    status: {
      health: 'excellent',
      lastUpdate: new Date().toISOString(),
      metrics: {
        tradesExecuted: 18,
        profitLoss: 4.2,
        successRate: 72.4,
        latency: 142
      },
      activePositions: 1,
      errors: []
    }
  },
  {
    id: '2',
    name: 'ETH Swing Trader',
    symbol: 'ETH-USDT',
    strategy: 'Mean Reversion',
    timeframe: '4h',
    isActive: false,
    status: {
      health: 'good',
      lastUpdate: new Date(Date.now() - 24*60*60*1000).toISOString(),
      metrics: {
        tradesExecuted: 12,
        profitLoss: -0.8,
        successRate: 58.3,
        latency: 165
      },
      activePositions: 0,
      errors: []
    }
  },
  {
    id: '3',
    name: 'SOL Scalper',
    symbol: 'SOL-USDT',
    strategy: 'Scalping',
    timeframe: '15m',
    isActive: false,
    status: {
      health: 'unknown',
      lastUpdate: new Date(Date.now() - 5*24*60*60*1000).toISOString(),
      metrics: {
        tradesExecuted: 0,
        profitLoss: 0,
        successRate: 0,
        latency: 0
      },
      activePositions: 0,
      errors: []
    }
  }
];

export default function BotManagementPage() {
  const [bots, setBots] = useState<Bot[]>(mockBots);
  const [selectedBot, setSelectedBot] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [newBotData, setNewBotData] = useState({
    name: '',
    symbol: '',
    strategy: '',
    timeframe: ''
  });

  // Fetch bots from the API
  useEffect(() => {
    const fetchBots = async () => {
      setIsLoading(true);
      try {
        // Try to fetch real data first
        try {
          const response = await fetch('http://localhost:3001/api/bots');
          if (response.ok) {
            const data = await response.json();
            if (data && Array.isArray(data)) {
              setBots(data);
            }
          }
        } catch (error) {
          console.log('Using mock data due to API error:', error);
          // Already using mock data from initial state
        }
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchBots();
  }, []);

  const handleCreateBot = async () => {
    setIsLoading(true);
    try {
      // API call would be here
      // For now, just add to the mock data
      const newBot: Bot = {
        id: `${bots.length + 1}`,
        name: newBotData.name,
        symbol: newBotData.symbol,
        strategy: newBotData.strategy,
        timeframe: newBotData.timeframe,
        isActive: false,
        status: {
          health: 'unknown',
          lastUpdate: new Date().toISOString(),
          metrics: {
            tradesExecuted: 0,
            profitLoss: 0,
            successRate: 0,
            latency: 0
          },
          activePositions: 0,
          errors: []
        }
      };
      setBots([...bots, newBot]);
      setCreateDialogOpen(false);
      setNewBotData({
        name: '',
        symbol: '',
        strategy: '',
        timeframe: ''
      });
    } catch (error) {
      console.error('Error creating bot:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleStartBot = async (botId: string) => {
    setIsLoading(true);
    try {
      // API call would be here
      // For now, just update the mock data
      setBots(bots.map(bot => 
        bot.id === botId 
          ? { ...bot, isActive: true } 
          : bot
      ));
    } catch (error) {
      console.error('Error starting bot:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleStopBot = async (botId: string) => {
    setIsLoading(true);
    try {
      // API call would be here
      // For now, just update the mock data
      setBots(bots.map(bot => 
        bot.id === botId 
          ? { ...bot, isActive: false } 
          : bot
      ));
    } catch (error) {
      console.error('Error stopping bot:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteBot = async (botId: string) => {
    setIsLoading(true);
    try {
      // API call would be here
      // For now, just update the mock data
      setBots(bots.filter(bot => bot.id !== botId));
      if (selectedBot === botId) {
        setSelectedBot(null);
      }
    } catch (error) {
      console.error('Error deleting bot:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const getSelectedBot = () => {
    return bots.find(bot => bot.id === selectedBot) || null;
  };

  const renderHealthBadge = (health: string) => {
    switch (health) {
      case 'excellent':
        return <Badge variant="outline" className="bg-green-50 text-green-600">Excellent</Badge>;
      case 'good':
        return <Badge variant="outline" className="bg-blue-50 text-blue-600">Good</Badge>;
      case 'degraded':
        return <Badge variant="outline" className="bg-yellow-50 text-yellow-600">Degraded</Badge>;
      case 'poor':
        return <Badge variant="outline" className="bg-orange-50 text-orange-600">Poor</Badge>;
      case 'critical':
        return <Badge variant="outline" className="bg-red-50 text-red-600">Critical</Badge>;
      default:
        return <Badge variant="outline">Unknown</Badge>;
    }
  };

  return (
    <div className="container mx-auto py-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Bot Management</h1>
          <p className="text-muted-foreground">Create, monitor, and control your trading bots</p>
        </div>
        <Dialog open={createDialogOpen} onOpenChange={setCreateDialogOpen}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="mr-2 h-4 w-4" />
              Create Bot
            </Button>
          </DialogTrigger>
          <DialogContent className="sm:max-w-[500px]">
            <DialogHeader>
              <DialogTitle>Create New Trading Bot</DialogTitle>
              <DialogDescription>
                Configure your new trading bot. After creation, you can set additional parameters.
              </DialogDescription>
            </DialogHeader>
            <div className="grid gap-4 py-4">
              <div className="grid gap-2">
                <Label htmlFor="name">Bot Name</Label>
                <Input
                  id="name"
                  placeholder="e.g., BTC Trend Follower"
                  value={newBotData.name}
                  onChange={(e) => setNewBotData({ ...newBotData, name: e.target.value })}
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="symbol">Trading Symbol</Label>
                <Input
                  id="symbol"
                  placeholder="e.g., BTC-USDT"
                  value={newBotData.symbol}
                  onChange={(e) => setNewBotData({ ...newBotData, symbol: e.target.value })}
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="strategy">Trading Strategy</Label>
                <Select
                  value={newBotData.strategy}
                  onValueChange={(value) => setNewBotData({ ...newBotData, strategy: value })}
                >
                  <SelectTrigger id="strategy">
                    <SelectValue placeholder="Select a strategy" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Trend Following">Trend Following</SelectItem>
                    <SelectItem value="Mean Reversion">Mean Reversion</SelectItem>
                    <SelectItem value="Breakout">Breakout</SelectItem>
                    <SelectItem value="Scalping">Scalping</SelectItem>
                    <SelectItem value="Momentum">Momentum</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="grid gap-2">
                <Label htmlFor="timeframe">Timeframe</Label>
                <Select
                  value={newBotData.timeframe}
                  onValueChange={(value) => setNewBotData({ ...newBotData, timeframe: value })}
                >
                  <SelectTrigger id="timeframe">
                    <SelectValue placeholder="Select a timeframe" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="1m">1 minute</SelectItem>
                    <SelectItem value="5m">5 minutes</SelectItem>
                    <SelectItem value="15m">15 minutes</SelectItem>
                    <SelectItem value="30m">30 minutes</SelectItem>
                    <SelectItem value="1h">1 hour</SelectItem>
                    <SelectItem value="4h">4 hours</SelectItem>
                    <SelectItem value="1d">1 day</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            <DialogFooter>
              <Button type="submit" onClick={handleCreateBot} disabled={isLoading}>
                Create Bot
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {bots.length === 0 ? (
        <div className="flex flex-col items-center justify-center p-12 text-center">
          <Bot className="h-16 w-16 text-muted-foreground mb-4" />
          <h3 className="font-semibold text-lg">No Bots Configured</h3>
          <p className="text-muted-foreground mb-6 max-w-md">
            You haven't set up any trading bots yet. Get started by creating your first bot.
          </p>
          <Button onClick={() => setCreateDialogOpen(true)}>
            <Plus className="mr-2 h-4 w-4" />
            Create Bot
          </Button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="md:col-span-1">
            <Card>
              <CardHeader>
                <CardTitle>Your Bots</CardTitle>
                <CardDescription>
                  {bots.filter(b => b.isActive).length} of {bots.length} bots active
                </CardDescription>
              </CardHeader>
              <CardContent className="p-0">
                <div className="space-y-1">
                  {bots.map((bot) => (
                    <div
                      key={bot.id}
                      className={`flex items-center justify-between p-3 cursor-pointer ${
                        selectedBot === bot.id ? 'bg-muted' : 'hover:bg-muted/50'
                      }`}
                      onClick={() => setSelectedBot(bot.id)}
                    >
                      <div className="flex items-center">
                        <div className={`w-2 h-2 rounded-full mr-3 ${
                          bot.isActive ? 'bg-green-500' : 'bg-gray-300'
                        }`} />
                        <div>
                          <div className="font-medium">{bot.name}</div>
                          <div className="text-xs text-muted-foreground">
                            {bot.symbol} • {bot.strategy}
                          </div>
                        </div>
                      </div>
                      {bot.isActive ? (
                        <Button 
                          variant="ghost" 
                          size="icon"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleStopBot(bot.id);
                          }}
                        >
                          <Square className="h-4 w-4" />
                        </Button>
                      ) : (
                        <Button 
                          variant="ghost" 
                          size="icon"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleStartBot(bot.id);
                          }}
                        >
                          <Play className="h-4 w-4" />
                        </Button>
                      )}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="md:col-span-2">
            {selectedBot ? (
              <Tabs defaultValue="overview">
                <TabsList className="mb-4">
                  <TabsTrigger value="overview">Overview</TabsTrigger>
                  <TabsTrigger value="performance">Performance</TabsTrigger>
                  <TabsTrigger value="settings">Settings</TabsTrigger>
                </TabsList>
                
                <TabsContent value="overview">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                    <Card>
                      <CardHeader>
                        <CardTitle>Bot Status</CardTitle>
                      </CardHeader>
                      <CardContent>
                        {getSelectedBot() && (
                          <div className="space-y-4">
                            <div className="grid grid-cols-2 gap-4">
                              <div>
                                <div className="text-sm font-medium text-muted-foreground">Status</div>
                                <div className="text-xl font-semibold">
                                  {getSelectedBot()?.isActive ? 'Running' : 'Stopped'}
                                </div>
                              </div>
                              <div>
                                <div className="text-sm font-medium text-muted-foreground">Health</div>
                                <div>{renderHealthBadge(getSelectedBot()?.status.health || 'unknown')}</div>
                              </div>
                            </div>
                            
                            <Separator />
                            
                            <div>
                              <div className="text-sm font-medium text-muted-foreground mb-2">Bot Information</div>
                              <div className="grid grid-cols-2 gap-y-2 text-sm">
                                <div>Trading Pair:</div>
                                <div className="font-medium">{getSelectedBot()?.symbol}</div>
                                
                                <div>Strategy:</div>
                                <div className="font-medium">{getSelectedBot()?.strategy}</div>
                                
                                <div>Timeframe:</div>
                                <div className="font-medium">{getSelectedBot()?.timeframe}</div>
                                
                                <div>Active Positions:</div>
                                <div className="font-medium">{getSelectedBot()?.status.activePositions}</div>
                              </div>
                            </div>
                            
                            <Separator />
                            
                            <div>
                              <div className="text-sm font-medium text-muted-foreground mb-2">Performance</div>
                              <div className="grid grid-cols-2 gap-y-2 text-sm">
                                <div>Win Rate:</div>
                                <div className="font-medium">{getSelectedBot()?.status.metrics.successRate.toFixed(1)}%</div>
                                
                                <div>P&L:</div>
                                <div className={`font-medium ${
                                  (getSelectedBot()?.status.metrics.profitLoss || 0) > 0 
                                    ? 'text-green-500' 
                                    : (getSelectedBot()?.status.metrics.profitLoss || 0) < 0 
                                      ? 'text-red-500' 
                                      : ''
                                }`}>
                                  {(getSelectedBot()?.status.metrics.profitLoss || 0) > 0 ? '+' : ''}
                                  {getSelectedBot()?.status.metrics.profitLoss.toFixed(2)}%
                                </div>
                                
                                <div>Trades Executed:</div>
                                <div className="font-medium">{getSelectedBot()?.status.metrics.tradesExecuted}</div>
                                
                                <div>Avg. Latency:</div>
                                <div className="font-medium">{getSelectedBot()?.status.metrics.latency} ms</div>
                              </div>
                            </div>
                          </div>
                        )}
                      </CardContent>
                    </Card>
                    
                    <Card>
                      <CardHeader>
                        <CardTitle>Bot Actions</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <div className="grid grid-cols-2 gap-2">
                          {getSelectedBot()?.isActive ? (
                            <Button 
                              variant="destructive" 
                              onClick={() => handleStopBot(selectedBot)}
                              disabled={isLoading}
                            >
                              <Square className="mr-2 h-4 w-4" />
                              Stop Bot
                            </Button>
                          ) : (
                            <Button
                              variant="default"
                              onClick={() => handleStartBot(selectedBot)}
                              disabled={isLoading}
                            >
                              <Play className="mr-2 h-4 w-4" />
                              Start Bot
                            </Button>
                          )}
                          
                          <Button variant="outline">
                            <BarChart3 className="mr-2 h-4 w-4" />
                            Backtest
                          </Button>
                        </div>
                        
                        <Separator />
                        
                        <Button 
                          variant="destructive" 
                          className="w-full"
                          onClick={() => handleDeleteBot(selectedBot)}
                          disabled={isLoading}
                        >
                          <Trash2 className="mr-2 h-4 w-4" />
                          Delete Bot
                        </Button>
                      </CardContent>
                    </Card>
                  </div>
                  
                  {/* Trading Activity */}
                  <Card>
                    <CardHeader>
                      <CardTitle>Recent Activity</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Time</TableHead>
                            <TableHead>Action</TableHead>
                            <TableHead>Symbol</TableHead>
                            <TableHead>Price</TableHead>
                            <TableHead>Status</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {getSelectedBot()?.isActive ? (
                            <>
                              <TableRow>
                                <TableCell className="font-medium">10:23:12</TableCell>
                                <TableCell>BUY</TableCell>
                                <TableCell>{getSelectedBot()?.symbol}</TableCell>
                                <TableCell>$26,785.50</TableCell>
                                <TableCell><Badge variant="outline" className="bg-green-50 text-green-600">Executed</Badge></TableCell>
                              </TableRow>
                              <TableRow>
                                <TableCell className="font-medium">09:45:37</TableCell>
                                <TableCell>SELL</TableCell>
                                <TableCell>{getSelectedBot()?.symbol}</TableCell>
                                <TableCell>$26,680.25</TableCell>
                                <TableCell><Badge variant="outline" className="bg-green-50 text-green-600">Executed</Badge></TableCell>
                              </TableRow>
                            </>
                          ) : (
                            <TableRow>
                              <TableCell colSpan={5} className="text-center text-muted-foreground py-6">
                                No recent activity - bot is not running
                              </TableCell>
                            </TableRow>
                          )}
                        </TableBody>
                      </Table>
                    </CardContent>
                  </Card>
                </TabsContent>
                
                <TabsContent value="performance">
                  <div className="space-y-6">
                    <Card>
                      <CardHeader>
                        <CardTitle>Performance Metrics</CardTitle>
                      </CardHeader>
                      <CardContent className="h-[350px] flex items-center justify-center text-muted-foreground">
                        <div className="text-center">
                          <BarChart3 className="h-16 w-16 mx-auto mb-4 opacity-30" />
                          <p>Performance chart would appear here</p>
                          <p className="text-sm">Tracking {getSelectedBot()?.status.metrics.tradesExecuted} trades</p>
                        </div>
                      </CardContent>
                    </Card>
                    
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                      <Card>
                        <CardHeader className="pb-2">
                          <CardTitle className="text-sm font-medium">Win Rate</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="text-2xl font-bold">
                            {getSelectedBot()?.status.metrics.successRate.toFixed(1)}%
                          </div>
                          <p className="text-xs text-muted-foreground">
                            Based on {getSelectedBot()?.status.metrics.tradesExecuted} trades
                          </p>
                        </CardContent>
                      </Card>
                      
                      <Card>
                        <CardHeader className="pb-2">
                          <CardTitle className="text-sm font-medium">Profit/Loss</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className={`text-2xl font-bold ${
                            (getSelectedBot()?.status.metrics.profitLoss || 0) > 0 
                              ? 'text-green-500' 
                              : (getSelectedBot()?.status.metrics.profitLoss || 0) < 0 
                                ? 'text-red-500' 
                                : ''
                          }`}>
                            {(getSelectedBot()?.status.metrics.profitLoss || 0) > 0 ? '+' : ''}
                            {getSelectedBot()?.status.metrics.profitLoss.toFixed(2)}%
                          </div>
                          <p className="text-xs text-muted-foreground">
                            Last 30 days
                          </p>
                        </CardContent>
                      </Card>
                      
                      <Card>
                        <CardHeader className="pb-2">
                          <CardTitle className="text-sm font-medium">Avg. Latency</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="text-2xl font-bold">
                            {getSelectedBot()?.status.metrics.latency} ms
                          </div>
                          <p className="text-xs text-muted-foreground">
                            Signal to execution
                          </p>
                        </CardContent>
                      </Card>
                    </div>
                  </div>
                </TabsContent>
                
                <TabsContent value="settings">
                  <Card>
                    <CardHeader>
                      <CardTitle>Bot Configuration</CardTitle>
                      <CardDescription>
                        Adjust settings for {getSelectedBot()?.name}
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        <div className="grid gap-2">
                          <Label htmlFor="bot-name">Bot Name</Label>
                          <Input id="bot-name" defaultValue={getSelectedBot()?.name} />
                        </div>
                        
                        <div className="grid gap-2">
                          <Label htmlFor="trading-symbol">Trading Symbol</Label>
                          <Input id="trading-symbol" defaultValue={getSelectedBot()?.symbol} />
                        </div>
                        
                        <div className="grid gap-2">
                          <Label htmlFor="strategy-select">Trading Strategy</Label>
                          <Select defaultValue={getSelectedBot()?.strategy}>
                            <SelectTrigger id="strategy-select">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="Trend Following">Trend Following</SelectItem>
                              <SelectItem value="Mean Reversion">Mean Reversion</SelectItem>
                              <SelectItem value="Breakout">Breakout</SelectItem>
                              <SelectItem value="Scalping">Scalping</SelectItem>
                              <SelectItem value="Momentum">Momentum</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                        
                        <div className="grid gap-2">
                          <Label htmlFor="timeframe-select">Timeframe</Label>
                          <Select defaultValue={getSelectedBot()?.timeframe}>
                            <SelectTrigger id="timeframe-select">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="1m">1 minute</SelectItem>
                              <SelectItem value="5m">5 minutes</SelectItem>
                              <SelectItem value="15m">15 minutes</SelectItem>
                              <SelectItem value="30m">30 minutes</SelectItem>
                              <SelectItem value="1h">1 hour</SelectItem>
                              <SelectItem value="4h">4 hours</SelectItem>
                              <SelectItem value="1d">1 day</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                        
                        <Separator />
                        
                        <div className="space-y-2">
                          <h3 className="text-sm font-medium">Advanced Settings</h3>
                          <p className="text-sm text-muted-foreground">
                            Configure risk parameters and trading rules
                          </p>
                          
                          <div className="mt-4 grid gap-4">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                              <div className="grid gap-2">
                                <Label htmlFor="risk-per-trade">Risk Per Trade (%)</Label>
                                <Input id="risk-per-trade" type="number" defaultValue="1.0" />
                              </div>
                              
                              <div className="grid gap-2">
                                <Label htmlFor="max-positions">Max Open Positions</Label>
                                <Input id="max-positions" type="number" defaultValue="3" />
                              </div>
                            </div>
                            
                            <div className="grid gap-2">
                              <Label htmlFor="stop-loss">Default Stop Loss (%)</Label>
                              <Input id="stop-loss" type="number" defaultValue="2.0" />
                            </div>
                            
                            <div className="grid gap-2">
                              <Label htmlFor="take-profit">Default Take Profit (%)</Label>
                              <Input id="take-profit" type="number" defaultValue="3.0" />
                            </div>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                    <CardFooter>
                      <Button className="ml-auto">Save Changes</Button>
                    </CardFooter>
                  </Card>
                </TabsContent>
              </Tabs>
            ) : (
              <div className="flex flex-col items-center justify-center p-12 text-center border rounded-lg">
                <h3 className="font-semibold text-lg">Select a Bot</h3>
                <p className="text-muted-foreground mb-6">
                  Choose a bot from the list to view its details and performance.
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
} 