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
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Label } from '@/components/ui/label';
import { Separator } from '@/components/ui/separator';
import { 
  AlertCircle, 
  Bell, 
  KeyRound, 
  LayoutDashboard, 
  Moon, 
  Save, 
  Settings as SettingsIcon, 
  Sun, 
  User, 
  Wallet
} from 'lucide-react';

export default function SettingsPage() {
  const [selectedTheme, setSelectedTheme] = useState<'light' | 'dark' | 'system'>('system');
  const [defaultLeverage, setDefaultLeverage] = useState(10);
  const [showPortfolioValue, setShowPortfolioValue] = useState(true);
  const [compactMode, setCompactMode] = useState(false);
  const [defaultTimeframe, setDefaultTimeframe] = useState('1h');
  const [confirmOrders, setConfirmOrders] = useState(true);
  const [defaultOrderType, setDefaultOrderType] = useState('limit');
  const [defaultSymbol, setDefaultSymbol] = useState('BTCUSD');
  
  // Notification settings
  const [emailNotifications, setEmailNotifications] = useState(true);
  const [pushNotifications, setPushNotifications] = useState(true);
  const [orderExecuted, setOrderExecuted] = useState(true);
  const [priceAlerts, setPriceAlerts] = useState(true);
  const [fundingPayments, setFundingPayments] = useState(false);
  const [marketNews, setMarketNews] = useState(false);
  
  // Account settings
  const [email, setEmail] = useState('user@example.com');
  const [username, setUsername] = useState('trader123');
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  
  // API keys
  const [deltaExchangeApiKey, setDeltaExchangeApiKey] = useState('');
  const [deltaExchangeSecret, setDeltaExchangeSecret] = useState('');

  // Form submission handlers
  const onGeneralSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    console.log('General settings saved');
  };

  const onTradingSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    console.log('Trading settings saved');
  };

  const onApiKeysSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    console.log('API keys saved');
  };

  const onNotificationsSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    console.log('Notification settings saved');
  };

  const onAccountSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    console.log('Account settings saved');
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Settings</h1>
          <p className="text-muted-foreground">
            Manage your account settings and preferences
          </p>
        </div>
      </div>

      <Tabs defaultValue="general" className="w-full">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="general">General</TabsTrigger>
          <TabsTrigger value="trading">Trading</TabsTrigger>
          <TabsTrigger value="api-keys">API Keys</TabsTrigger>
          <TabsTrigger value="notifications">Notifications</TabsTrigger>
          <TabsTrigger value="account">Account</TabsTrigger>
        </TabsList>
        
        {/* General Settings Tab */}
        <TabsContent value="general">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <SettingsIcon className="h-5 w-5 mr-2" />
                General Settings
              </CardTitle>
              <CardDescription>
                Customize the application appearance and behavior
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={onGeneralSubmit} className="space-y-8">
                <div className="space-y-4">
                  <div>
                    <h3 className="text-lg font-medium">Appearance</h3>
                    <p className="text-sm text-muted-foreground">
                      Customize how the application looks
                    </p>
                  </div>
                  <Separator />
                  <div className="grid gap-4">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div 
                        className={`flex flex-col items-center justify-between rounded-md border-2 p-4 cursor-pointer ${selectedTheme === 'light' ? 'border-primary' : 'border-muted'}`}
                        onClick={() => setSelectedTheme('light')}
                      >
                        <Sun className="h-6 w-6 mb-2" />
                        <div className="text-center">
                          <h4 className="text-sm font-medium">Light</h4>
                          <p className="text-xs text-muted-foreground">Light mode theme</p>
                        </div>
                      </div>
                      <div 
                        className={`flex flex-col items-center justify-between rounded-md border-2 p-4 cursor-pointer ${selectedTheme === 'dark' ? 'border-primary' : 'border-muted'}`}
                        onClick={() => setSelectedTheme('dark')}
                      >
                        <Moon className="h-6 w-6 mb-2" />
                        <div className="text-center">
                          <h4 className="text-sm font-medium">Dark</h4>
                          <p className="text-xs text-muted-foreground">Dark mode theme</p>
                        </div>
                      </div>
                      <div 
                        className={`flex flex-col items-center justify-between rounded-md border-2 p-4 cursor-pointer ${selectedTheme === 'system' ? 'border-primary' : 'border-muted'}`}
                        onClick={() => setSelectedTheme('system')}
                      >
                        <div className="flex space-x-1 mb-2">
                          <Sun className="h-6 w-6" />
                          <Moon className="h-6 w-6" />
                        </div>
                        <div className="text-center">
                          <h4 className="text-sm font-medium">System</h4>
                          <p className="text-xs text-muted-foreground">Follow system theme</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <div>
                    <h3 className="text-lg font-medium">Dashboard Settings</h3>
                    <p className="text-sm text-muted-foreground">
                      Customize your dashboard layout and preferences
                    </p>
                  </div>
                  <Separator />
                  <div className="grid gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="defaultTimeframe">Default Chart Timeframe</Label>
                      <select 
                        id="defaultTimeframe"
                        className="w-full rounded-md border border-input bg-background px-3 py-2"
                        value={defaultTimeframe}
                        onChange={(e) => setDefaultTimeframe(e.target.value)}
                      >
                        <option value="1m">1 Minute</option>
                        <option value="5m">5 Minutes</option>
                        <option value="15m">15 Minutes</option>
                        <option value="30m">30 Minutes</option>
                        <option value="1h">1 Hour</option>
                        <option value="4h">4 Hours</option>
                        <option value="1d">1 Day</option>
                        <option value="1w">1 Week</option>
                      </select>
                      <p className="text-sm text-muted-foreground">
                        The default timeframe for charts when you open them
                      </p>
                    </div>
                    
                    <div className="flex flex-row items-center justify-between space-x-2 rounded-md border p-4">
                      <div className="space-y-0.5">
                        <Label className="text-base">
                          Show Portfolio Value
                        </Label>
                        <p className="text-sm text-muted-foreground">
                          Display your total portfolio value in the dashboard header
                        </p>
                      </div>
                      <div className="flex items-center space-x-2">
                        <input
                          type="checkbox"
                          id="showPortfolioValue"
                          checked={showPortfolioValue}
                          onChange={() => setShowPortfolioValue(!showPortfolioValue)}
                          className="h-4 w-4"
                        />
                        <Label htmlFor="showPortfolioValue" className="sr-only">
                          Show Portfolio Value
                        </Label>
                      </div>
                    </div>
                    
                    <div className="flex flex-row items-center justify-between space-x-2 rounded-md border p-4">
                      <div className="space-y-0.5">
                        <Label className="text-base">
                          Compact Mode
                        </Label>
                        <p className="text-sm text-muted-foreground">
                          Use a more compact layout to fit more information on screen
                        </p>
                      </div>
                      <div className="flex items-center space-x-2">
                        <input
                          type="checkbox"
                          id="compactMode"
                          checked={compactMode}
                          onChange={() => setCompactMode(!compactMode)}
                          className="h-4 w-4"
                        />
                        <Label htmlFor="compactMode" className="sr-only">
                          Compact Mode
                        </Label>
                      </div>
                    </div>
                  </div>
                </div>
                
                <Button type="submit">Save Changes</Button>
              </form>
            </CardContent>
          </Card>
        </TabsContent>
        
        {/* Trading Settings Tab */}
        <TabsContent value="trading">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <LayoutDashboard className="h-5 w-5 mr-2" />
                Trading Settings
              </CardTitle>
              <CardDescription>
                Configure your default trading parameters
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={onTradingSubmit} className="space-y-8">
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="defaultLeverage">Default Leverage: {defaultLeverage}x</Label>
                    <input 
                      type="range" 
                      id="defaultLeverage"
                      min="1" 
                      max="100" 
                      value={defaultLeverage}
                      onChange={(e) => setDefaultLeverage(parseInt(e.target.value))}
                      className="w-full h-2 bg-secondary rounded-md appearance-none cursor-pointer"
                    />
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>1x</span>
                      <span>25x</span>
                      <span>50x</span>
                      <span>75x</span>
                      <span>100x</span>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Default leverage for new positions
                    </p>
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="defaultSymbol">Default Trading Pair</Label>
                    <select 
                      id="defaultSymbol"
                      className="w-full rounded-md border border-input bg-background px-3 py-2"
                      value={defaultSymbol}
                      onChange={(e) => setDefaultSymbol(e.target.value)}
                    >
                      <option value="BTCUSD">BTC/USD</option>
                      <option value="ETHUSD">ETH/USD</option>
                      <option value="SOLUSD">SOL/USD</option>
                      <option value="ADAUSD">ADA/USD</option>
                      <option value="DOTUSD">DOT/USD</option>
                    </select>
                    <p className="text-sm text-muted-foreground">
                      Default trading pair to show when opening the platform
                    </p>
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="defaultOrderType">Default Order Type</Label>
                    <select 
                      id="defaultOrderType"
                      className="w-full rounded-md border border-input bg-background px-3 py-2"
                      value={defaultOrderType}
                      onChange={(e) => setDefaultOrderType(e.target.value)}
                    >
                      <option value="market">Market</option>
                      <option value="limit">Limit</option>
                      <option value="stop">Stop</option>
                      <option value="take_profit">Take Profit</option>
                    </select>
                    <p className="text-sm text-muted-foreground">
                      Default order type when creating new orders
                    </p>
                  </div>
                  
                  <div className="flex flex-row items-center justify-between space-x-2 rounded-md border p-4">
                    <div className="space-y-0.5">
                      <Label className="text-base">
                        Confirm Orders
                      </Label>
                      <p className="text-sm text-muted-foreground">
                        Show confirmation dialog before placing orders
                      </p>
                    </div>
                    <div className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        id="confirmOrders"
                        checked={confirmOrders}
                        onChange={() => setConfirmOrders(!confirmOrders)}
                        className="h-4 w-4"
                      />
                      <Label htmlFor="confirmOrders" className="sr-only">
                        Confirm Orders
                      </Label>
                    </div>
                  </div>
                </div>
                
                <Button type="submit">Save Changes</Button>
              </form>
            </CardContent>
          </Card>
        </TabsContent>
        
        {/* API Keys Tab */}
        <TabsContent value="api-keys">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <KeyRound className="h-5 w-5 mr-2" />
                API Keys
              </CardTitle>
              <CardDescription>
                Manage your exchange API keys for automated trading
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={onApiKeysSubmit} className="space-y-8">
                <div className="space-y-4">
                  <div className="rounded-md border p-4">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center">
                        <img src="/delta-exchange-logo.png" alt="Delta Exchange" className="h-8 w-8 mr-2" />
                        <div>
                          <h3 className="text-lg font-medium">Delta Exchange</h3>
                          <p className="text-sm text-muted-foreground">Cryptocurrency derivatives exchange</p>
                        </div>
                      </div>
                      <Badge variant="outline">Connected</Badge>
                    </div>
                    
                    <div className="space-y-4">
                      <div className="space-y-2">
                        <Label htmlFor="deltaExchangeApiKey">API Key</Label>
                        <Input 
                          id="deltaExchangeApiKey"
                          type="password" 
                          placeholder="Enter your API key"
                          value={deltaExchangeApiKey}
                          onChange={(e) => setDeltaExchangeApiKey(e.target.value)} 
                        />
                        <p className="text-sm text-muted-foreground">
                          Your Delta Exchange API key (stored securely)
                        </p>
                      </div>
                      
                      <div className="space-y-2">
                        <Label htmlFor="deltaExchangeSecret">API Secret</Label>
                        <Input 
                          id="deltaExchangeSecret"
                          type="password" 
                          placeholder="Enter your API secret"
                          value={deltaExchangeSecret}
                          onChange={(e) => setDeltaExchangeSecret(e.target.value)} 
                        />
                        <p className="text-sm text-muted-foreground">
                          Your Delta Exchange API secret (stored securely)
                        </p>
                      </div>
                      
                      <div className="bg-muted/50 p-3 rounded-md">
                        <div className="flex items-center mb-2">
                          <AlertCircle className="h-5 w-5 text-muted-foreground mr-2" />
                          <h4 className="text-sm font-medium">Required Permissions</h4>
                        </div>
                        <ul className="ml-7 text-sm text-muted-foreground list-disc space-y-1">
                          <li>Read account information</li>
                          <li>Read balances and positions</li>
                          <li>Place and cancel orders</li>
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>
                
                <Button type="submit">Save API Keys</Button>
              </form>
            </CardContent>
          </Card>
        </TabsContent>
        
        {/* Notifications Tab */}
        <TabsContent value="notifications">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Bell className="h-5 w-5 mr-2" />
                Notification Settings
              </CardTitle>
              <CardDescription>
                Configure how and when you receive notifications
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={onNotificationsSubmit} className="space-y-8">
                <div className="space-y-4">
                  <div>
                    <h3 className="text-lg font-medium">Notification Channels</h3>
                    <p className="text-sm text-muted-foreground">
                      Choose how you want to receive notifications
                    </p>
                  </div>
                  <Separator />
                  <div className="space-y-4">
                    <div className="flex flex-row items-center justify-between space-x-2 rounded-md border p-4">
                      <div className="space-y-0.5">
                        <Label className="text-base">
                          Email Notifications
                        </Label>
                        <p className="text-sm text-muted-foreground">
                          Receive notifications via email
                        </p>
                      </div>
                      <div className="flex items-center space-x-2">
                        <input
                          type="checkbox"
                          id="emailNotifications"
                          checked={emailNotifications}
                          onChange={() => setEmailNotifications(!emailNotifications)}
                          className="h-4 w-4"
                        />
                        <Label htmlFor="emailNotifications" className="sr-only">
                          Email Notifications
                        </Label>
                      </div>
                    </div>
                    
                    <div className="flex flex-row items-center justify-between space-x-2 rounded-md border p-4">
                      <div className="space-y-0.5">
                        <Label className="text-base">
                          Push Notifications
                        </Label>
                        <p className="text-sm text-muted-foreground">
                          Receive browser push notifications
                        </p>
                      </div>
                      <div className="flex items-center space-x-2">
                        <input
                          type="checkbox"
                          id="pushNotifications"
                          checked={pushNotifications}
                          onChange={() => setPushNotifications(!pushNotifications)}
                          className="h-4 w-4"
                        />
                        <Label htmlFor="pushNotifications" className="sr-only">
                          Push Notifications
                        </Label>
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="text-lg font-medium mt-6">Notification Types</h3>
                    <p className="text-sm text-muted-foreground">
                      Choose which events trigger notifications
                    </p>
                  </div>
                  <Separator />
                  <div className="space-y-4">
                    <div className="flex flex-row items-center justify-between space-x-2 rounded-md border p-4">
                      <div className="space-y-0.5">
                        <Label className="text-base">
                          Order Executed
                        </Label>
                        <p className="text-sm text-muted-foreground">
                          Notify when orders are filled or executed
                        </p>
                      </div>
                      <div className="flex items-center space-x-2">
                        <input
                          type="checkbox"
                          id="orderExecuted"
                          checked={orderExecuted}
                          onChange={() => setOrderExecuted(!orderExecuted)}
                          className="h-4 w-4"
                        />
                        <Label htmlFor="orderExecuted" className="sr-only">
                          Order Executed
                        </Label>
                      </div>
                    </div>
                    
                    <div className="flex flex-row items-center justify-between space-x-2 rounded-md border p-4">
                      <div className="space-y-0.5">
                        <Label className="text-base">
                          Price Alerts
                        </Label>
                        <p className="text-sm text-muted-foreground">
                          Notify for price alerts you've set
                        </p>
                      </div>
                      <div className="flex items-center space-x-2">
                        <input
                          type="checkbox"
                          id="priceAlerts"
                          checked={priceAlerts}
                          onChange={() => setPriceAlerts(!priceAlerts)}
                          className="h-4 w-4"
                        />
                        <Label htmlFor="priceAlerts" className="sr-only">
                          Price Alerts
                        </Label>
                      </div>
                    </div>
                    
                    <div className="flex flex-row items-center justify-between space-x-2 rounded-md border p-4">
                      <div className="space-y-0.5">
                        <Label className="text-base">
                          Funding Payments
                        </Label>
                        <p className="text-sm text-muted-foreground">
                          Notify before and after funding payments for perpetual contracts
                        </p>
                      </div>
                      <div className="flex items-center space-x-2">
                        <input
                          type="checkbox"
                          id="fundingPayments"
                          checked={fundingPayments}
                          onChange={() => setFundingPayments(!fundingPayments)}
                          className="h-4 w-4"
                        />
                        <Label htmlFor="fundingPayments" className="sr-only">
                          Funding Payments
                        </Label>
                      </div>
                    </div>
                    
                    <div className="flex flex-row items-center justify-between space-x-2 rounded-md border p-4">
                      <div className="space-y-0.5">
                        <Label className="text-base">
                          Market News
                        </Label>
                        <p className="text-sm text-muted-foreground">
                          Notify for important market news and events
                        </p>
                      </div>
                      <div className="flex items-center space-x-2">
                        <input
                          type="checkbox"
                          id="marketNews"
                          checked={marketNews}
                          onChange={() => setMarketNews(!marketNews)}
                          className="h-4 w-4"
                        />
                        <Label htmlFor="marketNews" className="sr-only">
                          Market News
                        </Label>
                      </div>
                    </div>
                  </div>
                </div>
                
                <Button type="submit">Save Notification Settings</Button>
              </form>
            </CardContent>
          </Card>
        </TabsContent>
        
        {/* Account Tab */}
        <TabsContent value="account">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <User className="h-5 w-5 mr-2" />
                Account Settings
              </CardTitle>
              <CardDescription>
                Manage your account information and security settings
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={onAccountSubmit} className="space-y-8">
                <div className="space-y-4">
                  <div>
                    <h3 className="text-lg font-medium">Profile Information</h3>
                    <p className="text-sm text-muted-foreground">
                      Update your account details
                    </p>
                  </div>
                  <Separator />
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="email">Email</Label>
                      <Input 
                        id="email"
                        type="email" 
                        placeholder="your@email.com"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)} 
                      />
                      <p className="text-sm text-muted-foreground">
                        This is your registered email address
                      </p>
                    </div>
                    
                    <div className="space-y-2">
                      <Label htmlFor="username">Username</Label>
                      <Input 
                        id="username"
                        placeholder="Username"
                        value={username}
                        onChange={(e) => setUsername(e.target.value)} 
                      />
                      <p className="text-sm text-muted-foreground">
                        Your public username
                      </p>
                    </div>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <div>
                    <h3 className="text-lg font-medium">Security</h3>
                    <p className="text-sm text-muted-foreground">
                      Update your password and security settings
                    </p>
                  </div>
                  <Separator />
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="currentPassword">Current Password</Label>
                      <Input 
                        id="currentPassword"
                        type="password" 
                        placeholder="••••••••"
                        value={currentPassword}
                        onChange={(e) => setCurrentPassword(e.target.value)} 
                      />
                      <p className="text-sm text-muted-foreground">
                        Enter your current password to make changes
                      </p>
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="newPassword">New Password</Label>
                        <Input 
                          id="newPassword"
                          type="password" 
                          placeholder="••••••••"
                          value={newPassword}
                          onChange={(e) => setNewPassword(e.target.value)} 
                        />
                        <p className="text-sm text-muted-foreground">
                          Enter a new password
                        </p>
                      </div>
                      
                      <div className="space-y-2">
                        <Label htmlFor="confirmPassword">Confirm Password</Label>
                        <Input 
                          id="confirmPassword"
                          type="password" 
                          placeholder="••••••••"
                          value={confirmPassword}
                          onChange={(e) => setConfirmPassword(e.target.value)} 
                        />
                        <p className="text-sm text-muted-foreground">
                          Confirm your new password
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
                
                <Button type="submit">Save Account Settings</Button>
              </form>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
} 