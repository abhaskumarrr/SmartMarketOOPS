"use client"

import React from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import ConfigurableDashboard from '@/components/dashboard/ConfigurableDashboard';
import TradingDashboard from '@/components/dashboard/TradingDashboard';
import { PortfolioDashboard } from '@/components/dashboard/PortfolioDashboard';
import RealTimeDataChart from '@/components/charts/RealTimeDataChart';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

export default function DashboardPage() {
  return (
    <div className="flex flex-col space-y-4">
      <h1 className="text-2xl font-bold">Dashboard</h1>
      
      <Tabs defaultValue="configurable" className="w-full">
        <TabsList className="mb-4">
          <TabsTrigger value="configurable">Configurable</TabsTrigger>
          <TabsTrigger value="trading">Trading</TabsTrigger>
          <TabsTrigger value="portfolio">Portfolio</TabsTrigger>
        </TabsList>
        
        <TabsContent value="configurable" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="col-span-1 md:col-span-3">
              <RealTimeDataChart 
                symbol="BTCUSDT" 
                title="Bitcoin Price" 
                height={250} 
                dataKeys={['price', 'volume']}
              />
            </div>
          </div>
          
          <ConfigurableDashboard />
        </TabsContent>
        
        <TabsContent value="trading" className="space-y-4">
          <TradingDashboard />
        </TabsContent>
        
        <TabsContent value="portfolio" className="space-y-4">
          <PortfolioDashboard />
        </TabsContent>
      </Tabs>
    </div>
  );
}


