"use client"

import React from 'react';
import dynamic from 'next/dynamic';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

// Dynamically import dashboard components
const ConfigurableDashboard = dynamic(() => import('@/components/dashboard/ConfigurableDashboard'), { loading: () => <p>Loading Configurable Dashboard...</p>, ssr: false });
const TradingDashboard = dynamic(() => import('@/components/dashboard/TradingDashboard'), { loading: () => <p>Loading Trading Dashboard...</p>, ssr: false });
const PortfolioDashboard = dynamic(() => import('@/components/dashboard/PortfolioDashboard').then(mod => mod.PortfolioDashboard), { loading: () => <p>Loading Portfolio Dashboard...</p>, ssr: false });
const RealTimeDataChart = dynamic(() => import('@/components/charts/RealTimeDataChart'), { loading: () => <p>Loading Chart...</p>, ssr: false });

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


