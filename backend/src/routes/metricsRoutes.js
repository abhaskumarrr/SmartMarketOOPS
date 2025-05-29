/**
 * Performance Metrics API Routes
 * Provides endpoints for fetching trading metrics and performance data
 */

const express = require('express');
const router = express.Router();
const prisma = require('../utils/prismaClient');
const { auth } = require('../middleware/auth');

// Get daily performance data
router.get('/metrics/daily', auth, async (req, res) => {
  try {
    // In a real implementation, fetch from database
    // For now, returning placeholder data as we transition from mock to real
    const dailyPerfData = [
      { date: '2025-05-01', profit: 120, trades: 5, winRate: 60 },
      { date: '2025-05-02', profit: -50, trades: 3, winRate: 33 },
      { date: '2025-05-03', profit: 200, trades: 7, winRate: 71 },
      { date: '2025-05-04', profit: 75, trades: 4, winRate: 50 },
      { date: '2025-05-05', profit: -30, trades: 2, winRate: 0 },
      { date: '2025-05-06', profit: 300, trades: 8, winRate: 75 },
      { date: '2025-05-07', profit: -80, trades: 5, winRate: 40 },
      { date: '2025-05-08', profit: 220, trades: 6, winRate: 67 },
      { date: '2025-05-09', profit: 150, trades: 4, winRate: 75 },
      { date: '2025-05-10', profit: -20, trades: 3, winRate: 33 },
      { date: '2025-05-11', profit: 180, trades: 5, winRate: 80 },
      { date: '2025-05-12', profit: 260, trades: 7, winRate: 71 },
      { date: '2025-05-13', profit: -70, trades: 4, winRate: 25 },
      { date: '2025-05-14', profit: 130, trades: 6, winRate: 67 },
    ];
    
    return res.json({
      success: true,
      data: dailyPerfData
    });
  } catch (error) {
    console.error('Error fetching daily metrics:', error);
    return res.status(500).json({
      success: false,
      message: 'Failed to fetch daily metrics'
    });
  }
});

// Public endpoint for metrics summary
router.get('/metrics/summary', async (req, res) => {
  try {
    // In a real implementation, fetch from database and calculate
    // For now, returning placeholder data as we transition from mock to real
    const metricsData = {
      totalPnl: 1385,
      totalTrades: 69,
      winRate: 58,
      sharpeRatio: 1.8,
      maxDrawdown: 15,
      winningTrades: 40,
      losingTrades: 29,
      avgTradeDuration: 45
    };
    
    return res.json(metricsData);
  } catch (error) {
    console.error('Error fetching metrics summary:', error);
    return res.status(500).json({
      success: false,
      message: 'Failed to fetch metrics summary'
    });
  }
});

// Authenticated endpoint for metrics summary
router.get('/metrics/summary/auth', auth, async (req, res) => {
  try {
    // In a real implementation, fetch from database and calculate
    // For now, returning placeholder data as we transition from mock to real
    const metricsData = {
      totalProfit: 1385,
      totalTrades: 69,
      winRate: 58,
      avgProfit: 20.07,
      avgLoss: -50,
      profitFactor: 2.4,
      maxDrawdown: 130,
      bestTrade: 300,
      worstTrade: -80,
      avgTradeTime: '3h 15m',
      dailyVolume: 21550,
    };
    
    return res.json({
      success: true,
      data: metricsData
    });
  } catch (error) {
    console.error('Error fetching metrics summary:', error);
    return res.status(500).json({
      success: false,
      message: 'Failed to fetch metrics summary'
    });
  }
});

// Get asset allocation data
router.get('/metrics/allocation', async (req, res) => {
  try {
    // In a real implementation, fetch from database
    // For now, returning placeholder data as we transition from mock to real
    const assetAllocationData = [
      { name: 'BTC', value: 40 },
      { name: 'ETH', value: 25 },
      { name: 'SOL', value: 15 },
      { name: 'BNB', value: 10 },
      { name: 'Other', value: 10 },
    ];
    
    return res.json(assetAllocationData);
  } catch (error) {
    console.error('Error fetching asset allocation:', error);
    return res.status(500).json({
      success: false,
      message: 'Failed to fetch asset allocation'
    });
  }
});

// Authenticated endpoint for asset allocation
router.get('/metrics/allocation/auth', auth, async (req, res) => {
  try {
    // In a real implementation, fetch from database
    // For now, returning placeholder data as we transition from mock to real
    const assetAllocationData = [
      { name: 'BTC', value: 40 },
      { name: 'ETH', value: 25 },
      { name: 'SOL', value: 15 },
      { name: 'BNB', value: 10 },
      { name: 'Other', value: 10 },
    ];
    
    return res.json({
      success: true,
      data: assetAllocationData
    });
  } catch (error) {
    console.error('Error fetching asset allocation:', error);
    return res.status(500).json({
      success: false,
      message: 'Failed to fetch asset allocation'
    });
  }
});

// Get trade history
router.get('/trades', auth, async (req, res) => {
  try {
    // In a real implementation, fetch from database
    // For now, returning placeholder data as we transition from mock to real
    const trades = [
      {
        id: '1',
        symbol: 'BTC/USD',
        type: 'buy',
        price: 42500,
        amount: 0.5,
        value: 21250,
        fee: 21.25,
        timestamp: Date.now() - 1000000,
        status: 'executed',
        profit: 1275,
        profitPercentage: 6.0,
      },
      {
        id: '2',
        symbol: 'ETH/USD',
        type: 'sell',
        price: 2950,
        amount: 2.0,
        value: 5900,
        fee: 5.9,
        timestamp: Date.now() - 2000000,
        status: 'executed',
        profit: -118,
        profitPercentage: -2.0,
      },
      {
        id: '3',
        symbol: 'BTC/USD',
        type: 'sell',
        price: 43100,
        amount: 0.25,
        value: 10775,
        fee: 10.78,
        timestamp: Date.now() - 3000000,
        status: 'executed',
        profit: 645.5,
        profitPercentage: 6.0,
      },
      {
        id: '4',
        symbol: 'SOL/USD',
        type: 'buy',
        price: 148.5,
        amount: 10,
        value: 1485,
        fee: 1.49,
        timestamp: Date.now() - 4000000,
        status: 'executed',
        profit: 0,
        profitPercentage: 0,
      },
      {
        id: '5',
        symbol: 'ADA/USD',
        type: 'buy',
        price: 0.45,
        amount: 1000,
        value: 450,
        fee: 0.45,
        timestamp: Date.now() - 5000000,
        status: 'executed',
        profit: -9,
        profitPercentage: -2.0,
      },
      {
        id: '6',
        symbol: 'XRP/USD',
        type: 'sell',
        price: 0.62,
        amount: 500,
        value: 310,
        fee: 0.31,
        timestamp: Date.now() - 6000000,
        status: 'cancelled',
      },
      {
        id: '7',
        symbol: 'DOT/USD',
        type: 'buy',
        price: 6.2,
        amount: 50,
        value: 310,
        fee: 0.31,
        timestamp: Date.now() - 7000000,
        status: 'pending',
      },
      {
        id: '8',
        symbol: 'BNB/USD',
        type: 'buy',
        price: 545,
        amount: 1.5,
        value: 817.5,
        fee: 0.82,
        timestamp: Date.now() - 8000000,
        status: 'executed',
        profit: 40.88,
        profitPercentage: 5.0,
      },
      {
        id: '9',
        symbol: 'DOGE/USD',
        type: 'sell',
        price: 0.12,
        amount: 2000,
        value: 240,
        fee: 0.24,
        timestamp: Date.now() - 9000000,
        status: 'executed',
        profit: -12,
        profitPercentage: -5.0,
      },
      {
        id: '10',
        symbol: 'LINK/USD',
        type: 'buy',
        price: 16.2,
        amount: 20,
        value: 324,
        fee: 0.32,
        timestamp: Date.now() - 10000000,
        status: 'executed',
        profit: 32.4,
        profitPercentage: 10.0,
      },
    ];
    
    return res.json({
      success: true,
      data: trades
    });
  } catch (error) {
    console.error('Error fetching trades:', error);
    return res.status(500).json({
      success: false,
      message: 'Failed to fetch trades'
    });
  }
});

module.exports = router; 