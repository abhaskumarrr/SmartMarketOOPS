/**
 * Phase 4: Production Readiness - End-to-End Trading Workflows Tests
 * Target: Complete trading workflows from signal generation to execution
 */

import { describe, beforeAll, afterAll, beforeEach, afterEach, it, expect, jest } from '@jest/globals';
import request from 'supertest';
import { Server } from 'http';
import WebSocket from 'ws';
import app from '../../../src/app';
import prisma from '../../../src/config/prisma-readonly';

describe('Phase 4: End-to-End Trading Workflows', () => {
  let server: Server;
  let authToken: string;
  let testUserId: string;
  let wsClient: WebSocket;

  beforeAll(async () => {
    // Start the server
    server = app.listen(0);
    const port = (server.address() as any)?.port;

    // Create test user with trading permissions
    const testUser = await prisma.user.create({
      data: {
        email: 'trader@e2e.com',
        password: 'hashedpassword123',
        name: 'E2E Trader',
        role: 'USER',
        isEmailVerified: true,
        tradingEnabled: true,
      }
    });
    testUserId = testUser.id;

    // Set up initial account balance and risk parameters
    await prisma.riskManagement.create({
      data: {
        userId: testUserId,
        maxPositionSize: 10000,
        maxRiskPerTrade: 0.02,
        dailyRiskLimit: 0.1,
        accountBalance: 100000,
        leverage: 5,
      }
    });

    // Get auth token
    const loginResponse = await request(app)
      .post('/api/auth/login')
      .send({
        email: 'trader@e2e.com',
        password: 'password123'
      });
    
    authToken = loginResponse.body.token;

    // Set up WebSocket connection
    wsClient = new WebSocket(`ws://localhost:${port}/ws`, {
      headers: { Authorization: `Bearer ${authToken}` }
    });

    await new Promise((resolve) => {
      wsClient.on('open', resolve);
    });
  });

  afterAll(async () => {
    // Cleanup
    wsClient.close();
    
    await prisma.order.deleteMany({ where: { userId: testUserId } });
    await prisma.position.deleteMany({ where: { userId: testUserId } });
    await prisma.tradingSignal.deleteMany({});
    await prisma.riskManagement.deleteMany({ where: { userId: testUserId } });
    await prisma.user.delete({ where: { id: testUserId } });
    
    await prisma.$disconnect();
    server.close();
  });

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Complete Trading Signal to Execution Workflow', () => {
    it('should handle complete signal-to-trade workflow successfully', async () => {
      // Step 1: Generate ML-based trading signal
      const signalData = {
        symbol: 'BTCUSDT',
        direction: 'buy',
        confidence: 0.89,
        currentPrice: 50000,
        targetPrice: 52000,
        stopLoss: 48500,
        timeframe: '1h',
        indicators: {
          rsi: 35.5,
          macd: 1.2,
          bollingerBands: { upper: 51000, lower: 49000 },
        }
      };

      const signalResponse = await request(app)
        .post('/api/signals')
        .set('Authorization', `Bearer ${authToken}`)
        .send(signalData)
        .expect(201);

      const signal = signalResponse.body.signal;
      expect(signal.id).toBeDefined();
      expect(signal.confidence).toBe(0.89);

      // Step 2: Signal triggers automatic position sizing calculation
      const positionSizeResponse = await request(app)
        .post('/api/trading/calculate-position-size')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          signalId: signal.id,
          symbol: 'BTCUSDT',
          direction: 'buy',
          entryPrice: 50000,
          stopLoss: 48500,
          riskPercentage: 0.02,
        })
        .expect(200);

      const positionSize = positionSizeResponse.body.positionSize;
      expect(positionSize).toBeGreaterThan(0);
      expect(positionSize).toBeLessThanOrEqual(1.33); // Max position for 2% risk

      // Step 3: Risk management validation
      const riskValidationResponse = await request(app)
        .post('/api/trading/validate-risk')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          symbol: 'BTCUSDT',
          side: 'buy',
          quantity: positionSize,
          price: 50000,
          stopLoss: 48500,
        })
        .expect(200);

      expect(riskValidationResponse.body.approved).toBe(true);
      expect(riskValidationResponse.body.riskScore).toBeLessThanOrEqual(5);

      // Step 4: Execute the trade
      const orderResponse = await request(app)
        .post('/api/orders')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          symbol: 'BTCUSDT',
          type: 'market',
          side: 'buy',
          quantity: positionSize,
          signalId: signal.id,
        })
        .expect(201);

      const order = orderResponse.body.order;
      expect(order.status).toBe('pending');
      expect(order.quantity).toBe(positionSize);

      // Step 5: Simulate order execution
      const executionResponse = await request(app)
        .put(`/api/orders/${order.id}/execute`)
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          executionPrice: 50050,
          executedQuantity: positionSize,
          executionTime: new Date().toISOString(),
        })
        .expect(200);

      expect(executionResponse.body.order.status).toBe('filled');
      expect(executionResponse.body.order.executedPrice).toBe(50050);

      // Step 6: Verify position creation
      const positionResponse = await request(app)
        .get('/api/positions')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      const positions = positionResponse.body.positions;
      expect(positions).toHaveLength(1);
      expect(positions[0].symbol).toBe('BTCUSDT');
      expect(positions[0].side).toBe('long');
      expect(positions[0].size).toBe(positionSize);

      // Step 7: Set up stop-loss and take-profit orders
      const stopLossResponse = await request(app)
        .post('/api/orders')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          symbol: 'BTCUSDT',
          type: 'stop_loss',
          side: 'sell',
          quantity: positionSize,
          stopPrice: 48500,
          parentOrderId: order.id,
        })
        .expect(201);

      const takeProfitResponse = await request(app)
        .post('/api/orders')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          symbol: 'BTCUSDT',
          type: 'take_profit',
          side: 'sell',
          quantity: positionSize,
          limitPrice: 52000,
          parentOrderId: order.id,
        })
        .expect(201);

      expect(stopLossResponse.body.order.type).toBe('stop_loss');
      expect(takeProfitResponse.body.order.type).toBe('take_profit');

      // Cleanup test data
      await request(app)
        .delete(`/api/positions/${positions[0].id}`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);
    });

    it('should handle portfolio rebalancing workflow', async () => {
      // Create multiple positions
      const positions = [
        {
          symbol: 'BTCUSDT',
          side: 'long',
          size: 1.0,
          entryPrice: 50000,
          currentPrice: 51000,
        },
        {
          symbol: 'ETHUSDT',
          side: 'long',
          size: 10.0,
          entryPrice: 3000,
          currentPrice: 2900,
        },
        {
          symbol: 'ADAUSDT',
          side: 'short',
          size: 5000.0,
          entryPrice: 0.5,
          currentPrice: 0.52,
        },
      ];

      // Create positions in database
      const createdPositions = await Promise.all(
        positions.map(pos =>
          prisma.position.create({
            data: {
              userId: testUserId,
              symbol: pos.symbol,
              side: pos.side,
              size: pos.size,
              entryPrice: pos.entryPrice,
              currentPrice: pos.currentPrice,
              unrealizedPnl: (pos.currentPrice - pos.entryPrice) * pos.size * (pos.side === 'long' ? 1 : -1),
            }
          })
        )
      );

      // Step 1: Analyze current portfolio allocation
      const portfolioResponse = await request(app)
        .get('/api/portfolio/analysis')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(portfolioResponse.body.totalValue).toBeGreaterThan(0);
      expect(portfolioResponse.body.allocation).toHaveProperty('BTCUSDT');
      expect(portfolioResponse.body.allocation).toHaveProperty('ETHUSDT');
      expect(portfolioResponse.body.allocation).toHaveProperty('ADAUSDT');

      // Step 2: Get rebalancing recommendations
      const rebalanceResponse = await request(app)
        .post('/api/portfolio/rebalance-recommendations')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          targetAllocation: {
            'BTCUSDT': 0.4,
            'ETHUSDT': 0.3,
            'ADAUSDT': 0.2,
            'cash': 0.1,
          },
          maxTradeSize: 0.1, // 10% of portfolio per trade
        })
        .expect(200);

      const recommendations = rebalanceResponse.body.recommendations;
      expect(recommendations).toBeInstanceOf(Array);
      expect(recommendations.length).toBeGreaterThan(0);

      // Step 3: Execute rebalancing trades
      for (const recommendation of recommendations) {
        const tradeResponse = await request(app)
          .post('/api/orders')
          .set('Authorization', `Bearer ${authToken}`)
          .send({
            symbol: recommendation.symbol,
            type: 'market',
            side: recommendation.action, // 'buy' or 'sell'
            quantity: recommendation.quantity,
            reason: 'portfolio_rebalancing',
          })
          .expect(201);

        expect(tradeResponse.body.order.status).toBe('pending');
      }

      // Cleanup
      await prisma.position.deleteMany({
        where: { id: { in: createdPositions.map(p => p.id) } }
      });
    });
  });

  describe('Real-time Market Data and Trading Workflow', () => {
    it('should handle real-time price updates and trigger-based trading', async () => {
      // Set up a price alert/trigger
      const alertResponse = await request(app)
        .post('/api/alerts')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          symbol: 'BTCUSDT',
          condition: 'price_above',
          value: 51000,
          action: 'create_order',
          orderParams: {
            type: 'market',
            side: 'buy',
            quantity: 0.01,
          },
        })
        .expect(201);

      const alert = alertResponse.body.alert;
      expect(alert.id).toBeDefined();

      // Simulate real-time price update via WebSocket
      const priceUpdate = {
        type: 'price_update',
        symbol: 'BTCUSDT',
        price: 51500,
        volume: 1000,
        timestamp: Date.now(),
      };

      // Send price update to WebSocket
      wsClient.send(JSON.stringify(priceUpdate));

      // Wait for alert to be triggered
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Check if order was automatically created
      const ordersResponse = await request(app)
        .get('/api/orders?limit=1&sort=created_desc')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      const orders = ordersResponse.body.orders;
      expect(orders.length).toBeGreaterThan(0);
      expect(orders[0].symbol).toBe('BTCUSDT');
      expect(orders[0].side).toBe('buy');
      expect(orders[0].triggeredBy).toBe(alert.id);

      // Cleanup
      await request(app)
        .delete(`/api/alerts/${alert.id}`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);
    });

    it('should handle WebSocket-based order book updates', async () => {
      let orderBookReceived = false;
      let lastOrderBook: any = null;

      // Subscribe to order book updates
      wsClient.send(JSON.stringify({
        type: 'subscribe',
        channel: 'orderbook',
        symbol: 'BTCUSDT',
      }));

      // Listen for order book updates
      wsClient.on('message', (data) => {
        const message = JSON.parse(data.toString());
        if (message.type === 'orderbook' && message.symbol === 'BTCUSDT') {
          orderBookReceived = true;
          lastOrderBook = message.data;
        }
      });

      // Wait for order book data
      await new Promise(resolve => setTimeout(resolve, 2000));

      expect(orderBookReceived).toBe(true);
      expect(lastOrderBook).toBeDefined();
      expect(lastOrderBook.bids).toBeInstanceOf(Array);
      expect(lastOrderBook.asks).toBeInstanceOf(Array);
      expect(lastOrderBook.bids.length).toBeGreaterThan(0);
      expect(lastOrderBook.asks.length).toBeGreaterThan(0);

      // Verify bid/ask structure
      const firstBid = lastOrderBook.bids[0];
      const firstAsk = lastOrderBook.asks[0];
      
      expect(firstBid).toHaveLength(2); // [price, quantity]
      expect(firstAsk).toHaveLength(2);
      expect(typeof firstBid[0]).toBe('number'); // price
      expect(typeof firstBid[1]).toBe('number'); // quantity
    });
  });

  describe('Risk Management and Circuit Breakers', () => {
    it('should trigger circuit breakers on excessive losses', async () => {
      // Create multiple losing positions to trigger circuit breaker
      const losingPositions = await Promise.all([
        prisma.position.create({
          data: {
            userId: testUserId,
            symbol: 'BTCUSDT',
            side: 'long',
            size: 1.0,
            entryPrice: 50000,
            currentPrice: 45000,
            unrealizedPnl: -5000,
          }
        }),
        prisma.position.create({
          data: {
            userId: testUserId,
            symbol: 'ETHUSDT',
            side: 'long',
            size: 10.0,
            entryPrice: 3000,
            currentPrice: 2700,
            unrealizedPnl: -3000,
          }
        }),
      ]);

      // Attempt to place another risky order
      const orderResponse = await request(app)
        .post('/api/orders')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          symbol: 'ADAUSDT',
          type: 'market',
          side: 'buy',
          quantity: 10000, // Large position
        })
        .expect(403); // Should be blocked by circuit breaker

      expect(orderResponse.body.error).toContain('Circuit breaker activated');
      expect(orderResponse.body.reason).toContain('daily loss limit exceeded');

      // Check that user's trading is temporarily disabled
      const userResponse = await request(app)
        .get('/api/user/trading-status')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(userResponse.body.tradingEnabled).toBe(false);
      expect(userResponse.body.reason).toContain('circuit_breaker');

      // Cleanup
      await prisma.position.deleteMany({
        where: { id: { in: losingPositions.map(p => p.id) } }
      });
    });

    it('should handle margin call scenarios', async () => {
      // Create leveraged position with significant drawdown
      const leveragedPosition = await prisma.position.create({
        data: {
          userId: testUserId,
          symbol: 'BTCUSDT',
          side: 'long',
          size: 5.0, // 5x leverage on $50k = $250k exposure
          entryPrice: 50000,
          currentPrice: 42000, // 16% loss
          unrealizedPnl: -40000,
          leverage: 5,
          marginUsed: 50000,
        }
      });

      // Check margin level
      const marginResponse = await request(app)
        .get('/api/trading/margin-status')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(marginResponse.body.marginLevel).toBeLessThan(1.2); // Below safe threshold
      expect(marginResponse.body.marginCallTriggered).toBe(true);

      // Attempt to open new position should be blocked
      const newOrderResponse = await request(app)
        .post('/api/orders')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          symbol: 'ETHUSDT',
          type: 'market',
          side: 'buy',
          quantity: 1.0,
        })
        .expect(403);

      expect(newOrderResponse.body.error).toContain('Margin call active');

      // Cleanup
      await prisma.position.delete({ where: { id: leveragedPosition.id } });
    });
  });

  describe('Performance Monitoring and Analytics', () => {
    it('should track and report real-time performance metrics', async () => {
      // Create trading history for performance calculation
      const trades = await Promise.all([
        prisma.order.create({
          data: {
            userId: testUserId,
            symbol: 'BTCUSDT',
            type: 'market',
            side: 'buy',
            quantity: 0.1,
            status: 'filled',
            price: 50000,
            executedPrice: 50000,
            pnl: 500,
            executedAt: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000), // 7 days ago
          }
        }),
        prisma.order.create({
          data: {
            userId: testUserId,
            symbol: 'ETHUSDT',
            type: 'market',
            side: 'sell',
            quantity: 1.0,
            status: 'filled',
            price: 3000,
            executedPrice: 3000,
            pnl: -200,
            executedAt: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000), // 5 days ago
          }
        }),
        prisma.order.create({
          data: {
            userId: testUserId,
            symbol: 'ADAUSDT',
            type: 'market',
            side: 'buy',
            quantity: 2000,
            status: 'filled',
            price: 0.5,
            executedPrice: 0.5,
            pnl: 150,
            executedAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000), // 2 days ago
          }
        }),
      ]);

      // Get comprehensive performance analytics
      const analyticsResponse = await request(app)
        .get('/api/analytics/performance?period=30d&includeMetrics=all')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      const analytics = analyticsResponse.body;

      // Verify all required metrics are present
      expect(analytics).toHaveProperty('winRate');
      expect(analytics).toHaveProperty('totalPnl');
      expect(analytics).toHaveProperty('sharpeRatio');
      expect(analytics).toHaveProperty('maxDrawdown');
      expect(analytics).toHaveProperty('profitFactor');
      expect(analytics).toHaveProperty('averageWin');
      expect(analytics).toHaveProperty('averageLoss');
      expect(analytics).toHaveProperty('largestWin');
      expect(analytics).toHaveProperty('largestLoss');
      expect(analytics).toHaveProperty('consecutiveWins');
      expect(analytics).toHaveProperty('consecutiveLosses');

      // Verify calculations
      expect(analytics.totalPnl).toBe(450); // 500 - 200 + 150
      expect(analytics.winRate).toBeCloseTo(0.667, 2); // 2 wins out of 3 trades
      expect(analytics.profitFactor).toBeGreaterThan(1); // Profitable

      // Cleanup
      await prisma.order.deleteMany({
        where: { id: { in: trades.map(t => t.id) } }
      });
    });

    it('should generate detailed trading reports', async () => {
      const reportResponse = await request(app)
        .post('/api/reports/generate')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          type: 'trading_summary',
          period: {
            start: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(),
            end: new Date().toISOString(),
          },
          includeCharts: true,
          format: 'json',
        })
        .expect(200);

      const report = reportResponse.body.report;

      expect(report).toHaveProperty('summary');
      expect(report).toHaveProperty('trades');
      expect(report).toHaveProperty('performance');
      expect(report).toHaveProperty('riskMetrics');
      expect(report).toHaveProperty('charts');

      expect(report.summary).toHaveProperty('totalTrades');
      expect(report.summary).toHaveProperty('winningTrades');
      expect(report.summary).toHaveProperty('losingTrades');
      expect(report.summary).toHaveProperty('netPnl');

      expect(report.charts).toHaveProperty('equityCurve');
      expect(report.charts).toHaveProperty('drawdownChart');
      expect(report.charts).toHaveProperty('monthlyReturns');
    });
  });

  describe('Error Recovery and Fault Tolerance', () => {
    it('should handle and recover from temporary service outages', async () => {
      // Simulate external service failure
      const mockExternalError = jest.fn()
        .mockRejectedValueOnce(new Error('External service unavailable'))
        .mockResolvedValueOnce({ success: true });

      // Test retry mechanism
      const retryResponse = await request(app)
        .post('/api/trading/external-order')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          symbol: 'BTCUSDT',
          type: 'market',
          side: 'buy',
          quantity: 0.01,
        })
        .expect(200);

      expect(retryResponse.body.success).toBe(true);
      expect(retryResponse.body.retriesUsed).toBeGreaterThan(0);
    });

    it('should maintain data consistency during system failures', async () => {
      // Start a transaction that simulates partial failure
      const orderData = {
        symbol: 'BTCUSDT',
        type: 'market',
        side: 'buy',
        quantity: 0.1,
      };

      // Simulate database transaction failure midway
      const orderResponse = await request(app)
        .post('/api/orders/atomic')
        .set('Authorization', `Bearer ${authToken}`)
        .send(orderData);

      // Should either completely succeed or completely fail (no partial states)
      if (orderResponse.status === 201) {
        // If order was created, all related records should exist
        const order = orderResponse.body.order;
        
        const [dbOrder, auditLog, riskCheck] = await Promise.all([
          prisma.order.findUnique({ where: { id: order.id } }),
          prisma.auditLog.findFirst({ where: { entityId: order.id } }),
          prisma.riskValidation.findFirst({ where: { orderId: order.id } }),
        ]);

        expect(dbOrder).toBeTruthy();
        expect(auditLog).toBeTruthy();
        expect(riskCheck).toBeTruthy();
      } else {
        // If order failed, no related records should exist
        const orphanedRecords = await prisma.order.findMany({
          where: {
            userId: testUserId,
            createdAt: { gte: new Date(Date.now() - 60000) } // Last minute
          }
        });

        expect(orphanedRecords).toHaveLength(0);
      }
    });
  });
});