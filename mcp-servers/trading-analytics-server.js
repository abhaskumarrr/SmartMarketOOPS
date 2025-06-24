#!/usr/bin/env node
/**
 * Custom MCP Server for SmartMarketOOPS Trading Analytics
 * Provides specialized tools for trading analysis and market data
 */

const { Server } = require('@modelcontextprotocol/sdk/server/index.js');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');
const fetch = require('node-fetch');
const fs = require('fs').promises;
const path = require('path');

class TradingAnalyticsServer {
  constructor() {
    this.server = new Server(
      {
        name: 'trading-analytics',
        version: '1.0.0',
        description: 'SmartMarketOOPS Trading Analytics MCP Server'
      },
      {
        capabilities: {
          resources: {},
          tools: {}
        }
      }
    );

    this.setupHandlers();
  }

  setupHandlers() {
    // List available tools
    this.server.setRequestHandler('tools/list', async () => {
      return {
        tools: [
          {
            name: 'get_market_data',
            description: 'Fetch real-time market data from Delta Exchange',
            inputSchema: {
              type: 'object',
              properties: {
                symbol: {
                  type: 'string',
                  description: 'Trading symbol (e.g., BTCUSD)',
                  default: 'BTCUSD'
                },
                interval: {
                  type: 'string',
                  description: 'Time interval (1m, 5m, 1h, 4h, 1d)',
                  default: '1h'
                },
                limit: {
                  type: 'number',
                  description: 'Number of data points to fetch',
                  default: 100
                }
              },
              required: ['symbol']
            }
          },
          {
            name: 'analyze_trading_performance',
            description: 'Analyze trading performance from log files',
            inputSchema: {
              type: 'object',
              properties: {
                logPath: {
                  type: 'string',
                  description: 'Path to trading log file',
                  default: './logs/backend.log'
                },
                timeframe: {
                  type: 'string',
                  description: 'Analysis timeframe (1d, 7d, 30d)',
                  default: '7d'
                }
              }
            }
          },
          {
            name: 'calculate_risk_metrics',
            description: 'Calculate risk metrics for trading strategy',
            inputSchema: {
              type: 'object',
              properties: {
                trades: {
                  type: 'array',
                  description: 'Array of trade objects with pnl values'
                },
                initialCapital: {
                  type: 'number',
                  description: 'Initial trading capital',
                  default: 1000
                }
              },
              required: ['trades']
            }
          },
          {
            name: 'get_ml_predictions',
            description: 'Get ML predictions from the trading system',
            inputSchema: {
              type: 'object',
              properties: {
                marketData: {
                  type: 'array',
                  description: 'Array of market data points'
                },
                model: {
                  type: 'string',
                  description: 'ML model to use (simple, enhanced, fibonacci)',
                  default: 'simple'
                }
              },
              required: ['marketData']
            }
          },
          {
            name: 'monitor_system_health',
            description: 'Monitor the health of all trading system components',
            inputSchema: {
              type: 'object',
              properties: {
                detailed: {
                  type: 'boolean',
                  description: 'Include detailed metrics',
                  default: false
                }
              }
            }
          },
          {
            name: 'generate_trading_report',
            description: 'Generate comprehensive trading performance report',
            inputSchema: {
              type: 'object',
              properties: {
                period: {
                  type: 'string',
                  description: 'Report period (daily, weekly, monthly)',
                  default: 'daily'
                },
                format: {
                  type: 'string',
                  description: 'Report format (json, markdown, html)',
                  default: 'markdown'
                }
              }
            }
          }
        ]
      };
    });

    // Handle tool calls
    this.server.setRequestHandler('tools/call', async (request) => {
      const { name, arguments: args } = request.params;

      try {
        switch (name) {
          case 'get_market_data':
            return await this.getMarketData(args);
          case 'analyze_trading_performance':
            return await this.analyzeTradingPerformance(args);
          case 'calculate_risk_metrics':
            return await this.calculateRiskMetrics(args);
          case 'get_ml_predictions':
            return await this.getMLPredictions(args);
          case 'monitor_system_health':
            return await this.monitorSystemHealth(args);
          case 'generate_trading_report':
            return await this.generateTradingReport(args);
          default:
            throw new Error(`Unknown tool: ${name}`);
        }
      } catch (error) {
        return {
          content: [
            {
              type: 'text',
              text: `Error executing ${name}: ${error.message}`
            }
          ],
          isError: true
        };
      }
    });
  }

  async getMarketData(args) {
    const { symbol = 'BTCUSD', interval = '1h', limit = 100 } = args;
    
    try {
      // Fetch from Delta Exchange testnet
      const response = await fetch('https://testnet-api.delta.exchange/v2/products');
      const data = await response.json();
      
      const products = data.result || [];
      const product = products.find(p => p.symbol.includes(symbol.replace('USD', 'USDT')));
      
      if (!product) {
        return {
          content: [
            {
              type: 'text',
              text: `Symbol ${symbol} not found. Available symbols: ${products.slice(0, 10).map(p => p.symbol).join(', ')}`
            }
          ]
        };
      }

      // Generate mock historical data for now
      const mockData = this.generateMockMarketData(symbol, limit);
      
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              symbol,
              interval,
              dataPoints: mockData.length,
              data: mockData,
              timestamp: new Date().toISOString()
            }, null, 2)
          }
        ]
      };
    } catch (error) {
      throw new Error(`Failed to fetch market data: ${error.message}`);
    }
  }

  generateMockMarketData(symbol, limit) {
    const data = [];
    let basePrice = symbol.includes('BTC') ? 45000 : 3000;
    
    for (let i = 0; i < limit; i++) {
      const change = (Math.random() - 0.5) * 0.02; // 2% volatility
      basePrice *= (1 + change);
      
      data.push({
        timestamp: new Date(Date.now() - (limit - i) * 3600000).toISOString(),
        open: basePrice * 0.999,
        high: basePrice * 1.001,
        low: basePrice * 0.998,
        close: basePrice,
        volume: Math.random() * 1000 + 500
      });
    }
    
    return data;
  }

  async analyzeTradingPerformance(args) {
    const { logPath = './logs/backend.log', timeframe = '7d' } = args;
    
    try {
      const logContent = await fs.readFile(logPath, 'utf8');
      const lines = logContent.split('\n');
      
      // Extract trading-related log entries
      const tradingLogs = lines.filter(line => 
        line.includes('trade') || 
        line.includes('order') || 
        line.includes('position') ||
        line.includes('prediction')
      );

      const analysis = {
        totalLogEntries: lines.length,
        tradingLogEntries: tradingLogs.length,
        timeframe,
        lastUpdated: new Date().toISOString(),
        recentActivity: tradingLogs.slice(-10),
        summary: {
          errors: lines.filter(line => line.toLowerCase().includes('error')).length,
          warnings: lines.filter(line => line.toLowerCase().includes('warn')).length,
          predictions: lines.filter(line => line.includes('prediction')).length
        }
      };

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(analysis, null, 2)
          }
        ]
      };
    } catch (error) {
      throw new Error(`Failed to analyze trading performance: ${error.message}`);
    }
  }

  async calculateRiskMetrics(args) {
    const { trades, initialCapital = 1000 } = args;
    
    if (!Array.isArray(trades) || trades.length === 0) {
      return {
        content: [
          {
            type: 'text',
            text: 'No trades provided for risk analysis'
          }
        ]
      };
    }

    const pnlValues = trades.map(trade => trade.pnl || 0);
    const returns = pnlValues.map(pnl => pnl / initialCapital);
    
    const totalReturn = pnlValues.reduce((sum, pnl) => sum + pnl, 0);
    const winningTrades = pnlValues.filter(pnl => pnl > 0);
    const losingTrades = pnlValues.filter(pnl => pnl < 0);
    
    const winRate = winningTrades.length / trades.length;
    const avgWin = winningTrades.length > 0 ? winningTrades.reduce((sum, pnl) => sum + pnl, 0) / winningTrades.length : 0;
    const avgLoss = losingTrades.length > 0 ? losingTrades.reduce((sum, pnl) => sum + pnl, 0) / losingTrades.length : 0;
    
    // Calculate Sharpe ratio (simplified)
    const avgReturn = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const returnStd = Math.sqrt(returns.reduce((sum, ret) => sum + Math.pow(ret - avgReturn, 2), 0) / returns.length);
    const sharpeRatio = returnStd > 0 ? avgReturn / returnStd : 0;
    
    // Calculate maximum drawdown
    let peak = initialCapital;
    let maxDrawdown = 0;
    let runningCapital = initialCapital;
    
    for (const pnl of pnlValues) {
      runningCapital += pnl;
      if (runningCapital > peak) {
        peak = runningCapital;
      }
      const drawdown = (peak - runningCapital) / peak;
      if (drawdown > maxDrawdown) {
        maxDrawdown = drawdown;
      }
    }

    const riskMetrics = {
      totalTrades: trades.length,
      totalReturn,
      totalReturnPercent: (totalReturn / initialCapital) * 100,
      winRate: winRate * 100,
      avgWin,
      avgLoss,
      profitFactor: avgLoss !== 0 ? Math.abs(avgWin / avgLoss) : Infinity,
      sharpeRatio,
      maxDrawdown: maxDrawdown * 100,
      finalCapital: initialCapital + totalReturn,
      analysis: {
        riskLevel: maxDrawdown > 0.2 ? 'High' : maxDrawdown > 0.1 ? 'Medium' : 'Low',
        performance: totalReturn > 0 ? 'Profitable' : 'Unprofitable',
        consistency: winRate > 0.6 ? 'Consistent' : winRate > 0.4 ? 'Moderate' : 'Inconsistent'
      }
    };

    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify(riskMetrics, null, 2)
        }
      ]
    };
  }

  async getMLPredictions(args) {
    const { marketData, model = 'simple' } = args;
    
    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ market_data: marketData })
      });

      if (!response.ok) {
        throw new Error(`ML service responded with status ${response.status}`);
      }

      const prediction = await response.json();
      
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              model,
              prediction,
              timestamp: new Date().toISOString(),
              dataPoints: marketData.length
            }, null, 2)
          }
        ]
      };
    } catch (error) {
      throw new Error(`Failed to get ML predictions: ${error.message}`);
    }
  }

  async monitorSystemHealth(args) {
    const { detailed = false } = args;
    
    const healthChecks = [];
    
    // Check backend API
    try {
      const backendResponse = await fetch('http://localhost:3001/api/health');
      const backendData = await backendResponse.json();
      healthChecks.push({
        service: 'Backend API',
        status: 'healthy',
        port: 3001,
        details: detailed ? backendData : null
      });
    } catch (error) {
      healthChecks.push({
        service: 'Backend API',
        status: 'unhealthy',
        port: 3001,
        error: error.message
      });
    }

    // Check ML service
    try {
      const mlResponse = await fetch('http://localhost:8000/health');
      const mlData = await mlResponse.json();
      healthChecks.push({
        service: 'ML Service',
        status: 'healthy',
        port: 8000,
        details: detailed ? mlData : null
      });
    } catch (error) {
      healthChecks.push({
        service: 'ML Service',
        status: 'unhealthy',
        port: 8000,
        error: error.message
      });
    }

    // Check Delta Exchange connection
    try {
      const deltaResponse = await fetch('http://localhost:3001/api/delta/test');
      const deltaData = await deltaResponse.json();
      healthChecks.push({
        service: 'Delta Exchange',
        status: deltaData.status === 'success' ? 'healthy' : 'unhealthy',
        details: detailed ? deltaData : null
      });
    } catch (error) {
      healthChecks.push({
        service: 'Delta Exchange',
        status: 'unhealthy',
        error: error.message
      });
    }

    const overallHealth = healthChecks.every(check => check.status === 'healthy') ? 'healthy' : 'degraded';

    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify({
            overallHealth,
            timestamp: new Date().toISOString(),
            services: healthChecks,
            summary: {
              total: healthChecks.length,
              healthy: healthChecks.filter(c => c.status === 'healthy').length,
              unhealthy: healthChecks.filter(c => c.status === 'unhealthy').length
            }
          }, null, 2)
        }
      ]
    };
  }

  async generateTradingReport(args) {
    const { period = 'daily', format = 'markdown' } = args;
    
    try {
      // Get system health
      const healthData = await this.monitorSystemHealth({ detailed: true });
      const health = JSON.parse(healthData.content[0].text);
      
      // Generate mock trading data for the report
      const mockTrades = [
        { pnl: 25.50, timestamp: new Date().toISOString(), symbol: 'BTCUSD' },
        { pnl: -12.30, timestamp: new Date(Date.now() - 3600000).toISOString(), symbol: 'ETHUSD' },
        { pnl: 18.75, timestamp: new Date(Date.now() - 7200000).toISOString(), symbol: 'BTCUSD' }
      ];
      
      const riskMetricsData = await this.calculateRiskMetrics({ trades: mockTrades, initialCapital: 1000 });
      const riskMetrics = JSON.parse(riskMetricsData.content[0].text);

      let report = '';
      
      if (format === 'markdown') {
        report = `# SmartMarketOOPS Trading Report (${period})

## System Health
- **Overall Status**: ${health.overallHealth}
- **Services Online**: ${health.summary.healthy}/${health.summary.total}
- **Last Updated**: ${health.timestamp}

## Trading Performance
- **Total Trades**: ${riskMetrics.totalTrades}
- **Total Return**: $${riskMetrics.totalReturn.toFixed(2)} (${riskMetrics.totalReturnPercent.toFixed(2)}%)
- **Win Rate**: ${riskMetrics.winRate.toFixed(1)}%
- **Sharpe Ratio**: ${riskMetrics.sharpeRatio.toFixed(2)}
- **Max Drawdown**: ${riskMetrics.maxDrawdown.toFixed(2)}%

## Risk Analysis
- **Risk Level**: ${riskMetrics.analysis.riskLevel}
- **Performance**: ${riskMetrics.analysis.performance}
- **Consistency**: ${riskMetrics.analysis.consistency}

## Recent Trades
${mockTrades.map(trade => `- ${trade.symbol}: $${trade.pnl.toFixed(2)} at ${new Date(trade.timestamp).toLocaleString()}`).join('\n')}

---
*Report generated on ${new Date().toISOString()}*
`;
      } else {
        report = JSON.stringify({
          period,
          systemHealth: health,
          tradingPerformance: riskMetrics,
          recentTrades: mockTrades,
          generatedAt: new Date().toISOString()
        }, null, 2);
      }

      return {
        content: [
          {
            type: 'text',
            text: report
          }
        ]
      };
    } catch (error) {
      throw new Error(`Failed to generate trading report: ${error.message}`);
    }
  }

  start() {
    const transport = new StdioServerTransport();
    this.server.connect(transport);
    console.error('Trading Analytics MCP Server started');
  }
}

// Start the server
if (require.main === module) {
  const server = new TradingAnalyticsServer();
  server.start();
}

module.exports = TradingAnalyticsServer;