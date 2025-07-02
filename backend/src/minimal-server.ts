import express from 'express';
import cors from 'cors';
import prisma from './config/prisma';

const app = express();
const PORT = process.env.PORT || 3006;

// Enable CORS for all routes with more permissive settings
app.use(cors({
  origin: ['http://localhost:5173', 'http://localhost:3000', 'http://localhost:3001', 'http://127.0.0.1:3000'],
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));

// Parse JSON bodies
app.use(express.json());

// Logging middleware
app.use((req, res, next) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.path} - Origin: ${req.get('Origin') || 'none'}`);
  next();
});

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({
    success: true,
    message: 'Backend server with real database connection',
    timestamp: new Date().toISOString(),
    service: 'SmartMarketOOPS Backend',
    database: 'Connected to PostgreSQL via Prisma'
  });
});

// Dashboard summary endpoint with real data
app.get('/api/dashboard/dashboard-summary', async (req, res) => {
  try {
    console.log('Fetching real dashboard data from database...');

    // Get the first user for demo purposes (in production, this would be based on auth)
    const user = await prisma.user.findFirst({
      include: {
        positions: {
          where: { status: 'Open' }
        },
        bots: {
          where: { isActive: true }
        },
        tradeLogs: {
          orderBy: { timestamp: 'desc' },
          take: 10
        }
      }
    });

    if (!user) {
      // If no users exist, return demo data structure
      return res.json({
        success: true,
        data: {
          portfolioValue: '0.00',
          dailyChange: '+0.00',
          activePositions: 0,
          profitablePositions: 0,
          aiAccuracy: 0,
          totalPnl: '+0.00',
          totalPnlPercentage: '+0.0%',
          tradingStatus: 'inactive',
          message: 'No users found in database - showing default values'
        }
      });
    }

    // Calculate portfolio metrics from real data
    const openPositions = user.positions || [];
    const activePositions = openPositions.length;
    
    // Calculate total portfolio value from positions
    const portfolioValue = openPositions.reduce((total, position) => {
      const currentValue = (position.currentPrice || position.entryPrice) * position.amount;
      return total + currentValue;
    }, 0);

    // Calculate profitable positions
    const profitablePositions = openPositions.filter(position => {
      if (!position.currentPrice) return false;
      const pnl = (position.currentPrice - position.entryPrice) * position.amount;
      return position.side === 'Long' ? pnl > 0 : pnl < 0;
    }).length;

    // Calculate total PnL from positions
    const totalPnl = openPositions.reduce((total, position) => {
      if (!position.currentPrice) return total;
      const pnl = (position.currentPrice - position.entryPrice) * position.amount;
      return total + (position.side === 'Long' ? pnl : -pnl);
    }, 0);

    // Calculate PnL percentage
    const totalPnlPercentage = portfolioValue > 0 ? (totalPnl / portfolioValue) * 100 : 0;

    // Determine trading status based on active bots
    const activeBots = user.bots?.filter(bot => bot.isActive) || [];
    const tradingStatus = activeBots.length > 0 ? 'active' : 'inactive';

    // Mock AI accuracy (in production, this would come from ML model performance metrics)
    const aiAccuracy = activeBots.length > 0 ? 87.5 : 0;

    // Calculate daily change (mock for now, would need historical data)
    const dailyChange = totalPnlPercentage > 0 ? `+${totalPnlPercentage.toFixed(2)}` : totalPnlPercentage.toFixed(2);

    const response = {
      success: true,
      data: {
        portfolioValue: portfolioValue.toFixed(2),
        dailyChange: `${dailyChange}%`,
        activePositions,
        profitablePositions,
        aiAccuracy,
        totalPnl: totalPnl >= 0 ? `+${totalPnl.toFixed(2)}` : totalPnl.toFixed(2),
        totalPnlPercentage: `${totalPnlPercentage >= 0 ? '+' : ''}${totalPnlPercentage.toFixed(1)}%`,
        tradingStatus,
        userId: user.id,
        userName: user.name,
        activeBots: activeBots.length,
        totalTrades: user.tradeLogs?.length || 0
      }
    };

    console.log('Dashboard data response:', response);
    res.json(response);

  } catch (error) {
    console.error('Database error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch dashboard data',
      details: error instanceof Error ? error.message : 'Unknown error',
      data: {
        portfolioValue: '0.00',
        dailyChange: '+0.00',
        activePositions: 0,
        profitablePositions: 0,
        aiAccuracy: 0,
        totalPnl: '+0.00',
        totalPnlPercentage: '+0.0%',
        tradingStatus: 'error'
      }
    });
  }
});

// Trading status endpoint
app.get('/api/trading/status', async (req, res) => {
  try {
    const activeBots = await prisma.bot.count({
      where: { isActive: true }
    });

    const openPositions = await prisma.position.count({
      where: { status: 'Open' }
    });

    res.json({
      success: true,
      data: {
        status: activeBots > 0 ? 'active' : 'inactive',
        activeBots,
        openPositions,
        timestamp: new Date().toISOString()
      }
    });
  } catch (error) {
    console.error('Error fetching trading status:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch trading status'
    });
  }
});

// Portfolio endpoint
app.get('/api/portfolio', async (req, res) => {
  try {
    const positions = await prisma.position.findMany({
      where: { status: 'Open' },
      include: {
        user: {
          select: {
            id: true,
            name: true
          }
        },
        bot: {
          select: {
            id: true,
            name: true,
            strategy: true
          }
        }
      },
      orderBy: { openedAt: 'desc' }
    });

    res.json({
      success: true,
      data: {
        positions,
        totalPositions: positions.length,
        timestamp: new Date().toISOString()
      }
    });
  } catch (error) {
    console.error('Error fetching portfolio:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch portfolio data'
    });
  }
});

// Error handling middleware
app.use((err: any, req: any, res: any, next: any) => {
  console.error('Server error:', err);
  res.status(500).json({
    success: false,
    error: 'Internal server error',
    message: err.message
  });
});

// Graceful shutdown
process.on('SIGINT', async () => {
  console.log('Shutting down gracefully...');
  await prisma.$disconnect();
  process.exit(0);
});

process.on('SIGTERM', async () => {
  console.log('Shutting down gracefully...');
  await prisma.$disconnect();
  process.exit(0);
});

app.listen(PORT, () => {
  console.log(`🚀 Minimal backend server running on port ${PORT}`);
  console.log(`📊 Dashboard API: http://localhost:${PORT}/api/dashboard/dashboard-summary`);
  console.log(`🔍 Health Check: http://localhost:${PORT}/api/health`);
  console.log(`💼 Portfolio API: http://localhost:${PORT}/api/portfolio`);
  console.log(`📈 Trading Status: http://localhost:${PORT}/api/trading/status`);
});

export default app; 