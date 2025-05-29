/**
 * Database Seed Script
 * 
 * Populates the database with sample data for development and testing purposes.
 * Run with: npx prisma db seed
 */

const { PrismaClient } = require('../generated/prisma');
const bcrypt = require('bcryptjs');
const crypto = require('crypto');

// Mock encryption function since we can't easily import from TypeScript in a JS file
const encrypt = (data) => {
  // This is a simple mock for seeding purposes
  // In production, use the actual encryption module
  return JSON.stringify(data);
};

const prisma = new PrismaClient();

// Sample data constants
const SAMPLE_PASSWORD = 'Password123!';
const SALT_ROUNDS = 10;

async function main() {
  console.log('🌱 Starting database seeding...');

  // Clear existing data
  console.log('Clearing existing data...');
  await clearDatabase();

  // Create sample users
  console.log('Creating sample users...');
  const users = await createUsers();

  // Create API keys for users
  console.log('Creating API keys...');
  await createApiKeys(users);

  // Create trading bots
  console.log('Creating trading bots...');
  const bots = await createBots(users);

  // Create positions
  console.log('Creating positions...');
  await createPositions(users, bots);

  // Create trade logs
  console.log('Creating trade logs...');
  await createTradeLogs(users);

  // Create metrics
  console.log('Creating metrics...');
  await createMetrics();

  console.log('✅ Seeding completed successfully!');
}

/**
 * Clear all existing data from the database
 */
async function clearDatabase() {
  await prisma.position.deleteMany({});
  await prisma.tradeLog.deleteMany({});
  await prisma.apiKey.deleteMany({});
  await prisma.bot.deleteMany({});
  await prisma.metric.deleteMany({});
  await prisma.user.deleteMany({});
}

/**
 * Create sample users
 * @returns {Promise<Array>} Created users
 */
async function createUsers() {
  // Hash the sample password
  const hashedPassword = await bcrypt.hash(SAMPLE_PASSWORD, SALT_ROUNDS);

  const userData = [
    {
      name: 'Admin User',
      email: 'admin@example.com',
      password: hashedPassword,
    },
    {
      name: 'John Doe',
      email: 'john@example.com',
      password: hashedPassword,
    },
    {
      name: 'Jane Smith',
      email: 'jane@example.com',
      password: hashedPassword,
    },
  ];

  return Promise.all(
    userData.map(async (user) => {
      return prisma.user.create({
        data: user,
      });
    })
  );
}

/**
 * Create API keys for users
 * @param {Array} users - Array of created users
 */
async function createApiKeys(users) {
  // Sample exchange API key data
  const exchangeApiKey = {
    apiKey: 'delta_exchange_api_key_sample',
    apiSecret: 'delta_exchange_api_secret_sample',
    label: 'Delta Exchange',
    testnet: true,
  };

  const apiKeys = users.map((user) => ({
    key: crypto.randomBytes(16).toString('hex'),
    encryptedData: encrypt(exchangeApiKey),
    userId: user.id,
    scopes: 'trade,account,market',
    expiry: new Date(Date.now() + 90 * 24 * 60 * 60 * 1000), // 90 days
  }));

  return Promise.all(
    apiKeys.map((apiKey) => prisma.apiKey.create({ data: apiKey }))
  );
}

/**
 * Create sample trading bots
 * @param {Array} users - Array of created users
 * @returns {Promise<Array>} Created bots
 */
async function createBots(users) {
  const botData = [
    {
      name: 'BTC Trend Follower',
      symbol: 'BTCUSDT',
      strategy: 'trend-following',
      timeframe: '1h',
      parameters: {
        fastMAPeriod: 10,
        slowMAPeriod: 20,
        rsiPeriod: 14,
        rsiOverbought: 70,
        rsiOversold: 30,
      },
      isActive: false,
    },
    {
      name: 'ETH Mean Reversion',
      symbol: 'ETHUSDT',
      strategy: 'mean-reversion',
      timeframe: '15m',
      parameters: {
        lookbackPeriod: 20,
        deviationThreshold: 2,
        stopLossPercent: 1.5,
        takeProfitPercent: 3,
      },
      isActive: true,
    },
    {
      name: 'SOL Breakout',
      symbol: 'SOLUSDT',
      strategy: 'breakout',
      timeframe: '4h',
      parameters: {
        channelPeriod: 20,
        volumeThreshold: 1.5,
        riskPercentage: 1,
        profitTarget: 3,
      },
      isActive: false,
    },
  ];

  // Create bots for each user (3 bots per user)
  const bots = [];
  for (const user of users) {
    const userBots = await Promise.all(
      botData.map((bot) =>
        prisma.bot.create({
          data: {
            ...bot,
            userId: user.id,
            parameters: bot.parameters,
          },
        })
      )
    );
    bots.push(...userBots);
  }

  return bots;
}

/**
 * Create sample positions
 * @param {Array} users - Array of created users
 * @param {Array} bots - Array of created bots
 */
async function createPositions(users, bots) {
  const positions = [];

  // Create active positions for some bots
  for (let i = 0; i < bots.length; i++) {
    const bot = bots[i];
    
    // Only create positions for some bots
    if (i % 3 === 0) {
      positions.push({
        userId: bot.userId,
        botId: bot.id,
        symbol: bot.symbol,
        side: i % 2 === 0 ? 'Long' : 'Short',
        entryPrice: 30000 + Math.random() * 1000,
        currentPrice: 30000 + Math.random() * 1000,
        amount: 0.1 + Math.random() * 0.5,
        leverage: 1 + Math.floor(Math.random() * 10),
        takeProfitPrice: 31000 + Math.random() * 1000,
        stopLossPrice: 29000 + Math.random() * 1000,
        status: 'Open',
        pnl: Math.random() * 200 - 100,
        openedAt: new Date(Date.now() - Math.floor(Math.random() * 7 * 24 * 60 * 60 * 1000)),
        metadata: { exchange: 'Delta', orderId: crypto.randomBytes(8).toString('hex') },
      });
    }
  }

  // Create some closed positions
  for (let i = 0; i < 15; i++) {
    const user = users[i % users.length];
    const bot = bots[i % bots.length];
    
    positions.push({
      userId: user.id,
      botId: bot.id,
      symbol: bot.symbol,
      side: i % 2 === 0 ? 'Long' : 'Short',
      entryPrice: 30000 + Math.random() * 1000,
      currentPrice: 30000 + Math.random() * 1000,
      amount: 0.1 + Math.random() * 0.5,
      leverage: 1 + Math.floor(Math.random() * 10),
      takeProfitPrice: 31000 + Math.random() * 1000,
      stopLossPrice: 29000 + Math.random() * 1000,
      status: 'Closed',
      pnl: Math.random() * 400 - 200,
      openedAt: new Date(Date.now() - Math.floor(Math.random() * 30 * 24 * 60 * 60 * 1000)),
      closedAt: new Date(Date.now() - Math.floor(Math.random() * 7 * 24 * 60 * 60 * 1000)),
      metadata: { exchange: 'Delta', orderId: crypto.randomBytes(8).toString('hex') },
    });
  }

  return Promise.all(positions.map(position => prisma.position.create({ data: position })));
}

/**
 * Create sample trade logs
 * @param {Array} users - Array of created users
 */
async function createTradeLogs(users) {
  const tradeLogs = [];
  const symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'LTCUSDT', 'BNBUSDT'];
  const types = ['Buy', 'Sell'];
  const statuses = ['Success', 'Pending', 'Failed'];

  // Create 50 random trade logs
  for (let i = 0; i < 50; i++) {
    const user = users[i % users.length];
    const symbol = symbols[i % symbols.length];
    const type = types[i % 2];
    const status = statuses[Math.floor(Math.random() * statuses.length)];
    
    tradeLogs.push({
      userId: user.id,
      instrument: symbol,
      amount: 0.01 + Math.random() * 2,
      price: symbol.includes('BTC') ? 30000 + Math.random() * 5000 : 
             symbol.includes('ETH') ? 2000 + Math.random() * 500 : 
             100 + Math.random() * 100,
      timestamp: new Date(Date.now() - Math.floor(Math.random() * 30 * 24 * 60 * 60 * 1000)),
      orderId: crypto.randomBytes(8).toString('hex'),
      type,
      status,
    });
  }

  return Promise.all(tradeLogs.map(log => prisma.tradeLog.create({ data: log })));
}

/**
 * Create sample metrics
 */
async function createMetrics() {
  const metricNames = [
    'server.cpu.usage',
    'server.memory.usage',
    'api.response.time',
    'bot.orders.executed',
    'bot.position.profit',
  ];

  const metrics = [];
  const now = Date.now();
  const oneHour = 60 * 60 * 1000;

  // Create 100 sample metrics
  for (let i = 0; i < 100; i++) {
    const name = metricNames[i % metricNames.length];
    
    let value;
    switch (name) {
      case 'server.cpu.usage':
        value = Math.random() * 100;
        break;
      case 'server.memory.usage':
        value = Math.random() * 8192;
        break;
      case 'api.response.time':
        value = Math.random() * 1000;
        break;
      case 'bot.orders.executed':
        value = Math.floor(Math.random() * 50);
        break;
      case 'bot.position.profit':
        value = Math.random() * 1000 - 500;
        break;
      default:
        value = Math.random() * 100;
    }
    
    metrics.push({
      name,
      value,
      recordedAt: new Date(now - (i * oneHour)),
      tags: {
        environment: 'development',
        instance: `instance-${Math.floor(i / 20) + 1}`,
      },
    });
  }

  return Promise.all(metrics.map(metric => prisma.metric.create({ data: metric })));
}

// Run the seed function
main()
  .catch((e) => {
    console.error('❌ Error during seeding:', e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  }); 