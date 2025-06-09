/**
 * WebSocket Server
 * Provides real-time data streaming to clients
 */

const { Server } = require('socket.io');
const { instrument } = require('@socket.io/admin-ui');
const prisma = require('../utils/prismaClient');
const { verifyAccessToken } = require('../utils/jwt');
const { decrypt } = require('../utils/encryption');
const express = require('express');
const bodyParser = require('body-parser');

// Store active connections
const activeConnections = new Map();

// Store market data subscriptions
const marketDataSubscriptions = new Map();

// --- Real-time Data Ingestion from ML Backend ---
const realTimeRouter = express.Router();

realTimeRouter.use(bodyParser.json());

// POST /realtime/market-data
realTimeRouter.post('/market-data', async (req, res) => {
  try {
    const { symbol, data } = req.body;
    if (!symbol || !data) {
      return res.status(400).json({ error: 'Missing symbol or data' });
    }
    // Broadcast to all clients subscribed to this symbol
    if (marketDataSubscriptions.has(symbol)) {
      for (const socketId of marketDataSubscriptions.get(symbol)) {
        const userConnection = activeConnections.get(socketId);
        if (userConnection) {
          userConnection.socket.emit('market:data', { symbol, data });
        }
      }
    }
    res.status(200).json({ status: 'ok' });
  } catch (error) {
    console.error('Error broadcasting real-time market data:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

/**
 * Initialize WebSocket server
 * @param {object} httpServer - HTTP server instance
 * @returns {object} - Socket.io server instance
 */
function initializeWebsocketServer(httpServer) {
  // Create Socket.IO server
  const io = new Server(httpServer, {
    cors: {
      origin: ['https://admin.socket.io', process.env.FRONTEND_URL || 'http://localhost:3000'],
      credentials: true
    }
  });

  // Setup admin UI for monitoring
  if (process.env.NODE_ENV !== 'production') {
    instrument(io, {
      auth: false,
      mode: 'development',
    });
  }

  // Authentication middleware
  io.use(async (socket, next) => {
    try {
      const token = socket.handshake.auth.token;
      
      if (!token) {
        return next(new Error('Authentication token required'));
      }
      
      const decoded = verifyAccessToken(token);
      
      // Get user from database
      const user = await prisma.user.findUnique({
        where: { id: decoded.userId }
      });
      
      if (!user) {
        return next(new Error('User not found'));
      }
      
      // Store user data in socket
      socket.user = {
        id: user.id,
        name: user.name,
        email: user.email
      };
      
      next();
    } catch (error) {
      next(new Error('Invalid authentication token'));
    }
  });

  // Connection event
  io.on('connection', async (socket) => {
    console.log(`User connected: ${socket.id} (${socket.user.email})`);
    
    // Store the connection
    activeConnections.set(socket.id, {
      userId: socket.user.id,
      socket,
      subscriptions: new Set()
    });
    
    // Handle ping messages from client and respond with pong
    socket.on('ping', () => {
      socket.emit('pong');
    });
    
    // Handle special ping message from ReconnectingWebSocket implementation
    socket.on('message', (data) => {
      if (data === '__ping__') {
        socket.send('__pong__');
      }
    });
    
    // Handle market data subscription
    socket.on('subscribe:market', async (symbol) => {
      try {
        // Add to user's subscriptions
        const userConnection = activeConnections.get(socket.id);
        if (userConnection) {
          userConnection.subscriptions.add(symbol);
        }
        
        // Add to global subscriptions
        if (!marketDataSubscriptions.has(symbol)) {
          marketDataSubscriptions.set(symbol, new Set());
        }
        marketDataSubscriptions.get(symbol).add(socket.id);
        
        console.log(`User ${socket.user.email} subscribed to ${symbol}`);
        
        // Send initial market data
        const initialData = await getInitialMarketData(symbol, socket.user.id);
        socket.emit('market:data', { symbol, data: initialData });
      } catch (error) {
        console.error(`Error in market subscription: ${error.message}`);
        socket.emit('error', { message: 'Failed to subscribe to market data' });
      }
    });
    
    // Handle market data unsubscription
    socket.on('unsubscribe:market', (symbol) => {
      try {
        // Remove from user's subscriptions
        const userConnection = activeConnections.get(socket.id);
        if (userConnection) {
          userConnection.subscriptions.delete(symbol);
        }
        
        // Remove from global subscriptions
        if (marketDataSubscriptions.has(symbol)) {
          marketDataSubscriptions.get(symbol).delete(socket.id);
          if (marketDataSubscriptions.get(symbol).size === 0) {
            marketDataSubscriptions.delete(symbol);
          }
        }
        
        console.log(`User ${socket.user.email} unsubscribed from ${symbol}`);
      } catch (error) {
        console.error(`Error in market unsubscription: ${error.message}`);
      }
    });
    
    // Handle bot status subscription
    socket.on('subscribe:bot', async (botId) => {
      try {
        // Verify user owns this bot
        const bot = await prisma.bot.findFirst({
          where: {
            id: botId,
            userId: socket.user.id
          }
        });
        
        if (!bot) {
          return socket.emit('error', { message: 'Bot not found or access denied' });
        }
        
        // Add to user's subscriptions
        const userConnection = activeConnections.get(socket.id);
        if (userConnection) {
          userConnection.subscriptions.add(`bot:${botId}`);
        }
        
        console.log(`User ${socket.user.email} subscribed to bot ${botId}`);
      } catch (error) {
        console.error(`Error in bot subscription: ${error.message}`);
        socket.emit('error', { message: 'Failed to subscribe to bot updates' });
      }
    });
    
    // Handle bot status unsubscription
    socket.on('unsubscribe:bot', (botId) => {
      try {
        // Remove from user's subscriptions
        const userConnection = activeConnections.get(socket.id);
        if (userConnection) {
          userConnection.subscriptions.delete(`bot:${botId}`);
        }
        
        console.log(`User ${socket.user.email} unsubscribed from bot ${botId}`);
      } catch (error) {
        console.error(`Error in bot unsubscription: ${error.message}`);
      }
    });
    
    // Disconnect event
    socket.on('disconnect', () => {
      console.log(`User disconnected: ${socket.id} (${socket.user.email})`);
      
      // Get the user connection
      const userConnection = activeConnections.get(socket.id);
      
      // Remove from all subscriptions
      if (userConnection) {
        for (const symbol of userConnection.subscriptions) {
          if (symbol.startsWith('bot:')) {
            // Bot subscription - no global list to clean up
            continue;
          }
          
          // Market data subscription - clean up global list
          if (marketDataSubscriptions.has(symbol)) {
            marketDataSubscriptions.get(symbol).delete(socket.id);
            if (marketDataSubscriptions.get(symbol).size === 0) {
              marketDataSubscriptions.delete(symbol);
            }
          }
        }
      }
      
      // Remove from active connections
      activeConnections.delete(socket.id);
    });
  });

  console.log('WebSocket server initialized');
  return io;
}

/**
 * Get initial market data for a symbol using real Delta Exchange + CCXT data
 * @param {string} symbol - Market symbol
 * @param {string} userId - User ID
 * @returns {Promise<object>} - Market data
 */
async function getInitialMarketData(symbol, userId) {
  try {
    console.log(`📡 Fetching real market data for ${symbol} via WebSocket...`);

    // Fetch real market data from our real market data service
    const fetch = require('node-fetch');
    const response = await fetch(`http://localhost:3006/api/real-market-data/${symbol}`);

    if (response.ok) {
      const marketData = await response.json();

      if (marketData.success && marketData.data) {
        console.log(`✅ Real market data for ${symbol}: $${marketData.data.price}`);
        return {
          symbol: marketData.data.symbol,
          price: marketData.data.price,
          volume: marketData.data.volume24h,
          change: marketData.data.change24h,
          changePercentage: marketData.data.changePercentage24h,
          high24h: marketData.data.high24h,
          low24h: marketData.data.low24h,
          source: marketData.data.source,
          timestamp: marketData.data.timestamp
        };
      }
    }

    // Fallback: try to get from general market data endpoint
    const fallbackResponse = await fetch('http://localhost:3006/api/real-market-data');
    if (fallbackResponse.ok) {
      const allMarketData = await fallbackResponse.json();
      if (allMarketData.success && allMarketData.data) {
        const symbolData = allMarketData.data.find(item => item.symbol === symbol);
        if (symbolData) {
          console.log(`✅ Fallback real market data for ${symbol}: $${symbolData.price}`);
          return {
            symbol: symbolData.symbol,
            price: symbolData.price,
            volume: symbolData.volume24h,
            change: symbolData.change24h,
            changePercentage: symbolData.changePercentage24h,
            high24h: symbolData.high24h,
            low24h: symbolData.low24h,
            source: symbolData.source,
            timestamp: symbolData.timestamp
          };
        }
      }
    }

    throw new Error('No real market data available');
  } catch (error) {
    console.error(`❌ Error getting real market data for ${symbol}: ${error.message}`);
    return {
      symbol,
      error: 'Failed to fetch real market data',
      timestamp: new Date().toISOString()
    };
  }
}

/**
 * Broadcast market data to subscribed clients
 * @param {string} symbol - Market symbol
 * @param {object} data - Market data to broadcast
 */
function broadcastMarketData(symbol, data) {
  const subscribers = marketDataSubscriptions.get(symbol);
  
  if (!subscribers || subscribers.size === 0) {
    return;
  }
  
  // Add timestamp to data
  const dataWithTimestamp = {
    ...data,
    timestamp: new Date().toISOString()
  };
  
  // Broadcast to all subscribers
  for (const socketId of subscribers) {
    const connection = activeConnections.get(socketId);
    if (connection && connection.socket) {
      connection.socket.emit('market:data', { symbol, data: dataWithTimestamp });
    }
  }
}

/**
 * Broadcast bot update to the bot owner
 * @param {string} botId - Bot ID
 * @param {object} data - Bot data to broadcast
 */
function broadcastBotUpdate(botId, data) {
  // Find subscribers for this bot
  for (const [socketId, connection] of activeConnections.entries()) {
    if (connection.subscriptions.has(`bot:${botId}`)) {
      connection.socket.emit('bot:update', { botId, data });
    }
  }
}

/**
 * Broadcast trading signal to all users
 * @param {object} signal - Trading signal data
 */
function broadcastSignal(signal) {
  // Add timestamp to signal
  const signalWithTimestamp = {
    ...signal,
    timestamp: new Date().toISOString()
  };

  // Broadcast to all connected users
  for (const [socketId, connection] of activeConnections.entries()) {
    connection.socket.emit('signal', signalWithTimestamp);
  }
}

/**
 * Broadcast real portfolio updates to all clients
 */
async function broadcastRealPortfolioUpdate() {
  try {
    console.log('📊 Broadcasting real portfolio update...');
    const fetch = require('node-fetch');
    const response = await fetch('http://localhost:3006/api/real-market-data/portfolio');

    if (response.ok) {
      const portfolioData = await response.json();
      if (portfolioData.success) {
        // Broadcast to all connected users
        for (const [socketId, connection] of activeConnections.entries()) {
          connection.socket.emit('portfolio:update', portfolioData);
        }
        console.log(`✅ Broadcasted real portfolio update to ${activeConnections.size} clients`);
      }
    }
  } catch (error) {
    console.error('❌ Error broadcasting real portfolio update:', error);
  }
}

/**
 * Broadcast real market data updates to all clients
 */
async function broadcastRealMarketDataUpdate() {
  try {
    console.log('📡 Broadcasting real market data update...');
    const fetch = require('node-fetch');
    const response = await fetch('http://localhost:3006/api/real-market-data');

    if (response.ok) {
      const marketData = await response.json();
      if (marketData.success) {
        // Broadcast to all connected users
        for (const [socketId, connection] of activeConnections.entries()) {
          connection.socket.emit('market:update', marketData);
        }
        console.log(`✅ Broadcasted real market data to ${activeConnections.size} clients`);
      }
    }
  } catch (error) {
    console.error('❌ Error broadcasting real market data:', error);
  }
}

/**
 * Generic broadcast function to all clients
 * @param {string} event - Event name
 * @param {object} data - Data to broadcast
 */
function broadcastToClients(event, data) {
  // Add timestamp to data
  const dataWithTimestamp = {
    ...data,
    timestamp: new Date().toISOString()
  };

  // Broadcast to all connected users
  for (const [socketId, connection] of activeConnections.entries()) {
    connection.socket.emit(event, dataWithTimestamp);
  }
}

// Set up periodic real-time updates every 30 seconds
setInterval(async () => {
  if (activeConnections.size > 0) {
    await broadcastRealMarketDataUpdate();
    await broadcastRealPortfolioUpdate();
  }
}, 30000); // 30 seconds

console.log('🔄 Real-time data broadcasting enabled (30-second intervals)');

module.exports = {
  initializeWebsocketServer,
  broadcastMarketData,
  broadcastBotUpdate,
  broadcastSignal,
  broadcastToClients,
  broadcastRealPortfolioUpdate,
  broadcastRealMarketDataUpdate,
  realTimeRouter
};