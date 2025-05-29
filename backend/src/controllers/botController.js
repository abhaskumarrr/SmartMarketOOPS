/**
 * Trading Bot Controller
 * Handles trading bot configuration and control
 */

const prisma = require('../utils/prismaClient');
const axios = require('axios');

// In-memory storage for bot status
const botStatus = new Map();

/**
 * Create a new trading bot configuration
 * @route POST /api/bots
 * @access Private
 */
const createBot = async (req, res) => {
  try {
    const { 
      name, 
      symbol, 
      strategy, 
      timeframe, 
      parameters
    } = req.body;
    const userId = req.user.id;

    // Validate input
    if (!name || !symbol || !strategy || !timeframe) {
      return res.status(400).json({
        success: false,
        message: 'Name, symbol, strategy and timeframe are required'
      });
    }

    // Create bot configuration
    const bot = await prisma.$executeRaw`
      INSERT INTO "Bot" (
        "userId", 
        "name", 
        "symbol", 
        "strategy", 
        "timeframe", 
        "parameters", 
        "isActive", 
        "createdAt", 
        "updatedAt"
      ) 
      VALUES (
        ${userId}, 
        ${name}, 
        ${symbol}, 
        ${strategy}, 
        ${timeframe}, 
        ${JSON.stringify(parameters || {})}, 
        false, 
        NOW(), 
        NOW()
      )
      RETURNING *;
    `;

    res.status(201).json({
      success: true,
      message: 'Bot created successfully',
      data: bot
    });
  } catch (error) {
    console.error('Create bot error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while creating bot',
      error: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

/**
 * Get all trading bots for current user
 * @route GET /api/bots
 * @access Private
 */
const getBots = async (req, res) => {
  try {
    const userId = req.user.id;

    // Get all bots for user
    const bots = await prisma.$queryRaw`
      SELECT * FROM "Bot" WHERE "userId" = ${userId};
    `;

    res.status(200).json({
      success: true,
      data: bots
    });
  } catch (error) {
    console.error('Get bots error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while fetching bots',
      error: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

/**
 * Get a specific trading bot
 * @route GET /api/bots/:id
 * @access Private
 */
const getBot = async (req, res) => {
  try {
    const { id } = req.params;
    const userId = req.user.id;

    // Find bot by id and user id
    const bot = await prisma.$queryRaw`
      SELECT * FROM "Bot" WHERE "id" = ${id} AND "userId" = ${userId};
    `;

    if (!bot || bot.length === 0) {
      return res.status(404).json({
        success: false,
        message: 'Bot not found'
      });
    }

    res.status(200).json({
      success: true,
      data: bot[0]
    });
  } catch (error) {
    console.error('Get bot error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while fetching bot',
      error: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

/**
 * Update a trading bot
 * @route PUT /api/bots/:id
 * @access Private
 */
const updateBot = async (req, res) => {
  try {
    const { id } = req.params;
    const { 
      name, 
      symbol, 
      strategy, 
      timeframe, 
      parameters 
    } = req.body;
    const userId = req.user.id;

    // Find bot by id and user id
    const bot = await prisma.$queryRaw`
      SELECT * FROM "Bot" WHERE "id" = ${id} AND "userId" = ${userId};
    `;

    if (!bot || bot.length === 0) {
      return res.status(404).json({
        success: false,
        message: 'Bot not found'
      });
    }

    // Build update data
    const updateData = {};
    if (name) updateData.name = name;
    if (symbol) updateData.symbol = symbol;
    if (strategy) updateData.strategy = strategy;
    if (timeframe) updateData.timeframe = timeframe;
    if (parameters) updateData.parameters = JSON.stringify(parameters);
    updateData.updatedAt = 'NOW()';

    // Build query
    const setClause = Object.entries(updateData)
      .map(([key, value]) => `"${key}" = ${value === 'NOW()' ? 'NOW()' : `'${value}'`}`)
      .join(', ');

    // Update bot
    const updatedBot = await prisma.$executeRaw`
      UPDATE "Bot" 
      SET ${setClause}
      WHERE "id" = ${id} AND "userId" = ${userId}
      RETURNING *;
    `;

    res.status(200).json({
      success: true,
      message: 'Bot updated successfully',
      data: updatedBot
    });
  } catch (error) {
    console.error('Update bot error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while updating bot',
      error: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

/**
 * Delete a trading bot
 * @route DELETE /api/bots/:id
 * @access Private
 */
const deleteBot = async (req, res) => {
  try {
    const { id } = req.params;
    const userId = req.user.id;

    // Find bot by id and user id
    const bot = await prisma.$queryRaw`
      SELECT * FROM "Bot" WHERE "id" = ${id} AND "userId" = ${userId};
    `;

    if (!bot || bot.length === 0) {
      return res.status(404).json({
        success: false,
        message: 'Bot not found'
      });
    }

    // If bot is active, stop it first
    if (bot[0].isActive) {
      await stopBotProcess(id);
    }

    // Delete bot
    await prisma.$executeRaw`
      DELETE FROM "Bot" WHERE "id" = ${id} AND "userId" = ${userId};
    `;

    res.status(200).json({
      success: true,
      message: 'Bot deleted successfully'
    });
  } catch (error) {
    console.error('Delete bot error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while deleting bot',
      error: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

/**
 * Start a trading bot
 * @route POST /api/bots/:id/start
 * @access Private
 */
const startBot = async (req, res) => {
  try {
    const { id } = req.params;
    const userId = req.user.id;

    // Find bot by id and user id
    const bot = await prisma.$queryRaw`
      SELECT * FROM "Bot" WHERE "id" = ${id} AND "userId" = ${userId};
    `;

    if (!bot || bot.length === 0) {
      return res.status(404).json({
        success: false,
        message: 'Bot not found'
      });
    }

    // If bot is already active
    if (bot[0].isActive) {
      return res.status(400).json({
        success: false,
        message: 'Bot is already running'
      });
    }

    // Start bot process
    const success = await startBotProcess(bot[0]);

    if (!success) {
      return res.status(500).json({
        success: false,
        message: 'Failed to start bot'
      });
    }

    // Update bot status in database
    await prisma.$executeRaw`
      UPDATE "Bot" SET "isActive" = true WHERE "id" = ${id};
    `;

    res.status(200).json({
      success: true,
      message: 'Bot started successfully'
    });
  } catch (error) {
    console.error('Start bot error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while starting bot',
      error: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

/**
 * Stop a trading bot
 * @route POST /api/bots/:id/stop
 * @access Private
 */
const stopBot = async (req, res) => {
  try {
    const { id } = req.params;
    const userId = req.user.id;

    // Find bot by id and user id
    const bot = await prisma.$queryRaw`
      SELECT * FROM "Bot" WHERE "id" = ${id} AND "userId" = ${userId};
    `;

    if (!bot || bot.length === 0) {
      return res.status(404).json({
        success: false,
        message: 'Bot not found'
      });
    }

    // If bot is not active
    if (!bot[0].isActive) {
      return res.status(400).json({
        success: false,
        message: 'Bot is not running'
      });
    }

    // Stop bot process
    const success = await stopBotProcess(id);

    if (!success) {
      return res.status(500).json({
        success: false,
        message: 'Failed to stop bot'
      });
    }

    // Update bot status in database
    await prisma.$executeRaw`
      UPDATE "Bot" SET "isActive" = false WHERE "id" = ${id};
    `;

    res.status(200).json({
      success: true,
      message: 'Bot stopped successfully'
    });
  } catch (error) {
    console.error('Stop bot error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while stopping bot',
      error: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

/**
 * Get status of a trading bot
 * @route GET /api/bots/:id/status
 * @access Private
 */
const getBotStatus = async (req, res) => {
  try {
    const { id } = req.params;
    const userId = req.user.id;

    // Find bot by id and user id
    const bot = await prisma.$queryRaw`
      SELECT * FROM "Bot" WHERE "id" = ${id} AND "userId" = ${userId};
    `;

    if (!bot || bot.length === 0) {
      return res.status(404).json({
        success: false,
        message: 'Bot not found'
      });
    }

    // Get status from in-memory storage
    const status = botStatus.get(id) || {
      isRunning: bot[0].isActive,
      lastUpdate: new Date().toISOString(),
      trades: [],
      errors: []
    };

    res.status(200).json({
      success: true,
      data: {
        id: bot[0].id,
        name: bot[0].name,
        symbol: bot[0].symbol,
        strategy: bot[0].strategy,
        isActive: bot[0].isActive,
        status
      }
    });
  } catch (error) {
    console.error('Get bot status error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while fetching bot status',
      error: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

/**
 * Helper function to start a bot process
 * @param {Object} bot - Bot configuration
 * @returns {Promise<boolean>} - Success status
 */
async function startBotProcess(bot) {
  try {
    // In a real implementation, this would call the ML service to start the bot
    // For now, we'll just update the in-memory status
    botStatus.set(bot.id, {
      isRunning: true,
      lastUpdate: new Date().toISOString(),
      trades: [],
      errors: []
    });

    // Try to call ML service
    try {
      await axios.post(`http://ml:5000/api/bots/start`, {
        botId: bot.id,
        config: {
          symbol: bot.symbol,
          strategy: bot.strategy,
          timeframe: bot.timeframe,
          parameters: bot.parameters
        }
      });
    } catch (error) {
      console.error('Error calling ML service:', error);
      // Continue anyway for demo purposes
    }

    return true;
  } catch (error) {
    console.error('Start bot process error:', error);
    return false;
  }
}

/**
 * Helper function to stop a bot process
 * @param {string} botId - Bot ID
 * @returns {Promise<boolean>} - Success status
 */
async function stopBotProcess(botId) {
  try {
    // Update in-memory status
    const status = botStatus.get(botId);
    if (status) {
      status.isRunning = false;
      status.lastUpdate = new Date().toISOString();
      botStatus.set(botId, status);
    }

    // Try to call ML service
    try {
      await axios.post(`http://ml:5000/api/bots/stop`, {
        botId
      });
    } catch (error) {
      console.error('Error calling ML service:', error);
      // Continue anyway for demo purposes
    }

    return true;
  } catch (error) {
    console.error('Stop bot process error:', error);
    return false;
  }
}

module.exports = {
  createBot,
  getBots,
  getBot,
  updateBot,
  deleteBot,
  startBot,
  stopBot,
  getBotStatus
}; 