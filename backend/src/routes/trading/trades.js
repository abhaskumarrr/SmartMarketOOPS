const express = require('express');
const router = express.Router();
const { authenticateJWT } = require('../../middleware/authMiddleware');

// Sample trade data
const sampleTrades = [
  {
    id: 'trade-1',
    symbol: 'BTC/USDT',
    side: 'buy',
    type: 'market',
    price: 58432.15,
    quantity: 0.25,
    timestamp: Date.now() - 3600000 * 2,
    status: 'completed',
    fee: 3.65,
    totalValue: 14608.04
  },
  {
    id: 'trade-2',
    symbol: 'ETH/USDT',
    side: 'sell',
    type: 'limit',
    price: 3245.78,
    quantity: 1.5,
    timestamp: Date.now() - 3600000 * 5,
    status: 'completed',
    fee: 2.43,
    totalValue: 4868.67
  },
  {
    id: 'trade-3',
    symbol: 'SOL/USDT',
    side: 'buy',
    type: 'market',
    price: 124.35,
    quantity: 10,
    timestamp: Date.now() - 3600000 * 8,
    status: 'completed',
    fee: 0.62,
    totalValue: 1243.50
  },
  {
    id: 'trade-4',
    symbol: 'BTC/USDT',
    side: 'sell',
    type: 'limit',
    price: 59102.45,
    quantity: 0.15,
    timestamp: Date.now() - 3600000 * 12,
    status: 'completed',
    fee: 2.22,
    totalValue: 8865.37
  },
  {
    id: 'trade-5',
    symbol: 'ETH/USDT',
    side: 'buy',
    type: 'market',
    price: 3198.22,
    quantity: 0.75,
    timestamp: Date.now() - 3600000 * 24,
    status: 'completed',
    fee: 1.20,
    totalValue: 2398.67
  }
];

// Function to handle trade listing with filtering and pagination
const handleTradesList = (req, res, next) => {
  try {
    // Add pagination
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 10;
    const startIndex = (page - 1) * limit;
    const endIndex = page * limit;
    
    // Add filtering
    const symbol = req.query.symbol;
    const side = req.query.side;
    
    let filteredTrades = [...sampleTrades];
    
    if (symbol) {
      filteredTrades = filteredTrades.filter(trade => trade.symbol === symbol);
    }
    
    if (side) {
      filteredTrades = filteredTrades.filter(trade => trade.side === side);
    }
    
    // Add sorting
    const sortBy = req.query.sortBy || 'timestamp';
    const sortDir = req.query.sortDir === 'asc' ? 1 : -1;
    
    filteredTrades.sort((a, b) => {
      if (a[sortBy] < b[sortBy]) return -1 * sortDir;
      if (a[sortBy] > b[sortBy]) return 1 * sortDir;
      return 0;
    });
    
    // Prepare pagination result
    const paginatedTrades = filteredTrades.slice(startIndex, endIndex);
    
    // Return trades with pagination info
    res.json({
      success: true,
      data: paginatedTrades,
      pagination: {
        total: filteredTrades.length,
        page,
        limit,
        pages: Math.ceil(filteredTrades.length / limit)
      }
    });
  } catch (error) {
    next(error);
  }
};

// Function to handle single trade fetch
const handleSingleTrade = (req, res, next) => {
  try {
    const tradeId = req.params.id;
    const trade = sampleTrades.find(t => t.id === tradeId);
    
    if (!trade) {
      return res.status(404).json({
        success: false,
        message: `Trade with ID ${tradeId} not found`
      });
    }
    
    res.json({
      success: true,
      data: trade
    });
  } catch (error) {
    next(error);
  }
};

// Public routes (no auth required)
router.get('/public', handleTradesList);
router.get('/public/:id', handleSingleTrade);

// Authenticated routes
router.get('/', authenticateJWT, handleTradesList);
router.get('/:id', authenticateJWT, handleSingleTrade);

module.exports = router; 