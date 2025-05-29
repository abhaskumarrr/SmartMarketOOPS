/**
 * Delta Exchange API Mock Service
 * Used for unit testing
 */
const nock = require('nock');

// Base URLs
const MAINNET_BASE_URL = 'https://api.delta.exchange';
const TESTNET_BASE_URL = 'https://testnet-api.delta.exchange';

/**
 * Create a mock for Delta Exchange API
 * @param {boolean} isTestnet - Whether to mock testnet or mainnet
 * @returns {object} - Nock object with common mock methods
 */
function createDeltaApiMock(isTestnet = true) {
  const baseUrl = isTestnet ? TESTNET_BASE_URL : MAINNET_BASE_URL;
  const scope = nock(baseUrl);
  
  return {
    // Server time
    mockServerTime() {
      return scope
        .get('/v2/time')
        .reply(200, {
          server_time: Date.now(),
          server_time_iso: new Date().toISOString()
        });
    },
    
    // Products / Markets
    mockGetProducts() {
      return scope
        .get('/v2/products')
        .reply(200, {
          success: true,
          result: [
            {
              id: 1,
              symbol: 'BTCUSDT',
              description: 'Bitcoin Perpetual Futures',
              underlying_asset: { symbol: 'BTC', id: 1 },
              quote_asset: { symbol: 'USDT', id: 2 },
              is_active: true
            },
            {
              id: 2,
              symbol: 'ETHUSDT',
              description: 'Ethereum Perpetual Futures',
              underlying_asset: { symbol: 'ETH', id: 3 },
              quote_asset: { symbol: 'USDT', id: 2 },
              is_active: true
            }
          ]
        });
    },
    
    // Order book
    mockGetOrderBook(productId = 1) {
      return scope
        .get(`/v2/orderbooks/${productId}`)
        .reply(200, {
          success: true,
          result: {
            asks: [
              ['45000.5', '1.2'],
              ['45001.0', '0.8']
            ],
            bids: [
              ['44999.5', '0.5'],
              ['44998.0', '1.0']
            ],
            timestamp: Date.now()
          }
        });
    },
    
    // Recent trades
    mockGetRecentTrades(productId = 1) {
      return scope
        .get('/v2/trades')
        .query(true) // Match any query parameters
        .reply(200, {
          success: true,
          result: [
            {
              id: 123,
              price: '45000.0',
              size: '0.1',
              side: 'buy',
              timestamp: Date.now() - 5000
            },
            {
              id: 124,
              price: '44999.0',
              size: '0.2',
              side: 'sell',
              timestamp: Date.now() - 10000
            }
          ]
        });
    },
    
    // Wallet balance
    mockGetWalletBalance() {
      return scope
        .get('/v2/wallet/balances')
        .reply(200, {
          success: true,
          result: [
            {
              asset: 'USDT',
              available_balance: '10000.0',
              balance: '10000.0'
            },
            {
              asset: 'BTC',
              available_balance: '0.5',
              balance: '0.5'
            }
          ]
        });
    },
    
    // Positions
    mockGetPositions() {
      return scope
        .get('/v2/positions')
        .reply(200, {
          success: true,
          result: [
            {
              id: 1,
              symbol: 'BTCUSDT',
              size: '0.1',
              entry_price: '44000.0',
              mark_price: '45000.0',
              unrealized_pnl: '100.0',
              side: 'long'
            }
          ]
        });
    },
    
    // Orders
    mockGetOrders() {
      return scope
        .get('/v2/orders')
        .query(true) // Match any query parameters
        .reply(200, {
          success: true,
          result: [
            {
              id: 1001,
              symbol: 'BTCUSDT',
              price: '46000.0',
              size: '0.1',
              side: 'buy',
              order_type: 'limit',
              status: 'open'
            }
          ]
        });
    },
    
    // Create order
    mockCreateOrder() {
      return scope
        .post('/v2/orders')
        .reply(201, {
          success: true,
          result: {
            id: 1002,
            symbol: 'BTCUSDT',
            price: '46000.0',
            size: '0.1',
            side: 'buy',
            order_type: 'limit',
            status: 'open',
            created_at: new Date().toISOString()
          }
        });
    },
    
    // Cancel order
    mockCancelOrder(orderId = '1001') {
      return scope
        .delete(`/v2/orders/${orderId}`)
        .reply(200, {
          success: true,
          result: {
            id: parseInt(orderId),
            status: 'cancelled'
          }
        });
    },
    
    // Cancel all orders
    mockCancelAllOrders() {
      return scope
        .delete('/v2/orders')
        .query(true) // Match any query parameters
        .reply(200, {
          success: true,
          result: {
            cancelled_ids: [1001, 1002]
          }
        });
    },
    
    // Order history
    mockGetOrderHistory() {
      return scope
        .get('/v2/orders/history')
        .query(true) // Match any query parameters
        .reply(200, {
          success: true,
          result: [
            {
              id: 901,
              symbol: 'BTCUSDT',
              price: '44000.0',
              size: '0.1',
              side: 'buy',
              status: 'filled',
              created_at: new Date(Date.now() - 86400000).toISOString(), // 1 day ago
              updated_at: new Date(Date.now() - 86000000).toISOString()
            },
            {
              id: 902,
              symbol: 'ETHUSDT',
              price: '3000.0',
              size: '1.0',
              side: 'sell',
              status: 'cancelled',
              created_at: new Date(Date.now() - 172800000).toISOString(), // 2 days ago
              updated_at: new Date(Date.now() - 172000000).toISOString()
            }
          ]
        });
    },
    
    // Trade history
    mockGetTradeHistory() {
      return scope
        .get('/v2/fills')
        .query(true) // Match any query parameters
        .reply(200, {
          success: true,
          result: [
            {
              id: 801,
              order_id: 901,
              symbol: 'BTCUSDT',
              price: '44000.0',
              size: '0.1',
              side: 'buy',
              fee: '0.04',
              fee_asset: 'USDT',
              created_at: new Date(Date.now() - 86000000).toISOString()
            }
          ]
        });
    },
    
    // Error mock for testing error handling
    mockApiError(endpoint, statusCode = 400, errorMessage = 'Bad Request') {
      return scope
        .get(endpoint)
        .reply(statusCode, {
          success: false,
          error: {
            code: statusCode,
            message: errorMessage
          }
        });
    },
    
    // Rate limit error mock
    mockRateLimitError(endpoint) {
      return scope
        .get(endpoint)
        .reply(429, {
          success: false,
          error: {
            code: 429,
            message: 'Rate limit exceeded'
          }
        });
    }
  };
}

module.exports = {
  createDeltaApiMock,
  MAINNET_BASE_URL,
  TESTNET_BASE_URL
}; 