# Delta Exchange Trading Bot Integration Guide

## Overview

This guide provides comprehensive instructions for setting up and maintaining the Delta Exchange trading bot integration in SmartMarketOOPS. Delta Exchange offers a robust API for automated trading, allowing users to place orders programmatically and receive real-time market data.

## Key Features

- **Real-time market data** via WebSocket integration
- **Automated order execution** for both market and limit orders
- **Intelligent risk management** with stop loss and take profit orders
- **Multiple symbol support** including BTC, ETH, SOL, and more
- **Testnet environment** for safe testing without risking real funds

## Configuration

### API Credentials

You need valid Delta Exchange API credentials to use the trading bot:

1. Create an account on [Delta Exchange](https://www.delta.exchange/)
2. Go to API Management section in your account settings
3. Generate a new API key and secret
4. Add these credentials to your `.env` file:

```env
DELTA_EXCHANGE_API_KEY=your_api_key_here
DELTA_EXCHANGE_API_SECRET=your_api_secret_here
DELTA_EXCHANGE_TESTNET=true  # Use false for production
DELTA_EXCHANGE_BASE_URL=https://cdn-ind.testnet.deltaex.org  # For testnet
```

### Product IDs

Delta Exchange uses numeric product IDs to identify trading pairs. These IDs can change, so it's important to keep them updated:

```env
# Current Testnet IDs as of latest API call
DELTA_BTCUSD_PRODUCT_ID=84
DELTA_ETHUSD_PRODUCT_ID=1699
DELTA_SOLUSD_PRODUCT_ID=92572
DELTA_ADAUSD_PRODUCT_ID=101760

# Production IDs
# DELTA_BTCUSD_PRODUCT_ID=27
# DELTA_ETHUSD_PRODUCT_ID=3136
```

You can update these IDs by running:

```bash
node backend/scripts/fetch-delta-products.js
```

## Trading Bot Implementations

### 1. Live Trading Bot

`delta-testnet-live.js` implements a real trading bot that places actual orders on Delta Exchange testnet. Features:

- Daily OHLC zone-based trading strategy
- Real-time order execution
- Position management
- Performance tracking

Usage:

```bash
node backend/scripts/delta-testnet-live.js
```

### 2. Paper Trading Bot

`delta-paper-trading.js` provides a simulation environment for testing strategies without using real funds:

- Uses real market data
- Simulates order execution and fills
- Tracks simulated P&L
- No real orders placed

Usage:

```bash
node backend/scripts/delta-paper-trading.js
```

### 3. Intelligent Trading Bot

`intelligent-delta-live-trading.js` implements an advanced trading algorithm:

- Multiple market indicators
- Machine learning signal validation
- Dynamic position sizing
- Sophisticated risk management

Usage:

```bash
node backend/scripts/intelligent-delta-live-trading.js
```

## Service Architecture

The system is built around a unified Delta Exchange service (`DeltaExchangeUnified.ts`) that provides:

1. **Authentication** - Secure API authentication
2. **Market Data** - Real-time and historical prices
3. **Order Management** - Place, cancel, and modify orders
4. **Position Tracking** - Monitor open positions
5. **WebSocket Integration** - Real-time data streams

## TradingView Integration

Delta Exchange supports webhook integration with TradingView for strategy automation:

1. Set up a webhook URL in your Delta Exchange account
2. Create alerts in TradingView with your webhook URL
3. Configure the payload format to match Delta Exchange requirements

Example TradingView alert message:

```json
{
  "symbol": "{{ticker}}",
  "side": "{{strategy.order.action}}",
  "qty": {{strategy.order.contracts}},
  "trigger_time": "{{timenow}}"
}
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Verify API key and secret
   - Check IP whitelist settings
   - Ensure credentials have required permissions

2. **Order Placement Failures**
   - Verify product ID is correct
   - Check minimum order size requirements
   - Ensure sufficient balance

3. **WebSocket Connection Issues**
   - Check network connectivity
   - Verify WebSocket URL
   - Implement reconnection logic

## Resources

- [Delta Exchange API Documentation](https://docs.delta.exchange/)
- [Delta Exchange Trading Bot GitHub](https://github.com/delta-exchange/trading-bots)
- [TradingView Webhook Integration Guide](https://deltaexchangeindia.freshdesk.com/support/solutions/articles/80001141030-tutorial-trading-view-automation-on-delta-exchange)

## Maintenance

Regular maintenance tasks:

1. Update product IDs by running `fetch-delta-products.js`
2. Test API connectivity with `test-delta-connection.js`
3. Review and update risk parameters in bot configurations
4. Monitor API rate limits and adjust request frequency if needed

---

*Last Updated: Based on Delta Exchange API as of June 2024* 