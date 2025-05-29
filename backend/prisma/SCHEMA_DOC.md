# Database Schema Documentation

## Entity-Relationship Diagram

```
User <1---n> ApiKey
User <1---n> TradeLog
User <1---n> Bot
User <1---n> Position
Bot <1---n> Position

TradeLog: userId, instrument, amount, price, timestamp, orderId, type, status
Position: userId, botId, symbol, side, entryPrice, currentPrice, amount, leverage, takeProfitPrice, stopLossPrice, status, pnl, openedAt, closedAt, metadata
```

## Model Descriptions

### User
- id (PK, UUID)
- name, email (unique), password
- createdAt, updatedAt
- Relations: ApiKey[], TradeLog[], Bot[], Position[]

### ApiKey
- id (PK, UUID)
- key (unique), encryptedData, userId (FK), scopes, expiry, isRevoked
- Indexes: userId, key

### TradeLog
- id (PK, UUID)
- userId (FK), instrument, amount, price, timestamp, orderId, type, status
- Indexes: [userId, timestamp], [instrument, timestamp]

### Bot
- id (PK, UUID)
- userId (FK), name, symbol, strategy, timeframe, parameters, isActive, createdAt, updatedAt
- Indexes: userId, symbol, isActive

### Position
- id (PK, UUID)
- userId (FK), botId (FK), symbol, side, entryPrice, currentPrice, amount, leverage, takeProfitPrice, stopLossPrice, status, pnl, openedAt, closedAt, metadata
- Indexes: userId, botId, symbol, status

### Metric
- id (PK, UUID)
- name, value, recordedAt, tags
- Indexes: [name, recordedAt]

## Common Access Patterns

- Get all positions for a user: `findMany({ where: { userId } })`
- Get all open positions for a bot: `findMany({ where: { botId, status: 'Open' } })`
- Get trade logs for a user by instrument: `findMany({ where: { userId, instrument } })`
- Get active bots for a user: `findMany({ where: { userId, isActive: true } })`
- Get metrics by name and time: `findMany({ where: { name, recordedAt: { gte, lte } } })`

## Notes
- All sensitive data (API keys) is encrypted at rest.
- Indexes are used for all high-frequency queries.
- Soft delete is implemented for User and Bot.
- Middleware logs all queries and validates data. 