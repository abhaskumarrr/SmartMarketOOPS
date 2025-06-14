---
title: Delta Exchange Product IDs Need Update
labels: bug, high-priority
assignees: 
---

## Delta Exchange Product IDs Have Changed!

The automatic product ID verification has detected that Delta Exchange has updated their product IDs. This needs to be addressed immediately to ensure proper trading functionality.

### Next Steps

1. Review the GitHub Actions output in the [Verify Delta Exchange Product IDs workflow](../../actions/workflows/verify-delta-product-ids.yml)
2. Run the product ID fetcher locally to confirm the changes:
   ```bash
   node backend/scripts/fetch-delta-products.js
   ```
3. Update the product IDs in the `.env` file and `example.env`
4. Update any hardcoded product IDs in the codebase
5. Test trading functionality with updated IDs

### Current Values in Code

| Symbol | Environment | Current ID | 
|--------|------------|------------|
| BTCUSD | Testnet    | 84         |
| ETHUSD | Testnet    | 1699       |
| SOLUSD | Testnet    | 92572      |
| ADAUSD | Testnet    | 101760     |
| BTCUSD | Production | 27         |
| ETHUSD | Production | 3136       |

### Impact

If not addressed, the trading bot may:
- Attempt to place orders for non-existent products
- Trade incorrect assets
- Fail to execute trading strategies
- Encounter API errors

### Documentation

For more information, see:
- [DELTA_EXCHANGE_TESTING_GUIDE.md](../../DELTA_EXCHANGE_TESTING_GUIDE.md)
- [DELTA_EXCHANGE_IMPLEMENTATION_GUIDE.md](../../DELTA_EXCHANGE_IMPLEMENTATION_GUIDE.md)
- [Delta Exchange API Documentation](https://docs.delta.exchange/) 