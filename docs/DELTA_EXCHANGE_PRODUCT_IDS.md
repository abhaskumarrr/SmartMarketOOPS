# Delta Exchange Product IDs Reference

## ⚠️ CRITICAL: Environment-Specific Product IDs

**Product IDs are DIFFERENT between testnet and production environments!**

## 🔧 Testnet Environment
**API Base URL:** `https://cdn-ind.testnet.deltaex.org`

### Perpetual Futures (Testnet)
| Symbol | Product ID | Contract Value | Description |
|--------|------------|----------------|-------------|
| BTCUSD | **84** | 0.001 BTC | Bitcoin Perpetual |
| ETHUSD | **1699** | 0.01 ETH | Ethereum Perpetual |
| SOLUSD | **92572** | 1 SOL | Solana Perpetual |
| XRPUSD | **93723** | 1 XRP | Ripple Perpetual |
| DOGEUSD | **93724** | 100 DOGE | Dogecoin Perpetual |
| SHIBUSD | **92570** | 100000 SHIB | Shiba Inu Perpetual |

### Spot Trading (Testnet)
| Symbol | Product ID | Contract Value | Description |
|--------|------------|----------------|-------------|
| BTC_USDT | **2697** | 1 BTC | Bitcoin Spot |
| ETH_USDT | **2698** | 1 ETH | Ethereum Spot |
| DETO_USDT | **1548** | 1 DETO | Delta Token Spot |

## 🚀 Production Environment
**API Base URL:** `https://api.india.delta.exchange`

### Perpetual Futures (Production)
| Symbol | Product ID | Contract Value | Description |
|--------|------------|----------------|-------------|
| BTCUSD | **27** | 0.001 BTC | Bitcoin Perpetual |
| ETHUSD | **3136** | 0.01 ETH | Ethereum Perpetual |
| SOLUSD | **14823** | 1 SOL | Solana Perpetual |
| XRPUSD | **14969** | 1 XRP | Ripple Perpetual |
| DOGEUSD | **14745** | 100 DOGE | Dogecoin Perpetual |
| AVAXUSD | **14830** | 1 AVAX | Avalanche Perpetual |
| BCHUSD | **15001** | 0.01 BCH | Bitcoin Cash Perpetual |
| LTCUSD | **15040** | 0.1 LTC | Litecoin Perpetual |
| LINKUSD | **15041** | 1 LINK | Chainlink Perpetual |

## 📊 Contract Specifications

### Minimum Order Sizes
- **BTC Perpetual:** 1 contract (0.001 BTC)
- **ETH Perpetual:** 1 contract (0.01 ETH)
- **Other Perpetuals:** 1 contract minimum

### Leverage Limits
- **BTC:** Up to 200x leverage
- **ETH:** Up to 200x leverage
- **SOL:** Up to 100x leverage
- **Others:** Up to 100x leverage

## 🔧 Implementation Notes

### For Trading Scripts
```javascript
// Environment-specific product IDs
const PRODUCT_IDS = {
  testnet: {
    'BTCUSD': 84,
    'ETHUSD': 1699
  },
  production: {
    'BTCUSD': 27,
    'ETHUSD': 3136
  }
};

// Use based on environment
const isTestnet = process.env.NODE_ENV === 'testnet';
const productIds = isTestnet ? PRODUCT_IDS.testnet : PRODUCT_IDS.production;
```

### API Endpoints
```javascript
const API_ENDPOINTS = {
  testnet: 'https://cdn-ind.testnet.deltaex.org',
  production: 'https://api.india.delta.exchange'
};
```

## ⚠️ Common Mistakes to Avoid

1. **Using production IDs on testnet** - Will result in `invalid_contract` error
2. **Using testnet IDs on production** - Will result in `invalid_contract` error
3. **Fractional contract sizes** - Must use whole numbers (1, 2, 3, etc.)
4. **Wrong contract values** - BTC is 0.001, ETH is 0.01, others vary

## 🔍 How to Verify Product IDs

### Get All Products (Testnet)
```bash
curl "https://cdn-ind.testnet.deltaex.org/v2/products"
```

### Get All Products (Production)
```bash
curl "https://api.india.delta.exchange/v2/products"
```

### Filter for Perpetual Futures
```bash
curl "https://api.india.delta.exchange/v2/products" | jq '.result[] | select(.contract_type == "perpetual_futures") | {symbol, id, contract_value}'
```

## 📅 Last Updated
**Date:** June 5, 2025  
**Verified:** Both testnet and production environments  
**Status:** ✅ Active and operational

---

**⚠️ Always verify product IDs before deploying to production!**
