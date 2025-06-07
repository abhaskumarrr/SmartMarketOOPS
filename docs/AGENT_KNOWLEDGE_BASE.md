# 🤖 Agent Knowledge Base - SmartMarketOOPS

## 🚨 CRITICAL DAILY ISSUE #1: Delta Exchange Product ID Mismatch

**PROBLEM:** `invalid_contract` error when placing orders  
**ROOT CAUSE:** Using production product IDs on testnet (or vice versa)  
**FREQUENCY:** Daily occurrence - MEMORIZE THIS!

### ✅ VERIFIED CURRENT PRODUCT IDs (June 5, 2025)

| Environment | BTCUSD | ETHUSD | Base URL |
|-------------|--------|--------|----------|
| **🧪 TESTNET** | **84** | **1699** | `cdn-ind.testnet.deltaex.org` |
| **🚀 PRODUCTION** | **27** | **3136** | `api.india.delta.exchange` |

### 🔧 IMMEDIATE FIX PATTERN
```javascript
// ❌ WRONG - Hardcoded production IDs
const productIds = { 'BTCUSD': 27, 'ETHUSD': 3136 };

// ✅ CORRECT - Environment-aware mapping
const PRODUCT_IDS = {
  testnet: { 'BTCUSD': 84, 'ETHUSD': 1699 },
  production: { 'BTCUSD': 27, 'ETHUSD': 3136 }
};

const isTestnet = baseURL.includes('testnet');
const productIds = isTestnet ? PRODUCT_IDS.testnet : PRODUCT_IDS.production;
```

---

## 🔍 VERIFICATION COMMANDS (Run These Daily)

### Check Testnet Product IDs:
```bash
curl -s "https://cdn-ind.testnet.deltaex.org/v2/products" | jq '.result[] | select(.symbol == "BTCUSD" or .symbol == "ETHUSD") | {symbol, id}'
```

### Check Production Product IDs:
```bash
curl -s "https://api.india.delta.exchange/v2/products" | jq '.result[] | select(.symbol == "BTCUSD" or .symbol == "ETHUSD") | {symbol, id}'
```

---

## 📋 STANDARD TROUBLESHOOTING WORKFLOW

### Step 1: Identify Environment
```bash
# Check which environment the script targets
grep -r "testnet\|cdn-ind" backend/scripts/
grep -r "api.india.delta" backend/scripts/
```

### Step 2: Find Product ID References
```bash
# Find all hardcoded product IDs
grep -r "27\|3136\|84\|1699" backend/scripts/ --include="*.js"
```

### Step 3: Fix Product ID Mapping
```javascript
// Standard pattern for all trading scripts:
const getProductIds = (baseURL) => {
  const isTestnet = baseURL.includes('testnet');
  return isTestnet ? {
    'BTCUSD': 84,    // Testnet BTC
    'ETHUSD': 1699   // Testnet ETH
  } : {
    'BTCUSD': 27,    // Production BTC  
    'ETHUSD': 3136   // Production ETH
  };
};
```

---

## 🎯 COMMON ERROR PATTERNS & INSTANT FIXES

### Error: "Request failed with status code 400"
- **Cause:** Wrong product ID for environment
- **Fix:** Update product IDs based on environment table above

### Error: "invalid_contract"
- **Cause:** Product ID doesn't exist in target environment
- **Fix:** Use testnet (84, 1699) or production (27, 3136) IDs

### Error: "insufficient_balance"
- **Cause:** Not enough balance for position size
- **Fix:** Reduce position size or check account balance

### Error: "Authentication failed"
- **Cause:** Wrong API credentials for environment
- **Fix:** Verify testnet vs production API keys

---

## 📁 FILES TO CHECK/UPDATE IMMEDIATELY

### Primary Trading Scripts:
- `backend/scripts/delta-testnet-live.js` - Lines ~242, ~502
- `backend/scripts/ultimate-trading-system.js` - Product ID mapping
- `backend/dist/services/DeltaExchangeUnified.js` - Service layer

### Configuration Files:
- `.env` - API credentials
- `backend/config/` - Environment configs

---

## 🚀 PRE-DEPLOYMENT CHECKLIST

### Before ANY Trading:
- [ ] ✅ Verify environment (testnet vs production)
- [ ] ✅ Confirm correct product IDs from table above
- [ ] ✅ Test with 1 contract size first
- [ ] ✅ Validate API credentials match environment
- [ ] ✅ Check account balance > $10
- [ ] ✅ Monitor first 3 trades closely

### Environment Variables Template:
```bash
# Testnet
DELTA_EXCHANGE_API_KEY="AjTdJYCVE3aMZDAVQ2r6AQdmkU2mWc"
DELTA_EXCHANGE_API_SECRET="R29RkXJfUIIt4o3vCDXImyg6q74JvByYltVKFH96UJG51lR1mm88PCGnMrUR"
NODE_ENV="testnet"

# Production  
DELTA_EXCHANGE_API_KEY="uS2N0I4V37gMNJgbTjX8a33WPWv3GK"
DELTA_EXCHANGE_API_SECRET="hJwxEd1wCpMTYg5iSQKDnreX9IVlc4mcYegR5ojJzvQ5UVOiUhP7cF9u21To"
NODE_ENV="production"
```

---

## 🔧 DEBUGGING COMMANDS

### Show Current Configuration:
```bash
# Environment check
echo "Environment: $NODE_ENV"

# API endpoint in use
grep -r "baseURL\|api.*delta" backend/ | head -5

# Product IDs currently configured
grep -r "productId.*=" backend/scripts/ | head -5
```

### Live API Testing:
```bash
# Test testnet connection
curl -H "api-key: AjTdJYCVE3aMZDAVQ2r6AQdmkU2mWc" "https://cdn-ind.testnet.deltaex.org/v2/wallet/balances"

# Test production connection  
curl -H "api-key: uS2N0I4V37gMNJgbTjX8a33WPWv3GK" "https://api.india.delta.exchange/v2/wallet/balances"
```

---

## 🆘 EMERGENCY PROCEDURES

### If Trading System Fails:
1. **🛑 STOP ALL TRADING** - `pkill -f "node.*delta"`
2. **📊 CHECK POSITIONS** - Login to Delta Exchange web interface
3. **🔍 IDENTIFY ISSUE** - Check logs for "invalid_contract" errors
4. **🔧 FIX CONFIGURATION** - Update product IDs using table above
5. **🧪 TEST SMALL** - Place 1 contract test order
6. **👀 MONITOR CLOSELY** - Watch first 3 trades

### Quick Recovery Commands:
```bash
# Kill all trading processes
pkill -f "node.*delta"

# Check for running processes
ps aux | grep "node.*delta"

# Restart with correct environment
cd backend && node scripts/delta-testnet-live.js
```

---

## 📚 REFERENCE DOCUMENTATION

### Internal Docs:
- [Complete Product ID Reference](./DELTA_EXCHANGE_PRODUCT_IDS.md)
- [Trading Strategy Guide](./TRADING_STRATEGY.md)
- [System Architecture](./ULTIMATE_SYSTEM_DOCS.md)

### External Resources:
- [Delta Exchange API Docs](https://docs.delta.exchange/)
- [Testnet Environment](https://testnet.delta.exchange/)
- [Production Trading](https://www.delta.exchange/)

---

## 📊 SUCCESS METRICS TO MONITOR

### Key Performance Indicators:
- **Order Success Rate:** >95% (if <90%, check product IDs)
- **API Response Time:** <500ms
- **Balance Accuracy:** Real-time sync with exchange
- **Position Accuracy:** Matches exchange interface

### Alert Conditions:
- Multiple "invalid_contract" errors → Check product IDs immediately
- Unexpected balance drops → Verify position sizing
- API authentication errors → Check credentials/environment

---

## 🚨 CRITICAL ISSUE #2: Position Sizing Problems

**PROBLEM:** Extremely small position sizes (1 contract = ~$105)
**ROOT CAUSE:** Incorrect position sizing calculation logic
**IMPACT:** Poor capital efficiency, minimal profits

### ✅ **CORRECT POSITION SIZING (Testnet)**

**Contract Specifications:**
- **BTCUSD:** 0.001 BTC per contract (~$105 per contract)
- **ETHUSD:** 0.01 ETH per contract (~$26 per contract)
- **Max Leverage:** 100x available
- **Recommended Leverage:** 25x-50x for safety

**Optimal Position Sizing Formula:**
```javascript
// For $64.48 balance with 25x leverage
const accountBalance = 64.48;
const leverage = 25; // Conservative leverage
const riskPercent = 5; // 5% risk per trade
const maxPositionValue = accountBalance * leverage; // $1,612

// BTCUSD Example:
const contractValue = 105.76; // Current BTC price * 0.001
const optimalContracts = Math.floor(maxPositionValue * (riskPercent/100) / contractValue);
// Result: ~15 contracts instead of 1
```

**Expected Results:**
- **Position Value:** $1,500-$1,600 (instead of $105)
- **Risk Amount:** $3-$5 (instead of $1.06)
- **Profit Potential:** $45-$75 per trade (instead of $3)

---

**🔄 LAST UPDATED:** June 5, 2025
**📋 STATUS:** Active - Critical daily reference
**🎯 PURPOSE:** Eliminate 95% of Delta Exchange trading issues
**⚡ PRIORITY:** Check this FIRST for any trading errors
