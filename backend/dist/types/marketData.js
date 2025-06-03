"use strict";
/**
 * Market Data Types and Interfaces
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.SYMBOLS = exports.TIMEFRAMES = void 0;
// Constants
exports.TIMEFRAMES = {
    '1m': 60 * 1000,
    '5m': 5 * 60 * 1000,
    '15m': 15 * 60 * 1000,
    '1h': 60 * 60 * 1000,
    '4h': 4 * 60 * 60 * 1000,
    '1d': 24 * 60 * 60 * 1000,
    '1w': 7 * 24 * 60 * 60 * 1000,
};
exports.SYMBOLS = {
    BTCUSD: 'BTCUSD',
    ETHUSD: 'ETHUSD',
    ADAUSD: 'ADAUSD',
};
//# sourceMappingURL=marketData.js.map