"use strict";
/**
 * Signal Controller
 * Handles API endpoints for trading signal generation and retrieval
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.getLatestSignal = exports.getSignals = exports.generateSignals = void 0;
const signalGenerationService_1 = __importDefault(require("../services/trading/signalGenerationService"));
const signals_1 = require("../types/signals");
const logger_1 = require("../utils/logger");
// Create logger
const logger = (0, logger_1.createLogger)('SignalController');
/**
 * Generate trading signals for a symbol
 * @route POST /api/signals/generate
 */
const generateSignals = async (req, res) => {
    try {
        const { symbol, features, options } = req.body;
        // Validate request body
        if (!symbol) {
            res.status(400).json({
                success: false,
                error: {
                    code: 'MISSING_SYMBOL',
                    message: 'Symbol is required'
                },
                timestamp: new Date().toISOString()
            });
            return;
        }
        if (!features || Object.keys(features).length === 0) {
            res.status(400).json({
                success: false,
                error: {
                    code: 'MISSING_FEATURES',
                    message: 'Market features are required'
                },
                timestamp: new Date().toISOString()
            });
            return;
        }
        // Generate signals
        const signals = await signalGenerationService_1.default.generateSignals(symbol, features, options);
        res.status(200).json({
            success: true,
            data: {
                signals,
                count: signals.length
            },
            timestamp: new Date().toISOString()
        });
    }
    catch (error) {
        logger.error('Error generating signals', {
            error: error instanceof Error ? error.message : String(error)
        });
        res.status(500).json({
            success: false,
            error: {
                code: 'SIGNAL_GENERATION_ERROR',
                message: error instanceof Error ? error.message : 'Failed to generate signals'
            },
            timestamp: new Date().toISOString()
        });
    }
};
exports.generateSignals = generateSignals;
/**
 * Get signals based on filter criteria
 * @route GET /api/signals
 */
const getSignals = async (req, res) => {
    try {
        const { symbol, types, directions, minStrength, timeframes, minConfidenceScore, fromTimestamp, toTimestamp, status = 'active' } = req.query;
        // Build filter criteria
        const criteria = {};
        if (symbol) {
            criteria.symbol = String(symbol);
        }
        if (types) {
            criteria.types = (Array.isArray(types) ? types : [types])
                .map(t => String(t).toUpperCase())
                .filter(t => Object.values(signals_1.SignalType).includes(t));
        }
        if (directions) {
            criteria.directions = (Array.isArray(directions) ? directions : [directions])
                .map(d => String(d).toUpperCase())
                .filter(d => Object.values(signals_1.SignalDirection).includes(d));
        }
        if (minStrength && Object.values(signals_1.SignalStrength).includes(String(minStrength).toUpperCase())) {
            criteria.minStrength = String(minStrength).toUpperCase();
        }
        if (timeframes) {
            criteria.timeframes = (Array.isArray(timeframes) ? timeframes : [timeframes])
                .map(t => String(t).toUpperCase())
                .filter(t => Object.values(signals_1.SignalTimeframe).includes(t));
        }
        if (minConfidenceScore !== undefined) {
            const score = parseInt(String(minConfidenceScore), 10);
            if (!isNaN(score)) {
                criteria.minConfidenceScore = score;
            }
        }
        if (fromTimestamp) {
            criteria.fromTimestamp = String(fromTimestamp);
        }
        if (toTimestamp) {
            criteria.toTimestamp = String(toTimestamp);
        }
        if (status) {
            criteria.status = String(status);
        }
        // Get signals
        const signals = await signalGenerationService_1.default.getSignals(criteria);
        res.status(200).json({
            success: true,
            data: {
                signals,
                count: signals.length,
                criteria
            },
            timestamp: new Date().toISOString()
        });
    }
    catch (error) {
        logger.error('Error getting signals', {
            error: error instanceof Error ? error.message : String(error)
        });
        res.status(500).json({
            success: false,
            error: {
                code: 'SIGNAL_RETRIEVAL_ERROR',
                message: error instanceof Error ? error.message : 'Failed to retrieve signals'
            },
            timestamp: new Date().toISOString()
        });
    }
};
exports.getSignals = getSignals;
/**
 * Get latest signal for a symbol
 * @route GET /api/signals/:symbol/latest
 */
const getLatestSignal = async (req, res) => {
    try {
        const { symbol } = req.params;
        if (!symbol) {
            res.status(400).json({
                success: false,
                error: {
                    code: 'MISSING_SYMBOL',
                    message: 'Symbol is required'
                },
                timestamp: new Date().toISOString()
            });
            return;
        }
        // Get latest signal
        const signal = await signalGenerationService_1.default.getLatestSignal(symbol);
        if (!signal) {
            res.status(404).json({
                success: false,
                error: {
                    code: 'SIGNAL_NOT_FOUND',
                    message: `No signals found for symbol ${symbol}`
                },
                timestamp: new Date().toISOString()
            });
            return;
        }
        res.status(200).json({
            success: true,
            data: signal,
            timestamp: new Date().toISOString()
        });
    }
    catch (error) {
        logger.error(`Error getting latest signal`, {
            error: error instanceof Error ? error.message : String(error)
        });
        res.status(500).json({
            success: false,
            error: {
                code: 'SIGNAL_RETRIEVAL_ERROR',
                message: error instanceof Error ? error.message : 'Failed to retrieve latest signal'
            },
            timestamp: new Date().toISOString()
        });
    }
};
exports.getLatestSignal = getLatestSignal;
//# sourceMappingURL=signalController.js.map