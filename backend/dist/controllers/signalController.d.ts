/**
 * Signal Controller
 * Handles API endpoints for trading signal generation and retrieval
 */
import { Request, Response } from 'express';
/**
 * Generate trading signals for a symbol
 * @route POST /api/signals/generate
 */
export declare const generateSignals: (req: Request, res: Response) => Promise<void>;
/**
 * Get signals based on filter criteria
 * @route GET /api/signals
 */
export declare const getSignals: (req: Request, res: Response) => Promise<void>;
/**
 * Get latest signal for a symbol
 * @route GET /api/signals/:symbol/latest
 */
export declare const getLatestSignal: (req: Request, res: Response) => Promise<void>;
