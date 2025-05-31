/**
 * Trading Bot Controller
 * Handles trading bot configuration and control endpoints
 */
import { Request, Response } from 'express';
/**
 * Create a new trading bot
 * @route POST /api/bots
 * @access Private
 */
export declare const createBot: (req: Request, res: Response) => Promise<Response<any, Record<string, any>>>;
/**
 * Get all trading bots for current user
 * @route GET /api/bots
 * @access Private
 */
export declare const getBots: (req: Request, res: Response) => Promise<Response<any, Record<string, any>>>;
/**
 * Get a specific trading bot
 * @route GET /api/bots/:id
 * @access Private
 */
export declare const getBot: (req: Request, res: Response) => Promise<Response<any, Record<string, any>>>;
/**
 * Update a trading bot
 * @route PUT /api/bots/:id
 * @access Private
 */
export declare const updateBot: (req: Request, res: Response) => Promise<Response<any, Record<string, any>>>;
/**
 * Delete a trading bot
 * @route DELETE /api/bots/:id
 * @access Private
 */
export declare const deleteBot: (req: Request, res: Response) => Promise<Response<any, Record<string, any>>>;
/**
 * Start a trading bot
 * @route POST /api/bots/:id/start
 * @access Private
 */
export declare const startBot: (req: Request, res: Response) => Promise<Response<any, Record<string, any>>>;
/**
 * Stop a trading bot
 * @route POST /api/bots/:id/stop
 * @access Private
 */
export declare const stopBot: (req: Request, res: Response) => Promise<Response<any, Record<string, any>>>;
/**
 * Pause a trading bot
 * @route POST /api/bots/:id/pause
 * @access Private
 */
export declare const pauseBot: (req: Request, res: Response) => Promise<Response<any, Record<string, any>>>;
/**
 * Get bot status
 * @route GET /api/bots/:id/status
 * @access Private
 */
export declare const getBotStatus: (req: Request, res: Response) => Promise<Response<any, Record<string, any>>>;
/**
 * Configure risk settings for a bot
 * @route POST /api/bots/:id/risk
 * @access Private
 */
export declare const configureBotRiskSettings: (req: Request, res: Response) => Promise<Response<any, Record<string, any>>>;
/**
 * Update bot health status
 * @route POST /api/bots/:id/health
 * @access Private
 */
export declare const updateBotHealth: (req: Request, res: Response) => Promise<Response<any, Record<string, any>>>;
