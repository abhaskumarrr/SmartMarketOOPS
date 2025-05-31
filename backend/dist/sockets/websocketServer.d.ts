/**
 * Initialize WebSocket server
 * @param {object} httpServer - HTTP server instance
 * @returns {object} - Socket.io server instance
 */
export function initializeWebsocketServer(httpServer: object): object;
/**
 * Broadcast market data to subscribed clients
 * @param {string} symbol - Market symbol
 * @param {object} data - Market data to broadcast
 */
export function broadcastMarketData(symbol: string, data: object): void;
/**
 * Broadcast bot update to the bot owner
 * @param {string} botId - Bot ID
 * @param {object} data - Bot data to broadcast
 */
export function broadcastBotUpdate(botId: string, data: object): void;
/**
 * Broadcast trading signal to all users
 * @param {object} signal - Trading signal data
 */
export function broadcastSignal(signal: object): void;
export const realTimeRouter: import("express-serve-static-core").Router;
