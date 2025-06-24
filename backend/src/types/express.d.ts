/**
 * Express type extensions
 */

declare global {
  namespace Express {
    interface Request {
      requestId?: string;
      sessionID?: string;
    }
  }
}

export {}; 