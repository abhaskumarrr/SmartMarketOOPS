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

declare namespace Express {
  export interface Request {
    requestId?: string;
  }
}

export {}; 