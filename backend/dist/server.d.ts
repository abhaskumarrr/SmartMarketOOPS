/**
 * Main Server Entry Point
 * Sets up and starts the Express server with API routes
 */
import http from 'http';
import prisma from './utils/prismaClient';
declare const app: import("express-serve-static-core").Express;
declare const server: http.Server<typeof http.IncomingMessage, typeof http.ServerResponse>;
declare const io: any;
export { app, server, io, prisma };
