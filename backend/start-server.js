/**
 * Server Startup Script
 * Ensures proper environment variables are set before starting the server
 */

const { spawn } = require('child_process');
const path = require('path');

// Default database URL if not specified in environment
const DEFAULT_DATABASE_URL = 'postgresql://postgres:postgres@localhost:5432/smoops?schema=public';

// Set environment variables
process.env.DATABASE_URL = process.env.DATABASE_URL || DEFAULT_DATABASE_URL;

console.log('Starting server with the following configuration:');
console.log(`- DATABASE_URL: ${process.env.DATABASE_URL.replace(/\/\/(.+):(.+)@/, '//******:******@')}`);
console.log(`- NODE_ENV: ${process.env.NODE_ENV || 'development'}`);
console.log(`- PORT: ${process.env.PORT || 3001}`);

// Start the server
const serverProcess = spawn('node', [path.join(__dirname, 'src', 'server.js')], {
  stdio: 'inherit',
  env: process.env
});

// Handle server process events
serverProcess.on('close', (code) => {
  console.log(`Server process exited with code ${code}`);
});

// Handle termination signals to gracefully shut down
process.on('SIGINT', () => {
  console.log('Received SIGINT signal. Shutting down server...');
  serverProcess.kill('SIGINT');
});

process.on('SIGTERM', () => {
  console.log('Received SIGTERM signal. Shutting down server...');
  serverProcess.kill('SIGTERM');
}); 