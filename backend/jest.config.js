/**
 * Jest configuration for backend tests
 */

module.exports = {
  // Use typescript preset for ts-jest
  preset: 'ts-jest',
  
  // Use Node.js as the test environment
  testEnvironment: 'node',
  
  // Define test match patterns
  testMatch: [
    '**/tests/**/*.test.ts',
    '**/tests/**/*.test.js'
  ],
  
  // Files to ignore
  testPathIgnorePatterns: [
    '/node_modules/',
    '/dist/'
  ],
  
  // Set timeout for tests
  testTimeout: 30000,
  
  // Transform TypeScript files with ts-jest
  transform: {
    '^.+\\.tsx?$': ['ts-jest', {
      tsconfig: 'tsconfig.json',
    }]
  },
  
  // Coverage configuration
  collectCoverageFrom: [
    'src/**/*.{js,ts}',
    '!src/**/*.d.ts',
    '!src/server.{js,ts}'
  ],
  
  // Code coverage thresholds
  coverageThreshold: {
    global: {
      branches: 70,
      functions: 70,
      lines: 70,
      statements: 70
    }
  },
  
  // Coverage directory
  coverageDirectory: 'coverage',
  
  // Module name mapper for paths
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1'
  },
  
  // Setup files before tests
  setupFilesAfterEnv: [
    '<rootDir>/tests/setup.js'
  ],
  
  // Verbose output
  verbose: true
}; 