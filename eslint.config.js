const js = require('@eslint/js');
const typescript = require('@typescript-eslint/eslint-plugin');
const typescriptParser = require('@typescript-eslint/parser');

module.exports = [
  js.configs.recommended,
  {
    // Global ignores
    ignores: [
      '**/node_modules/**',
      '**/dist/**',
      '**/build/**',
      '**/.next/**',
      '**/ml/.venv/**',
      '**/ml/__pycache__/**',
      '**/__pycache__/**',
      '**/coverage/**',
      '**/temp/**',
      '**/tmp/**',
      '**/.git/**',
      '**/scripts/testing/**', // Temporarily ignore test scripts during foundation repair
      '**/scripts/maintenance/**', // Temporarily ignore maintenance scripts
      'mcp-servers/**', // Temporarily ignore MCP servers
      '**/generated/**', // Ignore all generated files
      '**/backend/generated/**', // Ignore Prisma generated files
      '**/frontend/.next/**', // Ignore Next.js generated files
      '**/tailwind.config.js', // Ignore Tailwind config
      '**/next.config.js', // Ignore Next.js config
      '**/jest.config.js', // Ignore Jest config
      '**/scripts/setup/**', // Temporarily ignore setup scripts during foundation repair
    ]
  },
  {
    // Test files - Jest environment
    files: ['**/*.test.ts', '**/*.test.js', '**/*.spec.ts', '**/*.spec.js', '**/tests/**/*.ts', '**/tests/**/*.js', '**/backend/tests/setup.js', '**/backend/tests/setup.ts'],
    plugins: {
      '@typescript-eslint': typescript
    },
    languageOptions: {
      parser: typescriptParser,
      globals: {
        // Jest globals
        jest: 'readonly',
        expect: 'readonly',
        describe: 'readonly',
        it: 'readonly',
        test: 'readonly',
        beforeAll: 'readonly',
        beforeEach: 'readonly',
        afterAll: 'readonly',
        afterEach: 'readonly',
        // Node.js globals for test setup
        require: 'readonly',
        module: 'readonly',
        process: 'readonly',
        __dirname: 'readonly',
        __filename: 'readonly',
        console: 'readonly',
        Buffer: 'readonly',
        setTimeout: 'readonly',
        clearTimeout: 'readonly',
        setInterval: 'readonly',
        clearInterval: 'readonly',
        global: 'readonly'
      }
    },
    rules: {
      'no-console': 'off',
      '@typescript-eslint/no-explicit-any': 'warn',
      '@typescript-eslint/no-unused-vars': 'warn'
    }
  },
  {
    // Backend TypeScript files - Node.js environment
    files: ['backend/**/*.ts', 'backend/**/*.js'],
    plugins: {
      '@typescript-eslint': typescript
    },
    languageOptions: {
      parser: typescriptParser,
      parserOptions: {
        ecmaVersion: 'latest',
        sourceType: 'module'
      },
      globals: {
        // Node.js globals
        require: 'readonly',
        module: 'readonly',
        process: 'readonly',
        __dirname: 'readonly',
        __filename: 'readonly',
        console: 'readonly',
        Buffer: 'readonly',
        setTimeout: 'readonly',
        clearTimeout: 'readonly',
        setInterval: 'readonly',
        clearInterval: 'readonly',
        global: 'readonly',
        AbortSignal: 'readonly',
        fetch: 'readonly',
        BufferEncoding: 'readonly'
      }
    },
    rules: {
      // Relaxed rules for foundation repair phase
      '@typescript-eslint/explicit-function-return-type': 'off',
      '@typescript-eslint/no-explicit-any': 'warn',
      '@typescript-eslint/no-unused-vars': 'warn',
      'no-console': 'off', // Allow console in backend
      'no-debugger': 'warn',
      'no-duplicate-imports': 'error',
      'no-unused-vars': 'off',
      'prefer-const': 'error'
    }
  },
  {
    // Frontend TypeScript/React files - Browser environment
    files: ['frontend/**/*.ts', 'frontend/**/*.tsx'],
    plugins: {
      '@typescript-eslint': typescript
    },
    languageOptions: {
      parser: typescriptParser,
      parserOptions: {
        ecmaVersion: 'latest',
        sourceType: 'module',
        ecmaFeatures: {
          jsx: true
        }
      },
      globals: {
        // Browser globals
        window: 'readonly',
        document: 'readonly',
        console: 'readonly',
        fetch: 'readonly',
        localStorage: 'readonly',
        sessionStorage: 'readonly',
        alert: 'readonly',
        setTimeout: 'readonly',
        clearTimeout: 'readonly',
        setInterval: 'readonly',
        clearInterval: 'readonly',
        WebSocket: 'readonly',
        URL: 'readonly',
        URLSearchParams: 'readonly',
        performance: 'readonly',
        PerformanceObserver: 'readonly',
        PerformanceNavigationTiming: 'readonly',
        PerformanceResourceTiming: 'readonly',
        requestAnimationFrame: 'readonly',
        Event: 'readonly',
        CloseEvent: 'readonly',
        RequestInit: 'readonly',
        NodeJS: 'readonly',
        navigator: 'readonly',
        PromiseRejectionEvent: 'readonly',
        // HTML DOM types
        HTMLDivElement: 'readonly',
        HTMLButtonElement: 'readonly',
        HTMLSpanElement: 'readonly',
        HTMLInputElement: 'readonly',
        HTMLTableElement: 'readonly',
        HTMLTableSectionElement: 'readonly',
        HTMLTableRowElement: 'readonly',
        HTMLTableCellElement: 'readonly',
        HTMLTableCaptionElement: 'readonly',
        HTMLParagraphElement: 'readonly',
        // React (might be auto-imported)
        React: 'readonly',
        // Process for Next.js API routes
        process: 'readonly'
      }
    },
    rules: {
      // Relaxed rules for foundation repair phase
      '@typescript-eslint/explicit-function-return-type': 'off',
      '@typescript-eslint/no-explicit-any': 'warn',
      '@typescript-eslint/no-unused-vars': 'warn',
      'no-console': ['warn', { allow: ['warn', 'error'] }],
      'no-debugger': 'warn',
      'no-duplicate-imports': 'error',
      'no-unused-vars': 'off',
      'prefer-const': 'error',
      'react/prop-types': 'off',
      'no-redeclare': 'error'
    }
  },
  {
    // Next.js configuration files
    files: ['**/next.config.js', '**/next.config.ts'],
    languageOptions: {
      globals: {
        require: 'readonly',
        module: 'readonly',
        process: 'readonly',
        __dirname: 'readonly'
      }
    },
    rules: {
      'no-console': 'off'
    }
  },
  {
    // General configuration files - Node.js environment
    files: ['*.config.js', '*.config.ts', 'tailwind.config.js'],
    languageOptions: {
      globals: {
        require: 'readonly',
        module: 'readonly',
        process: 'readonly',
        __dirname: 'readonly'
      }
    },
    rules: {
      'no-console': 'off'
    }
  },
  {
    // Script files - Node.js environment
    files: ['scripts/**/*.js', 'scripts/**/*.ts'],
    plugins: {
      '@typescript-eslint': typescript
    },
    languageOptions: {
      parser: typescriptParser,
      globals: {
        require: 'readonly',
        module: 'readonly',
        process: 'readonly',
        __dirname: 'readonly',
        __filename: 'readonly',
        console: 'readonly',
        Buffer: 'readonly',
        setTimeout: 'readonly',
        clearTimeout: 'readonly',
        setInterval: 'readonly',
        clearInterval: 'readonly'
      }
    },
    rules: {
      'no-console': 'off',
      '@typescript-eslint/no-unused-vars': 'warn',
      'no-unused-vars': 'off'
    }
  }
]; 