#!/usr/bin/env node

/**
 * File Structure Validation Script
 * Validates the organized file structure after cleanup
 */

const fs = require('fs');
const path = require('path');

// Colors for console output
const colors = {
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
  reset: '\x1b[0m',
  bold: '\x1b[1m'
};

// Log functions
const log = {
  info: (msg) => console.log(`${colors.blue}ℹ${colors.reset} ${msg}`),
  success: (msg) => console.log(`${colors.green}✓${colors.reset} ${msg}`),
  error: (msg) => console.log(`${colors.red}✗${colors.reset} ${msg}`),
  warning: (msg) => console.log(`${colors.yellow}⚠${colors.reset} ${msg}`),
  header: (msg) => console.log(`\n${colors.bold}${colors.cyan}${msg}${colors.reset}\n`)
};

// Expected directory structure
const expectedStructure = {
  'backend': {
    type: 'directory',
    required: true,
    children: {
      'src': { type: 'directory', required: true },
      'tests': { type: 'directory', required: true },
      'prisma': { type: 'directory', required: true },
      'package.json': { type: 'file', required: true },
      'tsconfig.json': { type: 'file', required: true }
    }
  },
  'frontend': {
    type: 'directory',
    required: true,
    children: {
      'src': { type: 'directory', required: true },
      'public': { type: 'directory', required: true },
      'package.json': { type: 'file', required: true },
      'next.config.ts': { type: 'file', required: true }
    }
  },
  'ml': {
    type: 'directory',
    required: true,
    children: {
      'src': { type: 'directory', required: true },
      'requirements.txt': { type: 'file', required: true }
    }
  },
  'docs': {
    type: 'directory',
    required: true,
    children: {
      'architecture': { type: 'directory', required: true },
      'api': { type: 'directory', required: true },
      'deployment': { type: 'directory', required: true },
      'development': { type: 'directory', required: true },
      'user': { type: 'directory', required: true }
    }
  },
  'data': {
    type: 'directory',
    required: true,
    children: {
      'backtest': { type: 'directory', required: true },
      'trading': { type: 'directory', required: true },
      'models': { type: 'directory', required: true },
      'sample': { type: 'directory', required: true }
    }
  },
  'scripts': {
    type: 'directory',
    required: true,
    children: {
      'setup': { type: 'directory', required: true },
      'deployment': { type: 'directory', required: true },
      'testing': { type: 'directory', required: true },
      'maintenance': { type: 'directory', required: true }
    }
  },
  'config': {
    type: 'directory',
    required: true,
    children: {
      'development': { type: 'directory', required: true },
      'production': { type: 'directory', required: true },
      'docker': { type: 'directory', required: true }
    }
  },
  'monitoring': {
    type: 'directory',
    required: false
  },
  '.github': {
    type: 'directory',
    required: false
  },
  '.cursor': {
    type: 'directory',
    required: false
  },
  'tools': {
    type: 'directory',
    required: false
  }
};

// Files that should NOT be in root directory
const forbiddenInRoot = [
  /.*backtest.*\.json$/,
  /.*trading.*\.json$/,
  /.*analysis.*\.md$/,
  /.*summary.*\.md$/i,
  /.*guide.*\.md$/i,
  /docker-compose.*\.yml$/,
  /.*\.log$/,
  /.*\.tmp$/
];

// Required files in root
const requiredInRoot = [
  'package.json',
  'README.md',
  '.gitignore',
  'example.env'
];

/**
 * Check if a path exists and get its type
 */
function getPathInfo(filePath) {
  try {
    const stats = fs.statSync(filePath);
    return {
      exists: true,
      type: stats.isDirectory() ? 'directory' : 'file',
      size: stats.size
    };
  } catch (error) {
    return {
      exists: false,
      type: null,
      size: 0
    };
  }
}

/**
 * Validate directory structure recursively
 */
function validateStructure(basePath, structure, currentPath = '') {
  const results = {
    passed: 0,
    failed: 0,
    warnings: 0,
    issues: []
  };

  for (const [name, config] of Object.entries(structure)) {
    const fullPath = path.join(basePath, currentPath, name);
    const pathInfo = getPathInfo(fullPath);
    const relativePath = path.join(currentPath, name);

    if (!pathInfo.exists) {
      if (config.required) {
        results.failed++;
        results.issues.push({
          type: 'error',
          message: `Required ${config.type} missing: ${relativePath}`
        });
      } else {
        results.warnings++;
        results.issues.push({
          type: 'warning',
          message: `Optional ${config.type} missing: ${relativePath}`
        });
      }
      continue;
    }

    if (pathInfo.type !== config.type) {
      results.failed++;
      results.issues.push({
        type: 'error',
        message: `Expected ${config.type} but found ${pathInfo.type}: ${relativePath}`
      });
      continue;
    }

    results.passed++;

    // Recursively validate children if it's a directory
    if (config.children && pathInfo.type === 'directory') {
      const childResults = validateStructure(basePath, config.children, relativePath);
      results.passed += childResults.passed;
      results.failed += childResults.failed;
      results.warnings += childResults.warnings;
      results.issues.push(...childResults.issues);
    }
  }

  return results;
}

/**
 * Check for forbidden files in root directory
 */
function checkRootDirectory() {
  const results = {
    passed: 0,
    failed: 0,
    warnings: 0,
    issues: []
  };

  try {
    const rootFiles = fs.readdirSync('.').filter(item => {
      const stats = fs.statSync(item);
      return stats.isFile();
    });

    // Check for forbidden files
    for (const file of rootFiles) {
      for (const pattern of forbiddenInRoot) {
        if (pattern.test(file)) {
          results.failed++;
          results.issues.push({
            type: 'error',
            message: `Forbidden file in root directory: ${file}`
          });
          break;
        }
      }
    }

    // Check for required files
    for (const requiredFile of requiredInRoot) {
      if (rootFiles.includes(requiredFile)) {
        results.passed++;
      } else {
        results.failed++;
        results.issues.push({
          type: 'error',
          message: `Required file missing in root: ${requiredFile}`
        });
      }
    }

    // Count clean files
    const cleanFiles = rootFiles.filter(file => {
      return !forbiddenInRoot.some(pattern => pattern.test(file));
    });

    if (cleanFiles.length <= 10) {
      results.passed++;
    } else {
      results.warnings++;
      results.issues.push({
        type: 'warning',
        message: `Root directory has ${cleanFiles.length} files (recommended: ≤10)`
      });
    }

  } catch (error) {
    results.failed++;
    results.issues.push({
      type: 'error',
      message: `Failed to read root directory: ${error.message}`
    });
  }

  return results;
}

/**
 * Check for build artifacts that should be ignored
 */
function checkBuildArtifacts() {
  const results = {
    passed: 0,
    failed: 0,
    warnings: 0,
    issues: []
  };

  const artifactPaths = [
    'backend/dist',
    'frontend/.next',
    'node_modules',
    '.venv',
    'venv'
  ];

  for (const artifactPath of artifactPaths) {
    const pathInfo = getPathInfo(artifactPath);
    if (pathInfo.exists) {
      results.warnings++;
      results.issues.push({
        type: 'warning',
        message: `Build artifact found (should be in .gitignore): ${artifactPath}`
      });
    } else {
      results.passed++;
    }
  }

  return results;
}

/**
 * Check .gitignore file
 */
function checkGitignore() {
  const results = {
    passed: 0,
    failed: 0,
    warnings: 0,
    issues: []
  };

  const gitignorePath = '.gitignore';
  const pathInfo = getPathInfo(gitignorePath);

  if (!pathInfo.exists) {
    results.failed++;
    results.issues.push({
      type: 'error',
      message: '.gitignore file is missing'
    });
    return results;
  }

  try {
    const gitignoreContent = fs.readFileSync(gitignorePath, 'utf8');
    const requiredPatterns = [
      'node_modules/',
      '.env',
      'dist/',
      '.next/',
      '*.log',
      '.venv',
      'venv/',
      '*.tmp'
    ];

    for (const pattern of requiredPatterns) {
      if (gitignoreContent.includes(pattern)) {
        results.passed++;
      } else {
        results.warnings++;
        results.issues.push({
          type: 'warning',
          message: `.gitignore missing pattern: ${pattern}`
        });
      }
    }

  } catch (error) {
    results.failed++;
    results.issues.push({
      type: 'error',
      message: `Failed to read .gitignore: ${error.message}`
    });
  }

  return results;
}

/**
 * Generate validation report
 */
function generateReport(results) {
  const totalTests = results.passed + results.failed + results.warnings;
  const successRate = totalTests > 0 ? Math.round((results.passed / totalTests) * 100) : 0;

  log.header('File Structure Validation Report');

  console.log(`Total Tests: ${totalTests}`);
  console.log(`${colors.green}Passed: ${results.passed}${colors.reset}`);
  console.log(`${colors.red}Failed: ${results.failed}${colors.reset}`);
  console.log(`${colors.yellow}Warnings: ${results.warnings}${colors.reset}`);
  console.log(`Success Rate: ${successRate}%\n`);

  if (results.issues.length > 0) {
    log.header('Issues Found');

    const errors = results.issues.filter(issue => issue.type === 'error');
    const warnings = results.issues.filter(issue => issue.type === 'warning');

    if (errors.length > 0) {
      console.log(`${colors.bold}${colors.red}Errors:${colors.reset}`);
      errors.forEach(issue => log.error(issue.message));
      console.log();
    }

    if (warnings.length > 0) {
      console.log(`${colors.bold}${colors.yellow}Warnings:${colors.reset}`);
      warnings.forEach(issue => log.warning(issue.message));
      console.log();
    }
  }

  // Overall assessment
  if (results.failed === 0) {
    if (results.warnings === 0) {
      log.success('🎉 File structure validation passed with no issues!');
    } else {
      log.success('✅ File structure validation passed with minor warnings.');
    }
  } else {
    log.error('❌ File structure validation failed. Please address the errors above.');
  }

  return results.failed === 0;
}

/**
 * Main validation function
 */
function main() {
  log.header('SmartMarketOOPS File Structure Validation');

  const allResults = {
    passed: 0,
    failed: 0,
    warnings: 0,
    issues: []
  };

  // Validate directory structure
  log.info('Validating directory structure...');
  const structureResults = validateStructure('.', expectedStructure);
  
  // Check root directory
  log.info('Checking root directory...');
  const rootResults = checkRootDirectory();
  
  // Check build artifacts
  log.info('Checking for build artifacts...');
  const artifactResults = checkBuildArtifacts();
  
  // Check .gitignore
  log.info('Validating .gitignore...');
  const gitignoreResults = checkGitignore();

  // Combine all results
  const resultSets = [structureResults, rootResults, artifactResults, gitignoreResults];
  for (const result of resultSets) {
    allResults.passed += result.passed;
    allResults.failed += result.failed;
    allResults.warnings += result.warnings;
    allResults.issues.push(...result.issues);
  }

  // Generate report
  const success = generateReport(allResults);
  
  process.exit(success ? 0 : 1);
}

// Run validation
if (require.main === module) {
  main();
}

module.exports = { main };