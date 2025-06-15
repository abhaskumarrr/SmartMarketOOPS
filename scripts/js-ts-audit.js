#!/usr/bin/env node

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const util = require('util');
const exec = util.promisify(require('child_process').exec);

// ANSI colors for output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m'
};

// Configuration
const config = {
  excludeDirs: ['node_modules', 'dist', 'build', '.next', 'coverage'],
  fileExtensions: ['.js', '.jsx', '.ts', '.tsx']
};

// Helper function to format output
function log(message, color = '') {
  console.log(color + message + colors.reset);
}

// Find all JS/TS files
function findJstsFiles(dir) {
  let results = [];
  const files = fs.readdirSync(dir);

  files.forEach(file => {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);

    if (stat.isDirectory() && !config.excludeDirs.includes(file)) {
      results = results.concat(findJstsFiles(filePath));
    } else if (config.fileExtensions.includes(path.extname(file))) {
      results.push(filePath);
    }
  });

  return results;
}

// Run ESLint check
async function runEslint(files) {
  log('\n🔍 Running ESLint...', colors.cyan);
  try {
    const eslintResult = execSync(`npx eslint ${files.join(' ')} --format stylish`, { encoding: 'utf8' });
    log('✅ ESLint completed successfully', colors.green);
    console.log(eslintResult);
  } catch (error) {
    log('⚠️  ESLint found issues:', colors.yellow);
    console.log(error.stdout);
  }
}

// Run TypeScript check
async function runTypeCheck() {
  log('\n📝 Running TypeScript type check...', colors.cyan);
  try {
    const tscResult = execSync('npx tsc --noEmit', { encoding: 'utf8' });
    log('✅ TypeScript check completed successfully', colors.green);
    console.log(tscResult);
  } catch (error) {
    log('⚠️  TypeScript found issues:', colors.yellow);
    console.log(error.stdout);
  }
}

// Check for security vulnerabilities
async function runSecurityCheck() {
  log('\n🔒 Running security audit...', colors.cyan);
  try {
    const auditResult = execSync('npm audit', { encoding: 'utf8' });
    log('✅ Security audit completed successfully', colors.green);
    console.log(auditResult);
  } catch (error) {
    log('⚠️  Security audit found issues:', colors.yellow);
    console.log(error.stdout);
  }
}

// Check bundle size
async function checkBundleSize() {
  log('\n📦 Analyzing bundle size...', colors.cyan);
  try {
    execSync('npx next build', { stdio: 'inherit' });
    log('✅ Bundle size analysis completed', colors.green);
  } catch (error) {
    log('⚠️  Bundle size analysis failed:', colors.yellow);
    console.error(error);
  }
}

// Main function
async function main() {
  log('🚀 Starting JavaScript/TypeScript code audit...', colors.bright);

  // Find all JS/TS files
  const files = findJstsFiles('.');
  log(`\nFound ${files.length} JavaScript/TypeScript files to analyze`, colors.bright);

  // Run checks
  await runEslint(files);
  await runTypeCheck();
  await runSecurityCheck();
  await checkBundleSize();

  log('\n✨ Audit completed!', colors.bright);
}

main().catch(error => {
  console.error('Error during audit:', error);
  process.exit(1);
}); 