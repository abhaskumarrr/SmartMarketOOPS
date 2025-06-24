#!/usr/bin/env node

/**
 * Comprehensive JavaScript and TypeScript Audit Script
 * Checks for code quality, security issues, and potential problems
 */

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

// Helper function to execute commands safely
async function safeExec(command, options = {}) {
  try {
    const { stdout, stderr } = await exec(command, options);
    return { success: true, output: stdout, error: stderr };
  } catch (error) {
    return { success: false, output: '', error: error.message };
  }
}

// Find all JS/TS files
function findFiles(dir, files = []) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    
    if (config.excludeDirs.some(excluded => entry.name === excluded)) {
      continue;
    }

    if (entry.isDirectory()) {
      findFiles(fullPath, files);
    } else if (config.fileExtensions.includes(path.extname(entry.name))) {
      files.push(fullPath);
    }
  }

  return files;
}

// Analyze file complexity
function analyzeComplexity(filePath) {
  const content = fs.readFileSync(filePath, 'utf8');
  const lines = content.split('\n');
  
  return {
    lines: lines.length,
    functions: (content.match(/function\s+\w+\s*\(|const\s+\w+\s*=\s*\(|=>\s*{/g) || []).length,
    classes: (content.match(/class\s+\w+/g) || []).length,
    todos: (content.match(/\/\/\s*TODO|\/\*\s*TODO/g) || []).length,
    consoleStatements: (content.match(/console\.(log|warn|error|info|debug)/g) || []).length
  };
}

// Main audit function
async function runAudit() {
  log('\n🔍 Starting comprehensive JS/TS files audit...', colors.bright);

  // 1. Find all JS/TS files
  log('\n📁 Scanning for JavaScript and TypeScript files...', colors.cyan);
  const files = findFiles(process.cwd());
  log(`Found ${files.length} files to analyze.`, colors.green);

  // 2. Run ESLint
  log('\n🔬 Running ESLint...', colors.cyan);
  const eslintResult = await safeExec('npx eslint . --ext .js,.jsx,.ts,.tsx --max-warnings=0');
  if (eslintResult.success) {
    log('✅ ESLint check passed', colors.green);
  } else {
    log('⚠️ ESLint found some issues:', colors.yellow);
    console.log(eslintResult.error);
  }

  // 3. Run npm audit
  log('\n🔒 Running security audit...', colors.cyan);
  const npmAuditResult = await safeExec('npm audit');
  if (npmAuditResult.success) {
    log('✅ No security vulnerabilities found', colors.green);
  } else {
    log('⚠️ Security vulnerabilities detected:', colors.yellow);
    console.log(npmAuditResult.error);
  }

  // 4. Analyze code complexity
  log('\n📊 Analyzing code complexity...', colors.cyan);
  let totalStats = {
    files: files.length,
    lines: 0,
    functions: 0,
    classes: 0,
    todos: 0,
    consoleStatements: 0
  };

  const complexFiles = [];

  files.forEach(file => {
    const stats = analyzeComplexity(file);
    totalStats.lines += stats.lines;
    totalStats.functions += stats.functions;
    totalStats.classes += stats.classes;
    totalStats.todos += stats.todos;
    totalStats.consoleStatements += stats.consoleStatements;

    if (stats.lines > 300 || stats.functions > 20) {
      complexFiles.push({
        file: path.relative(process.cwd(), file),
        ...stats
      });
    }
  });

  // Print statistics
  log('\n📈 Code Statistics:', colors.bright);
  console.table({
    'Total Files': totalStats.files,
    'Total Lines': totalStats.lines,
    'Total Functions': totalStats.functions,
    'Total Classes': totalStats.classes,
    'TODO Comments': totalStats.todos,
    'Console Statements': totalStats.consoleStatements
  });

  // Report complex files
  if (complexFiles.length > 0) {
    log('\n⚠️ Files that might need attention:', colors.yellow);
    console.table(complexFiles);
  }

  // 5. Check for potential memory leaks
  log('\n🔍 Checking for potential memory leak patterns...', colors.cyan);
  const memoryLeakPatterns = [
    'setInterval\\([^,]+\\)',
    'setTimeout\\([^,]+\\)',
    'new\\s+Worker\\(',
    'addEventListener\\(',
    '\\.on\\(\'',
    'new\\s+WebSocket\\('
  ];

  const potentialLeaks = [];
  files.forEach(file => {
    const content = fs.readFileSync(file, 'utf8');
    memoryLeakPatterns.forEach(pattern => {
      const matches = content.match(new RegExp(pattern, 'g'));
      if (matches) {
        potentialLeaks.push({
          file: path.relative(process.cwd(), file),
          pattern: pattern,
          occurrences: matches.length
        });
      }
    });
  });

  if (potentialLeaks.length > 0) {
    log('\n⚠️ Patterns that might cause memory leaks:', colors.yellow);
    console.table(potentialLeaks);
  }

  // Final summary
  log('\n✨ Audit Summary:', colors.bright);
  log(`- ${totalStats.files} files analyzed`, colors.cyan);
  log(`- ${complexFiles.length} complex files identified`, colors.cyan);
  log(`- ${potentialLeaks.length} potential memory leak patterns found`, colors.cyan);
  log(`- ${totalStats.todos} TODO comments to address`, colors.cyan);
  
  if (totalStats.consoleStatements > 0) {
    log(`⚠️ Found ${totalStats.consoleStatements} console statements - consider removing in production`, colors.yellow);
  }

  log('\n✅ Audit completed!', colors.green);
}

// Run the audit
runAudit().catch(error => {
  log('\n❌ Error during audit:', colors.red);
  console.error(error);
  process.exit(1);
}); 