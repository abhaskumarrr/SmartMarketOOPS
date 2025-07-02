#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const chalk = require('chalk');

/**
 * Script to fix toast system calls across the frontend
 * Converts toast({ variant: 'destructive' }) to toast.destructive({ })
 */

console.log(chalk.bold.blue('🔧 SmartMarketOOPS Toast System Fixer'));
console.log('=====================================\n');

const frontendDir = path.join(__dirname, '..', 'frontend', 'src');

function findTsxFiles(dir) {
  let files = [];
  const items = fs.readdirSync(dir, { withFileTypes: true });
  
  for (const item of items) {
    const fullPath = path.join(dir, item.name);
    if (item.isDirectory()) {
      files.push(...findTsxFiles(fullPath));
    } else if (item.isFile() && (item.name.endsWith('.tsx') || item.name.endsWith('.ts'))) {
      files.push(fullPath);
    }
  }
  
  return files;
}

function fixToastCalls(content) {
  let modified = false;
  
  // Pattern 1: toast({ variant: 'destructive', ... }) -> toast.destructive({ ... })
  const destructivePattern = /toast\(\s*{\s*([^}]*?)variant:\s*['"]destructive['"][,]?\s*([^}]*?)\s*}\s*\)/g;
  content = content.replace(destructivePattern, (match, before, after) => {
    modified = true;
    const cleanBefore = before.replace(/,$/, '').trim();
    const cleanAfter = after.replace(/^,/, '').trim();
    const params = [cleanBefore, cleanAfter].filter(p => p).join(', ');
    return `toast.destructive({ ${params} })`;
  });
  
  // Pattern 2: toast({ variant: 'success', ... }) -> toast.success({ ... })
  const successPattern = /toast\(\s*{\s*([^}]*?)variant:\s*['"]success['"][,]?\s*([^}]*?)\s*}\s*\)/g;
  content = content.replace(successPattern, (match, before, after) => {
    modified = true;
    const cleanBefore = before.replace(/,$/, '').trim();
    const cleanAfter = after.replace(/^,/, '').trim();
    const params = [cleanBefore, cleanAfter].filter(p => p).join(', ');
    return `toast.success({ ${params} })`;
  });
  
  // Pattern 3: toast({ variant: 'info', ... }) -> toast.info({ ... })
  const infoPattern = /toast\(\s*{\s*([^}]*?)variant:\s*['"]info['"][,]?\s*([^}]*?)\s*}\s*\)/g;
  content = content.replace(infoPattern, (match, before, after) => {
    modified = true;
    const cleanBefore = before.replace(/,$/, '').trim();
    const cleanAfter = after.replace(/^,/, '').trim();
    const params = [cleanBefore, cleanAfter].filter(p => p).join(', ');
    return `toast.info({ ${params} })`;
  });
  
  // Pattern 4: toast({ variant: 'default', ... }) -> toast.default({ ... })
  // Pattern 5: toast({ ... }) (no variant, defaults to default) -> toast.default({ ... })
  const defaultPattern = /toast\(\s*{\s*([^}]*?)(?:variant:\s*['"]default['"][,]?\s*)?([^}]*?)\s*}\s*\)/g;
  content = content.replace(defaultPattern, (match, before, after) => {
    // Skip if already processed by previous patterns
    if (match.includes('toast.')) return match;
    
    modified = true;
    const cleanBefore = before.replace(/,$/, '').trim();
    const cleanAfter = after.replace(/^,/, '').trim();
    const params = [cleanBefore, cleanAfter].filter(p => p).join(', ');
    return `toast.default({ ${params} })`;
  });
  
  return { content, modified };
}

function processFile(filePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf8');
    
    // Skip files that don't contain toast calls
    if (!content.includes('toast(')) {
      return { processed: false };
    }
    
    const result = fixToastCalls(content);
    
    if (result.modified) {
      fs.writeFileSync(filePath, result.content, 'utf8');
      return { processed: true, modified: true };
    }
    
    return { processed: true, modified: false };
  } catch (error) {
    console.error(chalk.red(`Error processing ${filePath}:`), error.message);
    return { processed: false, error: error.message };
  }
}

async function main() {
  console.log(chalk.yellow('🔍 Scanning for TypeScript/TSX files...'));
  const files = findTsxFiles(frontendDir);
  console.log(chalk.green(`Found ${files.length} files to check\n`));
  
  let processedCount = 0;
  let modifiedCount = 0;
  let errorCount = 0;
  
  for (const file of files) {
    const relativePath = path.relative(path.join(__dirname, '..'), file);
    const result = processFile(file);
    
    if (result.error) {
      console.log(chalk.red(`❌ ${relativePath}: ${result.error}`));
      errorCount++;
    } else if (result.modified) {
      console.log(chalk.green(`✅ ${relativePath}: Fixed toast calls`));
      modifiedCount++;
      processedCount++;
    } else if (result.processed) {
      // console.log(chalk.gray(`➖ ${relativePath}: No changes needed`));
      processedCount++;
    }
  }
  
  console.log('\n' + chalk.bold('📊 Summary:'));
  console.log(`${chalk.green('✅ Files processed:')} ${processedCount}`);
  console.log(`${chalk.blue('🔧 Files modified:')} ${modifiedCount}`);
  console.log(`${chalk.red('❌ Errors:')} ${errorCount}`);
  
  if (modifiedCount > 0) {
    console.log('\n' + chalk.green.bold('🎉 Toast system fixes applied successfully!'));
    console.log(chalk.yellow('Next: Run "npm run build" to verify the fixes work.'));
  } else {
    console.log('\n' + chalk.blue('ℹ️ No toast fixes needed.'));
  }
}

if (require.main === module) {
  main().catch(console.error);
}

module.exports = { fixToastCalls }; 