#!/usr/bin/env node

/**
 * MCP Environment Setup Script
 * Helps users set up environment variables for MCP servers
 */

const fs = require('fs');
const path = require('path');
const readline = require('readline');

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
  header: (msg) => console.log(`\n${colors.bold}${colors.cyan}${msg}${colors.reset}\n`),
  question: (msg) => `${colors.yellow}?${colors.reset} ${msg}`
};

// Create readline interface
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

// Promisify readline question
const question = (query) => new Promise(resolve => rl.question(query, resolve));

// Environment variables configuration
const envVars = {
  'AI Providers': {
    OPENAI_API_KEY: {
      description: 'OpenAI API key for GPT models',
      url: 'https://platform.openai.com/api-keys',
      required: false,
      format: 'sk-...'
    },
    ANTHROPIC_API_KEY: {
      description: 'Anthropic API key for Claude models',
      url: 'https://console.anthropic.com/',
      required: false,
      format: 'sk-ant-...'
    },
    OPENROUTER_API_KEY: {
      description: 'OpenRouter API key for multi-model access',
      url: 'https://openrouter.ai/keys',
      required: false,
      format: 'sk-or-...'
    },
    GOOGLE_API_KEY: {
      description: 'Google AI API key for Gemini models',
      url: 'https://makersuite.google.com/',
      required: false,
      format: 'AI...'
    }
  },
  'Search & Web': {
    BRAVE_API_KEY: {
      description: 'Brave Search API key for web search',
      url: 'https://api.search.brave.com/',
      required: false,
      format: 'BSA...'
    }
  },
  'Development': {
    GITHUB_PERSONAL_ACCESS_TOKEN: {
      description: 'GitHub Personal Access Token for repository access',
      url: 'https://github.com/settings/tokens',
      required: false,
      format: 'ghp_...'
    }
  }
};

/**
 * Check if .env file exists
 */
function checkEnvFile() {
  const envPath = path.join(process.cwd(), '.env');
  return fs.existsSync(envPath);
}

/**
 * Read existing .env file
 */
function readEnvFile() {
  const envPath = path.join(process.cwd(), '.env');
  try {
    const content = fs.readFileSync(envPath, 'utf8');
    const vars = {};
    
    content.split('\n').forEach(line => {
      const match = line.match(/^([^=]+)=(.*)$/);
      if (match) {
        vars[match[1]] = match[2];
      }
    });
    
    return vars;
  } catch (error) {
    return {};
  }
}

/**
 * Write environment variables to .env file
 */
function writeEnvFile(vars) {
  const envPath = path.join(process.cwd(), '.env');
  const exampleEnvPath = path.join(process.cwd(), 'example.env');
  
  try {
    // Read existing content or example.env as template
    let content = '';
    
    if (fs.existsSync(envPath)) {
      content = fs.readFileSync(envPath, 'utf8');
    } else if (fs.existsSync(exampleEnvPath)) {
      content = fs.readFileSync(exampleEnvPath, 'utf8');
      log.info('Using example.env as template');
    }
    
    // Add MCP section if it doesn't exist
    if (!content.includes('# MCP SERVERS CONFIGURATION')) {
      content += '\n\n# ============================================================================\n';
      content += '# MCP SERVERS CONFIGURATION\n';
      content += '# ============================================================================\n\n';
    }
    
    // Update or add environment variables
    for (const [key, value] of Object.entries(vars)) {
      const regex = new RegExp(`^${key}=.*$`, 'm');
      if (regex.test(content)) {
        content = content.replace(regex, `${key}=${value}`);
      } else {
        content += `${key}=${value}\n`;
      }
    }
    
    fs.writeFileSync(envPath, content);
    log.success(`Environment variables written to ${envPath}`);
  } catch (error) {
    log.error(`Failed to write .env file: ${error.message}`);
  }
}

/**
 * Setup wizard for environment variables
 */
async function setupWizard() {
  log.header('SmartMarketOOPS MCP Environment Setup Wizard');
  
  console.log('This wizard will help you set up API keys for MCP servers.');
  console.log('You can skip any API keys you don\'t have or don\'t want to configure now.\n');
  
  const existingVars = readEnvFile();
  const newVars = {};
  
  for (const [category, vars] of Object.entries(envVars)) {
    log.header(`${category} Configuration`);
    
    for (const [varName, config] of Object.entries(vars)) {
      console.log(`\n${colors.bold}${varName}${colors.reset}`);
      console.log(`Description: ${config.description}`);
      console.log(`Get your key at: ${colors.cyan}${config.url}${colors.reset}`);
      console.log(`Format: ${config.format}`);
      
      if (existingVars[varName] && !existingVars[varName].includes('_HERE')) {
        console.log(`Current value: ${colors.green}[CONFIGURED]${colors.reset}`);
        const update = await question(log.question('Update this value? (y/N): '));
        if (update.toLowerCase() !== 'y') {
          continue;
        }
      }
      
      const value = await question(log.question(`Enter ${varName} (or press Enter to skip): `));
      
      if (value.trim()) {
        newVars[varName] = value.trim();
        log.success(`${varName} configured`);
      } else {
        log.info(`${varName} skipped`);
      }
    }
  }
  
  if (Object.keys(newVars).length > 0) {
    console.log(`\n${colors.bold}Summary:${colors.reset}`);
    for (const key of Object.keys(newVars)) {
      console.log(`  ${colors.green}✓${colors.reset} ${key}`);
    }
    
    const confirm = await question(log.question('\nSave these configurations to .env file? (Y/n): '));
    if (confirm.toLowerCase() !== 'n') {
      writeEnvFile(newVars);
      log.success('Configuration saved successfully!');
    } else {
      log.info('Configuration not saved');
    }
  } else {
    log.info('No new configurations to save');
  }
}

/**
 * Quick setup mode
 */
async function quickSetup() {
  log.header('Quick Setup Mode');
  
  const requiredVars = {
    OPENAI_API_KEY: 'OpenAI API key',
    GITHUB_PERSONAL_ACCESS_TOKEN: 'GitHub Personal Access Token'
  };
  
  const newVars = {};
  
  for (const [varName, description] of Object.entries(requiredVars)) {
    const value = await question(log.question(`Enter ${description} (optional): `));
    if (value.trim()) {
      newVars[varName] = value.trim();
    }
  }
  
  if (Object.keys(newVars).length > 0) {
    writeEnvFile(newVars);
    log.success('Quick setup completed!');
  } else {
    log.info('No configurations provided');
  }
}

/**
 * Validate existing configuration
 */
function validateConfig() {
  log.header('Validating MCP Configuration');
  
  const existingVars = readEnvFile();
  const issues = [];
  const configured = [];
  
  // Check all possible MCP environment variables
  const allVars = Object.values(envVars).reduce((acc, category) => {
    return { ...acc, ...category };
  }, {});
  
  for (const [varName, config] of Object.entries(allVars)) {
    if (existingVars[varName]) {
      if (existingVars[varName].includes('_HERE') || existingVars[varName] === '') {
        issues.push(`${varName}: Placeholder value detected`);
      } else {
        configured.push(varName);
      }
    } else {
      issues.push(`${varName}: Not configured`);
    }
  }
  
  console.log(`\n${colors.bold}Configured Variables:${colors.reset}`);
  if (configured.length > 0) {
    configured.forEach(var => console.log(`  ${colors.green}✓${colors.reset} ${var}`));
  } else {
    console.log(`  ${colors.yellow}None configured${colors.reset}`);
  }
  
  console.log(`\n${colors.bold}Issues Found:${colors.reset}`);
  if (issues.length > 0) {
    issues.forEach(issue => console.log(`  ${colors.red}✗${colors.reset} ${issue}`));
  } else {
    console.log(`  ${colors.green}No issues found${colors.reset}`);
  }
  
  return issues.length === 0;
}

/**
 * Main function
 */
async function main() {
  const args = process.argv.slice(2);
  const mode = args[0] || 'wizard';
  
  try {
    switch (mode) {
      case 'wizard':
        await setupWizard();
        break;
      case 'quick':
        await quickSetup();
        break;
      case 'validate':
        validateConfig();
        break;
      case 'help':
        console.log(`
Usage: node setup-mcp-env.js [mode]

Modes:
  wizard    - Interactive setup wizard (default)
  quick     - Quick setup for essential keys only
  validate  - Validate existing configuration
  help      - Show this help message

Examples:
  node setup-mcp-env.js wizard
  node setup-mcp-env.js quick
  node setup-mcp-env.js validate
        `);
        break;
      default:
        log.error(`Unknown mode: ${mode}. Use 'help' for usage information.`);
        process.exit(1);
    }
  } catch (error) {
    log.error(`Setup failed: ${error.message}`);
    process.exit(1);
  } finally {
    rl.close();
  }
}

// Run the setup
if (require.main === module) {
  main();
}

module.exports = { main };