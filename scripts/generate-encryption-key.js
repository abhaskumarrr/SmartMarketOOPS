#!/usr/bin/env node

/**
 * Script to generate a secure encryption key for SMOOPs trading bot
 * This key is used for encrypting sensitive information like API keys
 */

const crypto = require('crypto');
const chalk = require('chalk');
const fs = require('fs');
const path = require('path');
const readline = require('readline');

// Generate a secure random key of specified length
function generateSecureKey(byteLength = 32) {
  return crypto.randomBytes(byteLength).toString('hex');
}

// Update the .env file with the new key
function updateEnvFile(key) {
  const envPath = path.join(__dirname, '..', '.env');
  let envContent;
  
  try {
    envContent = fs.readFileSync(envPath, 'utf8');
  } catch (error) {
    console.error(chalk.red('Error reading .env file:'), error.message);
    console.log(chalk.yellow('Your generated key is:'), chalk.bold(key));
    console.log(chalk.yellow('Add this to your .env file manually as ENCRYPTION_MASTER_KEY=<key>'));
    return;
  }

  // Check if ENCRYPTION_MASTER_KEY already exists
  if (envContent.includes('ENCRYPTION_MASTER_KEY=')) {
    // Replace the existing key
    const newEnvContent = envContent.replace(
      /ENCRYPTION_MASTER_KEY=.*/,
      `ENCRYPTION_MASTER_KEY=${key}`
    );
    
    try {
      fs.writeFileSync(envPath, newEnvContent);
      console.log(chalk.green('Successfully updated encryption key in .env file!'));
    } catch (error) {
      console.error(chalk.red('Error writing to .env file:'), error.message);
      console.log(chalk.yellow('Your generated key is:'), chalk.bold(key));
      console.log(chalk.yellow('Add this to your .env file manually as ENCRYPTION_MASTER_KEY=<key>'));
    }
  } else {
    // Append the key to the file
    try {
      fs.appendFileSync(envPath, `\nENCRYPTION_MASTER_KEY=${key}\n`);
      console.log(chalk.green('Successfully added encryption key to .env file!'));
    } catch (error) {
      console.error(chalk.red('Error writing to .env file:'), error.message);
      console.log(chalk.yellow('Your generated key is:'), chalk.bold(key));
      console.log(chalk.yellow('Add this to your .env file manually as ENCRYPTION_MASTER_KEY=<key>'));
    }
  }
}

// Main function
function main() {
  console.log(chalk.bold('SMOOPs Encryption Key Generator'));
  console.log('===============================');
  console.log('');
  console.log(chalk.yellow('This utility generates a secure random key for encrypting sensitive data.'));
  console.log(chalk.red.bold('WARNING: Changing the encryption key will make existing encrypted data inaccessible!'));
  console.log('');
  
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  // Check if .env file exists
  const envPath = path.join(__dirname, '..', '.env');
  const envExists = fs.existsSync(envPath);
  
  if (envExists) {
    const envContent = fs.readFileSync(envPath, 'utf8');
    const hasKey = envContent.includes('ENCRYPTION_MASTER_KEY=');
    
    if (hasKey) {
      rl.question(chalk.yellow('An encryption key already exists. Replace it? (y/N): '), (answer) => {
        if (answer.toLowerCase() === 'y' || answer.toLowerCase() === 'yes') {
          const key = generateSecureKey();
          updateEnvFile(key);
          console.log(chalk.green('Generated new encryption key:'), chalk.bold(key));
        } else {
          console.log(chalk.blue('Operation cancelled. Existing key preserved.'));
        }
        rl.close();
      });
    } else {
      // No key exists, but .env does
      console.log(chalk.blue('No encryption key found in .env file. Generating a new one...'));
      const key = generateSecureKey();
      updateEnvFile(key);
      console.log(chalk.green('Generated new encryption key:'), chalk.bold(key));
      rl.close();
    }
  } else {
    // No .env file exists
    console.log(chalk.yellow('No .env file found. Generating key only...'));
    const key = generateSecureKey();
    console.log(chalk.green('Generated new encryption key:'), chalk.bold(key));
    console.log(chalk.yellow('Copy this key and add it to your .env file as ENCRYPTION_MASTER_KEY=<key>'));
    rl.close();
  }
}

// Run the script
main(); 