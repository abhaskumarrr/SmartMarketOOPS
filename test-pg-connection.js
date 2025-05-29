const { Pool } = require('pg');
const bcrypt = require('bcryptjs');
const crypto = require('crypto');
const fs = require('fs');
const dotenv = require('dotenv');

// Load environment variables
dotenv.config();

// Log environment for debugging
console.log('Current DATABASE_URL:', process.env.DATABASE_URL);

// Parse the DATABASE_URL if available
let connectionConfig;
if (process.env.DATABASE_URL) {
  try {
    const url = new URL(process.env.DATABASE_URL);
    connectionConfig = {
      user: url.username || 'postgres',
      password: url.password || 'postgres',
      host: url.hostname || 'localhost',
      port: parseInt(url.port || '5432'),
      database: url.pathname.split('/')[1] || 'smoops',
      ssl: url.searchParams.get('sslmode') === 'require' ? true : false
    };
    console.log('Parsed connection config:', connectionConfig);
  } catch (error) {
    console.error('Error parsing DATABASE_URL:', error);
  }
} else {
  // Fallback configuration
  connectionConfig = {
    user: 'postgres',
    host: 'localhost',
    database: 'smoops',
    password: 'postgres',
    port: 5432,
  };
  console.log('Using fallback connection config:', connectionConfig);
}

// Configuration for PostgreSQL connection
const pool = new Pool(connectionConfig);

async function testDatabaseConnection() {
  let client;
  try {
    // Connect to the database
    client = await pool.connect();
    console.log('Successfully connected to PostgreSQL database');
    
    // Test a simple query
    const result = await client.query('SELECT id, name, email FROM "User" LIMIT 1');
    console.log('Query result:', result.rows);
    
    // Generate a unique email to avoid conflicts
    const uniqueId = crypto.randomBytes(4).toString('hex');
    const email = `test_${uniqueId}@example.com`;
    
    // Hash the password
    const hashedPassword = await bcrypt.hash('TestPassword123', 10);
    
    // Create a new user using direct SQL
    const insertQuery = `
      INSERT INTO "User" (
        id, 
        name, 
        email, 
        password, 
        "isVerified", 
        "createdAt", 
        "updatedAt"
      ) VALUES (
        $1, $2, $3, $4, $5, $6, $7
      ) RETURNING id, name, email, "isVerified";
    `;
    
    const insertValues = [
      crypto.randomUUID(),
      'Test User Direct SQL',
      email,
      hashedPassword,
      true,
      new Date(),
      new Date()
    ];
    
    const insertResult = await client.query(insertQuery, insertValues);
    console.log('User created successfully:', insertResult.rows[0]);
    
  } catch (error) {
    console.error('Error working with PostgreSQL:', error);
  } finally {
    // Release the client back to the pool
    if (client) client.release();
    
    // End the pool
    await pool.end();
    console.log('Connection pool ended');
  }
}

testDatabaseConnection(); 