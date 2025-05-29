const { Pool } = require('pg');

// Configuration for PostgreSQL connection
const pool = new Pool({
  user: 'postgres',
  host: 'localhost',
  database: 'smoops',
  password: 'postgres',
  port: 5432,
});

async function testConnection() {
  try {
    // Connect to the database
    const client = await pool.connect();
    console.log('Successfully connected to PostgreSQL database');
    
    // Test a simple query
    const result = await client.query('SELECT id, name, email FROM "User" LIMIT 1');
    console.log('Query result:', result.rows);
    
    // Release the client back to the pool
    client.release();
    
    // End the pool
    await pool.end();
    console.log('Connection pool ended');
  } catch (error) {
    console.error('Error connecting to PostgreSQL:', error);
  }
}

testConnection(); 