const { Pool } = require('pg');
const bcrypt = require('bcryptjs');
const crypto = require('crypto');

// Direct PostgreSQL connection
const pool = new Pool({
  user: 'postgres',
  host: 'localhost',
  database: 'smoops',
  password: 'postgres',
  port: 5432,
});

async function createTestUser() {
  let client;
  try {
    // Connect to the database
    client = await pool.connect();
    console.log('Successfully connected to PostgreSQL database');
    
    // Hash the password
    const hashedPassword = await bcrypt.hash('TestPassword123', 10);
    
    // Generate a unique email to avoid conflicts
    const uniqueId = crypto.randomBytes(4).toString('hex');
    const email = `test_${uniqueId}@example.com`;
    
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
    console.log('Test user created successfully:', insertResult.rows[0]);
  } catch (error) {
    console.error('Error creating test user:', error);
  } finally {
    if (client) client.release();
    await pool.end();
    console.log('Connection pool ended');
  }
}

createTestUser(); 