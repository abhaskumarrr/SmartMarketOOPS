const { Client } = require('pg');

async function testConnection() {
  const client = new Client({
    host: 'localhost',
    port: 5432,
    database: 'postgres',
    user: 'smoops_user',
    password: 'smoops_password',
  });

  console.log('Attempting to connect with:', {
    host: 'localhost',
    port: 5432,
    database: 'postgres',
    user: 'smoops_user'
  });

  try {
    await client.connect();
    console.log('✅ Database connection successful');
    
    const result = await client.query('SELECT version()');
    console.log('✅ Database version:', result.rows[0].version);
    
    // Test if we can create a table
    await client.query(`
      CREATE TABLE IF NOT EXISTS test_table (
        id SERIAL PRIMARY KEY,
        name VARCHAR(100)
      )
    `);
    console.log('✅ Table creation successful');
    
    // Clean up
    await client.query('DROP TABLE IF EXISTS test_table');
    console.log('✅ Table cleanup successful');
    
  } catch (error) {
    console.error('❌ Database connection failed:', error.message);
  } finally {
    await client.end();
  }
}

testConnection();
