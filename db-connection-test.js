const { Pool } = require('pg');

// Create a connection pool
const pool = new Pool({
  user: 'abhaskumarrr',
  host: 'localhost',
  database: 'smoops',
  password: '',
  port: 5432,
});

async function testConnection() {
  let client;
  try {
    // Connect to the database
    client = await pool.connect();
    console.log('✅ Successfully connected to PostgreSQL database');
    
    // Test a simple query
    const res = await client.query('SELECT current_database() as db, current_user as user');
    console.log('Database:', res.rows[0].db);
    console.log('User:', res.rows[0].user);
    
    // List schemas
    const schemas = await client.query("SELECT schema_name FROM information_schema.schemata");
    console.log('Available schemas:', schemas.rows.map(row => row.schema_name).join(', '));

    // Check if tables exist
    const tables = await client.query("SELECT table_name FROM information_schema.tables WHERE table_schema='public'");
    console.log('Tables in public schema:', tables.rows.map(row => row.table_name).join(', ') || 'No tables found');
  } catch (error) {
    console.error('❌ Error connecting to PostgreSQL:', error);
  } finally {
    // Close the connection
    if (client) {
      client.release();
      console.log('Connection released');
    }
    await pool.end();
    console.log('Pool ended');
  }
}

// Run the test
testConnection(); 