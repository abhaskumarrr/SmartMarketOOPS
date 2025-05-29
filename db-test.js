const { Client } = require('pg');

async function testDatabaseConnection() {
  // Create a new client - just test with various configurations
  const configs = [
    {
      user: 'abhaskumarrr',
      host: 'localhost',
      database: 'smoops',
      password: '',
      port: 5432,
    },
    {
      user: 'postgres',
      host: 'localhost',
      database: 'smoops',
      password: 'postgres',
      port: 5432,
    },
    {
      user: 'abhaskumarrr',
      host: 'localhost',
      database: 'postgres', // Try the default database
      password: '',
      port: 5432,
    }
  ];

  // Try each configuration
  for (const config of configs) {
    const client = new Client(config);
    
    try {
      console.log(`Trying to connect with config:`, config);
      await client.connect();
      console.log('CONNECTION SUCCESSFUL with config:', config);
      
      // Test a simple query
      const result = await client.query('SELECT current_database() as db, current_user as user');
      console.log('Database:', result.rows[0].db);
      console.log('User:', result.rows[0].user);
      
      // List schemas
      const schemas = await client.query("SELECT schema_name FROM information_schema.schemata");
      console.log('Available schemas:', schemas.rows.map(row => row.schema_name).join(', '));
      
      // List databases
      const databases = await client.query("SELECT datname FROM pg_database WHERE datistemplate = false");
      console.log('Available databases:', databases.rows.map(row => row.datname).join(', '));
      
      await client.end();
      return; // Exit after first successful connection
    } catch (error) {
      console.error(`Connection failed with config:`, config);
      console.error('Error:', error.message);
      
      try {
        await client.end();
      } catch (e) {
        // Ignore disconnection errors
      }
      
      console.log('-------------------');
    }
  }
  
  console.error('All connection attempts failed.');
}

testDatabaseConnection(); 