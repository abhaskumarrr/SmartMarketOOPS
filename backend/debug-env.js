require('dotenv').config();

console.log('🔍 Environment Variables Debug:');
console.log('DELTA_EXCHANGE_API_KEY:', process.env.DELTA_EXCHANGE_API_KEY);
console.log('DELTA_EXCHANGE_API_SECRET:', process.env.DELTA_EXCHANGE_API_SECRET ? '***' + process.env.DELTA_EXCHANGE_API_SECRET.slice(-4) : 'undefined');
console.log('DELTA_EXCHANGE_TESTNET:', process.env.DELTA_EXCHANGE_TESTNET);
console.log('DELTA_EXCHANGE_BASE_URL:', process.env.DELTA_EXCHANGE_BASE_URL);

// Test if the credentials match what we expect
const expectedKey = 'AjTdJYCVE3aMZDAVQ2r6AQdmkU2mWc';
const expectedSecret = 'R29RkXJfUIIt4o3vCDXImyg6q74JvByYltVKFH96UJG51lR1mm88PCGnMrUR';

console.log('\n🔍 Credential Verification:');
console.log('API Key matches expected:', process.env.DELTA_EXCHANGE_API_KEY === expectedKey);
console.log('API Secret matches expected:', process.env.DELTA_EXCHANGE_API_SECRET === expectedSecret);

if (process.env.DELTA_EXCHANGE_API_SECRET !== expectedSecret) {
    console.log('❌ API Secret mismatch!');
    console.log('Expected:', expectedSecret);
    console.log('Actual:', process.env.DELTA_EXCHANGE_API_SECRET);
}
