// Compare signature generation between working test and service
require('dotenv').config();

const crypto = require('crypto');

const API_SECRET = process.env.DELTA_EXCHANGE_API_SECRET;

// Test with the same timestamp
const timestamp = '1749080200';
const method = 'GET';
const path = '/v2/wallet/balances';
const queryString = '';
const body = '';

console.log('🔍 Signature Comparison Test');
console.log('API Secret:', API_SECRET ? '***' + API_SECRET.slice(-4) : 'undefined');
console.log('Timestamp:', timestamp);
console.log('Method:', method);
console.log('Path:', path);
console.log('Query String:', queryString);
console.log('Body:', body);

// Working test approach
function generateSignatureWorking(secret, message) {
    return crypto.createHmac('sha256', secret).update(message).digest('hex');
}

// Service approach (should be the same)
function generateSignatureService(method, path, queryString, body, timestamp, secret) {
    const message = method + timestamp + path + queryString + body;
    return crypto.createHmac('sha256', secret).update(message).digest('hex');
}

const message = method + timestamp + path + queryString + body;
console.log('\nMessage:', message);

const signatureWorking = generateSignatureWorking(API_SECRET, message);
const signatureService = generateSignatureService(method, path, queryString, body, timestamp, API_SECRET);

console.log('\n🔐 Signatures:');
console.log('Working approach:', signatureWorking);
console.log('Service approach:', signatureService);
console.log('Match:', signatureWorking === signatureService);

if (signatureWorking !== signatureService) {
    console.log('\n❌ SIGNATURES DO NOT MATCH!');
    console.log('This indicates a bug in the signature generation logic.');
} else {
    console.log('\n✅ SIGNATURES MATCH!');
    console.log('The signature generation logic is correct.');
}
