#!/usr/bin/env node
/**
 * Delta Exchange Market Price Test
 * Using real-time prices for orders
 */

const axios = require('axios');
const crypto = require('crypto');
require('dotenv').config();

const API_KEY = process.env.DELTA_EXCHANGE_API_KEY;
const API_SECRET = process.env.DELTA_EXCHANGE_API_SECRET;
const BASE_URL = 'https://testnet-api.delta.exchange';

console.log('💹 DELTA EXCHANGE MARKET PRICE TEST');
console.log('==================================');

async function getMarkPrice(symbol) {
    try {
        const response = await axios.get(`${BASE_URL}/v2/tickers`, {
            params: { symbols: symbol }
        });
        
        if (response.data && response.data.result && response.data.result.length > 0) {
            const ticker = response.data.result[0];
            console.log('\nMarket Data for', symbol);
            console.log('Mark Price:', ticker.mark_price);
            console.log('Spot Price:', ticker.spot_price);
            console.log('Last Price:', ticker.last_price);
            return {
                markPrice: parseFloat(ticker.mark_price),
                spotPrice: parseFloat(ticker.spot_price),
                lastPrice: parseFloat(ticker.last_price)
            };
        }
        throw new Error('No price data found');
    } catch (error) {
        console.error('Error fetching price:', error.message);
        return null;
    }
}

function generateSignature(timestamp, method, endpoint, body = '') {
    const signaturePayload = `${timestamp}${method}${endpoint}${body}`;
    return crypto
        .createHmac('sha256', API_SECRET)
        .update(signaturePayload)
        .digest('hex');
}

async function placeOrder(productId, side, price) {
    const timestamp = Math.floor(Date.now() / 1000).toString();
    const endpoint = '/v2/orders';
    
    // Add a small spread for limit orders
    const limitPrice = side === 'buy' 
        ? (price * 1.001).toFixed(2)  // 0.1% above market for buy
        : (price * 0.999).toFixed(2); // 0.1% below market for sell
    
    const orderData = {
        "product_id": productId,
        "size": 0.001, // Minimum size for BTC
        "side": side,
        "order_type": "limit_order",
        "limit_price": limitPrice,
        "time_in_force": "gtc"
    };

    const body = JSON.stringify(orderData);
    const signature = generateSignature(timestamp, 'POST', endpoint, body);

    try {
        const response = await axios({
            method: 'post',
            url: `${BASE_URL}${endpoint}`,
            headers: {
                'api-key': API_KEY,
                'timestamp': timestamp,
                'signature': signature,
                'Content-Type': 'application/json',
                'User-Agent': 'delta-exchange-client/1.0.0',
                'Accept': 'application/json'
            },
            data: orderData
        });

        console.log(`\n✅ ${side.toUpperCase()} Order Placed!`);
        console.log('Order Details:', response.data);
        return response.data;
    } catch (error) {
        console.log('\n❌ Order Error:', error.response?.data || error.message);
        return null;
    }
}

async function main() {
    // Test with BTCUSDT
    const symbol = 'BTCUSDT';
    const productId = 84; // BTCUSDT product ID for testnet
    
    // Get current market price
    const priceData = await getMarkPrice(symbol);
    if (!priceData) {
        console.log('❌ Could not fetch market price. Aborting.');
        return;
    }

    // Use mark price for orders
    const currentPrice = priceData.markPrice;
    console.log('\nUsing current price:', currentPrice);

    // Place test orders
    console.log('\n1️⃣ Placing Buy Order...');
    await placeOrder(productId, 'buy', currentPrice);

    console.log('\n2️⃣ Placing Sell Order...');
    await placeOrder(productId, 'sell', currentPrice);
}

main().catch(console.error);
