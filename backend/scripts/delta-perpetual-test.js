#!/usr/bin/env node
/**
 * Delta Exchange Perpetual Test
 * Using BTCUSDT perpetual futures
 */

const axios = require('axios');
const crypto = require('crypto');
require('dotenv').config();

const API_KEY = process.env.DELTA_EXCHANGE_API_KEY;
const API_SECRET = process.env.DELTA_EXCHANGE_API_SECRET;
const BASE_URL = 'https://testnet-api.delta.exchange';

console.log('💹 DELTA EXCHANGE PERPETUAL TEST');
console.log('================================');

async function getProductInfo() {
    try {
        // Get all products and filter for BTCUSDT perpetual
        const response = await axios.get(`${BASE_URL}/v2/products`);
        
        if (response.data?.result) {
            const product = response.data.result.find(p => 
                p.symbol === 'BTCUSDT' && 
                p.product_type === 'perpetual_futures'
            );
            
            if (product) {
                console.log('\nProduct Info:', {
                    id: product.id,
                    symbol: product.symbol,
                    description: product.description,
                    contract_unit_currency: product.contract_unit_currency,
                    price_decimals: product.price_decimals,
                    quantity_decimals: product.quantity_decimals,
                    min_quantity: product.min_quantity,
                    tick_size: product.tick_size
                });
                return product;
            }
        }
        throw new Error('BTCUSDT perpetual not found');
    } catch (error) {
        console.error('Error fetching product:', error.message);
        return null;
    }
}

async function getCurrentPrice(productId) {
    try {
        // Get L1 orderbook for the most accurate price
        const response = await axios.get(`${BASE_URL}/v2/l2orderbook/${productId}`);
        
        if (response.data?.result) {
            const orderbook = response.data.result;
            if (orderbook.buy.length > 0 && orderbook.sell.length > 0) {
                const bestBid = parseFloat(orderbook.buy[0][0]);
                const bestAsk = parseFloat(orderbook.sell[0][0]);
                const midPrice = (bestBid + bestAsk) / 2;
                
                console.log('\nOrderbook Data:');
                console.log('Best Bid:', bestBid);
                console.log('Best Ask:', bestAsk);
                console.log('Mid Price:', midPrice);
                
                return {
                    bid: bestBid,
                    ask: bestAsk,
                    mid: midPrice
                };
            }
        }
        throw new Error('No orderbook data found');
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

async function placeOrder(product, side, priceData) {
    const timestamp = Math.floor(Date.now() / 1000).toString();
    const endpoint = '/v2/orders';
    
    // Use the appropriate price based on side
    const basePrice = side === 'buy' ? priceData.ask : priceData.bid;
    // Add a small spread for limit orders
    const limitPrice = side === 'buy' 
        ? (basePrice * 1.001).toFixed(product.price_decimals || 2)  // 0.1% above ask for buy
        : (basePrice * 0.999).toFixed(product.price_decimals || 2); // 0.1% below bid for sell
    
    const orderData = {
        "product_id": product.id,
        "size": product.min_quantity || 0.001,
        "side": side,
        "order_type": "limit_order",
        "limit_price": limitPrice,
        "time_in_force": "gtc"
    };

    console.log(`\nPlacing ${side} order:`, orderData);
    
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
    // Get BTCUSDT perpetual information
    const product = await getProductInfo();
    if (!product) {
        console.log('❌ Could not fetch product information. Aborting.');
        return;
    }

    // Get current market price
    const priceData = await getCurrentPrice(product.id);
    if (!priceData) {
        console.log('❌ Could not fetch valid market price. Aborting.');
        return;
    }

    if (priceData.mid < 1000 || priceData.mid > 200000) {
        console.log('❌ Price seems invalid:', priceData);
        console.log('Expected BTC price to be between 1,000 and 200,000 USDT');
        return;
    }

    console.log('\nUsing price data:', priceData);

    // Place test orders
    console.log('\n1️⃣ Placing Buy Order...');
    await placeOrder(product, 'buy', priceData);

    console.log('\n2️⃣ Placing Sell Order...');
    await placeOrder(product, 'sell', priceData);
}

main().catch(console.error);
