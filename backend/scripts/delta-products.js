#!/usr/bin/env node
const axios = require('axios');

const BASE_URL = 'https://testnet-api.delta.exchange';

async function listProducts() {
    try {
        const response = await axios.get(`${BASE_URL}/v2/products`);
        if (response.data?.result) {
            console.log('\nAvailable Products:');
            response.data.result
                .filter(p => p.product_type === 'perpetual_futures')
                .forEach(p => {
                    console.log({
                        id: p.id,
                        symbol: p.symbol,
                        type: p.product_type,
                        description: p.description,
                        status: p.status
                    });
                });
        }
    } catch (error) {
        console.error('Error:', error.message);
    }
}

listProducts();
