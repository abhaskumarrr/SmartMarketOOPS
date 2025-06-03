const ccxt = require('ccxt');
// Position checker for Delta Exchange
class PositionChecker {
    constructor() {
        // Try both testnet and production configurations
        this.exchanges = {
            testnet: new ccxt.delta({
                apiKey: 'YsA1TIH5EXk8fl0AYkDtV464ErNa4T',
                secret: 'kKBR52xNlKEGLQXOEAOnJlCUip60g4vblyI0BAi5h5scaIWfVom2KQ9RCMat',
                sandbox: true,
                enableRateLimit: true,
                options: { defaultType: 'spot' }
            }),
            production: new ccxt.delta({
                apiKey: 'YsA1TIH5EXk8fl0AYkDtV464ErNa4T',
                secret: 'kKBR52xNlKEGLQXOEAOnJlCUip60g4vblyI0BAi5h5scaIWfVom2KQ9RCMat',
                sandbox: false,
                enableRateLimit: true,
                options: { defaultType: 'spot' }
            })
        };
    }
    async checkConnection(exchangeName, exchange) {
        try {
            console.log(`🔄 Testing ${exchangeName} connection...`);
            await exchange.loadMarkets();
            console.log(`✅ ${exchangeName}: Markets loaded successfully`);
            // Test balance fetch
            const balance = await exchange.fetchBalance();
            console.log(`✅ ${exchangeName}: Balance fetched successfully`);
            return { success: true, balance };
        }
        catch (error) {
            console.log(`❌ ${exchangeName}: ${error.message}`);
            return { success: false, error: error.message };
        }
    }
    async checkPositions(exchangeName, exchange) {
        try {
            console.log(`📊 Checking positions on ${exchangeName}...`);
            const positions = await exchange.fetchPositions();
            const openPositions = positions.filter(pos => Math.abs(pos.size || 0) > 0 || Math.abs(pos.contracts || 0) > 0);
            console.log(`📊 ${exchangeName}: Found ${positions.length} total positions, ${openPositions.length} open`);
            if (openPositions.length > 0) {
                console.log(`\n🔥 OPEN POSITIONS ON ${exchangeName.toUpperCase()}:`);
                console.log('═'.repeat(60));
                openPositions.forEach((pos, index) => {
                    console.log(`\n📈 Position ${index + 1}:`);
                    console.log(`   Symbol: ${pos.symbol}`);
                    console.log(`   Side: ${pos.side || 'N/A'}`);
                    console.log(`   Size: ${pos.size || pos.contracts || 0}`);
                    console.log(`   Entry Price: $${pos.entryPrice || pos.averagePrice || 'N/A'}`);
                    console.log(`   Mark Price: $${pos.markPrice || 'N/A'}`);
                    console.log(`   Unrealized PnL: $${pos.unrealizedPnl || 'N/A'}`);
                    console.log(`   Percentage: ${pos.percentage || 'N/A'}%`);
                    if (pos.liquidationPrice) {
                        console.log(`   Liquidation Price: $${pos.liquidationPrice}`);
                    }
                });
            }
            else {
                console.log(`📭 No open positions found on ${exchangeName}`);
            }
            return openPositions;
        }
        catch (error) {
            console.log(`❌ Error checking positions on ${exchangeName}: ${error.message}`);
            return [];
        }
    }
    async checkOrders(exchangeName, exchange) {
        try {
            console.log(`📋 Checking open orders on ${exchangeName}...`);
            const orders = await exchange.fetchOpenOrders();
            console.log(`📋 ${exchangeName}: Found ${orders.length} open orders`);
            if (orders.length > 0) {
                console.log(`\n📋 OPEN ORDERS ON ${exchangeName.toUpperCase()}:`);
                console.log('═'.repeat(60));
                orders.forEach((order, index) => {
                    console.log(`\n📝 Order ${index + 1}:`);
                    console.log(`   ID: ${order.id}`);
                    console.log(`   Symbol: ${order.symbol}`);
                    console.log(`   Side: ${order.side}`);
                    console.log(`   Type: ${order.type}`);
                    console.log(`   Amount: ${order.amount}`);
                    console.log(`   Price: $${order.price || 'Market'}`);
                    console.log(`   Status: ${order.status}`);
                    console.log(`   Filled: ${order.filled || 0}/${order.amount}`);
                    if (order.stopPrice) {
                        console.log(`   Stop Price: $${order.stopPrice}`);
                    }
                });
            }
            else {
                console.log(`📭 No open orders found on ${exchangeName}`);
            }
            return orders;
        }
        catch (error) {
            console.log(`❌ Error checking orders on ${exchangeName}: ${error.message}`);
            return [];
        }
    }
    async checkBalance(exchangeName, exchange) {
        try {
            const balance = await exchange.fetchBalance();
            console.log(`\n💰 BALANCE ON ${exchangeName.toUpperCase()}:`);
            console.log('═'.repeat(50));
            const currencies = Object.keys(balance.total).filter(currency => balance.total[currency] > 0);
            if (currencies.length > 0) {
                currencies.forEach(currency => {
                    const total = balance.total[currency];
                    const free = balance.free[currency] || 0;
                    const used = balance.used[currency] || 0;
                    console.log(`   ${currency}:`);
                    console.log(`     Total: ${total}`);
                    console.log(`     Free: ${free}`);
                    console.log(`     Used: ${used}`);
                });
            }
            else {
                console.log(`   No balances found`);
            }
            return balance;
        }
        catch (error) {
            console.log(`❌ Error checking balance on ${exchangeName}: ${error.message}`);
            return null;
        }
    }
    async checkAllPositions() {
        console.log('🚀 CHECKING DELTA EXCHANGE POSITIONS');
        console.log('═'.repeat(70));
        let foundOpenPositions = false;
        let foundOpenOrders = false;
        for (const [exchangeName, exchange] of Object.entries(this.exchanges)) {
            console.log(`\n🔍 Checking ${exchangeName.toUpperCase()} environment...`);
            console.log('─'.repeat(50));
            const connectionResult = await this.checkConnection(exchangeName, exchange);
            if (connectionResult.success) {
                // Check positions
                const positions = await this.checkPositions(exchangeName, exchange);
                if (positions.length > 0) {
                    foundOpenPositions = true;
                }
                // Check orders
                const orders = await this.checkOrders(exchangeName, exchange);
                if (orders.length > 0) {
                    foundOpenOrders = true;
                }
                // Check balance
                await this.checkBalance(exchangeName, exchange);
            }
            else {
                console.log(`⚠️ Cannot access ${exchangeName} - API credentials may be invalid`);
            }
            console.log(''); // Add spacing
        }
        // Summary
        console.log('\n📊 SUMMARY');
        console.log('═'.repeat(30));
        if (foundOpenPositions) {
            console.log('🔥 OPEN POSITIONS FOUND - Trading bot should manage these');
        }
        else {
            console.log('📭 No open positions found');
        }
        if (foundOpenOrders) {
            console.log('📋 OPEN ORDERS FOUND - Monitor for execution');
        }
        else {
            console.log('📭 No open orders found');
        }
        if (!foundOpenPositions && !foundOpenOrders) {
            console.log('✅ No active trades - Ready for new positions');
        }
    }
}
// Run the position checker
async function main() {
    try {
        const checker = new PositionChecker();
        await checker.checkAllPositions();
    }
    catch (error) {
        console.error('❌ Fatal error:', error);
    }
}
main();
//# sourceMappingURL=check-positions.js.map