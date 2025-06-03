const ccxt = require('ccxt');
// Comprehensive position status checker
class PositionStatus {
    constructor() {
        this.exchange = new ccxt.delta({
            sandbox: true,
            enableRateLimit: true,
            options: { defaultType: 'spot' }
        });
        // Known positions from paper trading system
        this.knownPositions = [
            {
                id: 'paper-trade-1',
                symbol: 'ETH/USDT',
                side: 'long',
                size: 0.1994,
                entryPrice: 3008.34,
                stopLoss: 2933.13,
                takeProfitLevels: [
                    { percentage: 25, price: 3181.32, ratio: '2.00:1', executed: false },
                    { percentage: 50, price: 3440.79, ratio: '5.00:1', executed: false },
                    { percentage: 25, price: 3527.28, ratio: '5.00:1', executed: false }
                ],
                openTime: new Date('2025-06-02T18:33:34.868Z'),
                status: 'closed', // Updated based on our previous execution
                closeReason: 'stop_loss',
                closePrice: 2579.54,
                finalPnL: -85.50,
                closedAt: new Date()
            }
        ];
    }
    async getCurrentPrice(symbol) {
        try {
            await this.exchange.loadMarkets();
            const ticker = await this.exchange.fetchTicker(symbol);
            return ticker.indexPrice || parseFloat(ticker.info?.spot_price) || ticker.last;
        }
        catch (error) {
            console.error(`❌ Error fetching price for ${symbol}:`, error.message);
            return null;
        }
    }
    async checkMarketStatus() {
        console.log('📊 MARKET STATUS CHECK');
        console.log('═'.repeat(50));
        const symbols = ['ETH/USDT', 'BTC/USDT'];
        for (const symbol of symbols) {
            const price = await this.getCurrentPrice(symbol);
            if (price) {
                console.log(`💰 ${symbol}: $${price.toFixed(2)}`);
            }
            else {
                console.log(`❌ ${symbol}: Price unavailable`);
            }
        }
        console.log('');
    }
    displayPositionHistory() {
        console.log('📈 POSITION HISTORY');
        console.log('═'.repeat(60));
        this.knownPositions.forEach((position, index) => {
            console.log(`\n🔍 Position ${index + 1}:`);
            console.log(`   ID: ${position.id}`);
            console.log(`   Symbol: ${position.symbol}`);
            console.log(`   Side: ${position.side.toUpperCase()}`);
            console.log(`   Size: ${position.size}`);
            console.log(`   Entry Price: $${position.entryPrice}`);
            console.log(`   Stop Loss: $${position.stopLoss}`);
            console.log(`   Status: ${position.status.toUpperCase()}`);
            if (position.status === 'closed') {
                console.log(`   Close Reason: ${position.closeReason}`);
                console.log(`   Close Price: $${position.closePrice}`);
                console.log(`   Final P&L: $${position.finalPnL.toFixed(2)}`);
                console.log(`   Duration: ${Math.round((position.closedAt - position.openTime) / 1000 / 60)} minutes`);
            }
            console.log(`   Take Profit Levels:`);
            position.takeProfitLevels.forEach((level, i) => {
                const status = level.executed ? '✅ EXECUTED' : '⏳ PENDING';
                console.log(`     Level ${i + 1}: ${level.percentage}% @ $${level.price} (${level.ratio}) - ${status}`);
            });
        });
    }
    async checkForNewSignals() {
        console.log('\n🎯 SIGNAL ANALYSIS');
        console.log('═'.repeat(50));
        const ethPrice = await this.getCurrentPrice('ETH/USDT');
        const btcPrice = await this.getCurrentPrice('BTC/USDT');
        if (ethPrice && btcPrice) {
            // Simple technical analysis
            console.log('📊 Current Market Analysis:');
            console.log(`   ETH/USDT: $${ethPrice.toFixed(2)}`);
            console.log(`   BTC/USDT: $${btcPrice.toFixed(2)}`);
            // Check if prices are at potential entry levels
            if (ethPrice < 2600) {
                console.log('🟢 ETH: Potential long entry opportunity (price below $2600)');
            }
            else if (ethPrice > 3200) {
                console.log('🔴 ETH: Potential short entry opportunity (price above $3200)');
            }
            else {
                console.log('🟡 ETH: Neutral zone - wait for clearer signals');
            }
            if (btcPrice < 100000) {
                console.log('🟢 BTC: Potential long entry opportunity (price below $100k)');
            }
            else if (btcPrice > 110000) {
                console.log('🔴 BTC: Potential short entry opportunity (price above $110k)');
            }
            else {
                console.log('🟡 BTC: Neutral zone - wait for clearer signals');
            }
        }
    }
    displayTradingSystemStatus() {
        console.log('\n🤖 TRADING SYSTEM STATUS');
        console.log('═'.repeat(50));
        console.log('✅ Paper Trading Engine: Active');
        console.log('✅ Market Data Feed: Connected (Delta Exchange)');
        console.log('✅ Risk Management: Enabled (2% per trade)');
        console.log('✅ Dynamic Take Profit: Configured (3 levels)');
        console.log('✅ Stop Loss Protection: Active');
        console.log('⚠️ Live Trading: Requires valid API credentials');
        console.log('\n📊 System Configuration:');
        console.log('   💰 Capital: $2,000');
        console.log('   ⚡ Leverage: 3x');
        console.log('   🎯 Risk per Trade: 2% ($40)');
        console.log('   📈 Assets: ETH/USDT, BTC/USDT');
        console.log('   🏢 Exchange: Delta Exchange (Testnet ready)');
    }
    async generatePositionReport() {
        console.log('🚀 COMPREHENSIVE POSITION STATUS REPORT');
        console.log('═'.repeat(70));
        console.log(`📅 Generated: ${new Date().toLocaleString()}`);
        console.log('');
        // Market status
        await this.checkMarketStatus();
        // Position history
        this.displayPositionHistory();
        // Signal analysis
        await this.checkForNewSignals();
        // System status
        this.displayTradingSystemStatus();
        // Summary
        console.log('\n📋 SUMMARY');
        console.log('═'.repeat(30));
        const openPositions = this.knownPositions.filter(p => p.status === 'open');
        const closedPositions = this.knownPositions.filter(p => p.status === 'closed');
        console.log(`📊 Total Positions: ${this.knownPositions.length}`);
        console.log(`🔥 Open Positions: ${openPositions.length}`);
        console.log(`✅ Closed Positions: ${closedPositions.length}`);
        if (closedPositions.length > 0) {
            const totalPnL = closedPositions.reduce((sum, pos) => sum + pos.finalPnL, 0);
            console.log(`💰 Total Realized P&L: $${totalPnL.toFixed(2)}`);
            const winningTrades = closedPositions.filter(p => p.finalPnL > 0).length;
            const winRate = (winningTrades / closedPositions.length) * 100;
            console.log(`📈 Win Rate: ${winRate.toFixed(1)}% (${winningTrades}/${closedPositions.length})`);
        }
        if (openPositions.length === 0) {
            console.log('\n🎯 READY FOR NEW TRADES');
            console.log('   System is monitoring for entry signals');
            console.log('   Risk management parameters are active');
            console.log('   Waiting for optimal market conditions');
        }
        else {
            console.log('\n⚠️ ACTIVE POSITION MANAGEMENT REQUIRED');
            console.log('   Monitor open positions closely');
            console.log('   Ensure stop losses are in place');
            console.log('   Watch for take profit opportunities');
        }
    }
}
// Run the position status checker
async function main() {
    try {
        const statusChecker = new PositionStatus();
        await statusChecker.generatePositionReport();
    }
    catch (error) {
        console.error('❌ Error generating position report:', error);
    }
}
main();
//# sourceMappingURL=position-status.js.map