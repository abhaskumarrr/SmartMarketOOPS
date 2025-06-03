const ccxt = require('ccxt');
// Simple trade management script for Delta Exchange
class TradeManager {
    constructor() {
        this.exchange = new ccxt.delta({
            sandbox: true,
            enableRateLimit: true,
            options: { defaultType: 'spot' }
        });
        // Open trade from the paper trading system
        this.openTrade = {
            symbol: 'ETH/USDT',
            side: 'BUY',
            amount: 0.1994,
            entryPrice: 3008.34,
            stopLoss: 2933.13,
            takeProfitLevels: [
                { percentage: 25, price: 3181.32, ratio: '2.00:1' },
                { percentage: 50, price: 3440.79, ratio: '5.00:1' },
                { percentage: 25, price: 3527.28, ratio: '5.00:1' }
            ],
            openTime: new Date('2025-06-02T18:33:34.868Z')
        };
    }
    async getCurrentPrice(symbol) {
        try {
            await this.exchange.loadMarkets();
            const ticker = await this.exchange.fetchTicker(symbol);
            // Delta Exchange provides price in indexPrice or spot_price
            return ticker.indexPrice || parseFloat(ticker.info?.spot_price) || ticker.last;
        }
        catch (error) {
            console.error(`❌ Error fetching price for ${symbol}:`, error.message);
            return null;
        }
    }
    async manageOpenTrade() {
        console.log('🔄 MANAGING OPEN TRADE');
        console.log('=====================================');
        const currentPrice = await this.getCurrentPrice(this.openTrade.symbol);
        if (!currentPrice) {
            console.log('❌ Cannot get current price, skipping trade management');
            return;
        }
        console.log(`📊 Current ${this.openTrade.symbol} Price: $${currentPrice.toFixed(2)}`);
        console.log(`📈 Entry Price: $${this.openTrade.entryPrice}`);
        console.log(`🛑 Stop Loss: $${this.openTrade.stopLoss}`);
        const pnl = (currentPrice - this.openTrade.entryPrice) * this.openTrade.amount;
        const pnlPercent = ((currentPrice - this.openTrade.entryPrice) / this.openTrade.entryPrice) * 100;
        console.log(`💰 Current P&L: $${pnl.toFixed(2)} (${pnlPercent.toFixed(2)}%)`);
        // Check stop loss
        if (currentPrice <= this.openTrade.stopLoss) {
            console.log('🚨 STOP LOSS TRIGGERED!');
            console.log(`   Closing position at $${currentPrice.toFixed(2)}`);
            console.log(`   Loss: $${pnl.toFixed(2)} (${pnlPercent.toFixed(2)}%)`);
            return 'STOP_LOSS_HIT';
        }
        // Check take profit levels
        let action = 'HOLD';
        for (let i = 0; i < this.openTrade.takeProfitLevels.length; i++) {
            const level = this.openTrade.takeProfitLevels[i];
            if (currentPrice >= level.price && !level.executed) {
                console.log(`🎯 TAKE PROFIT LEVEL ${i + 1} HIT!`);
                console.log(`   Price: $${level.price} (${level.ratio})`);
                console.log(`   Closing ${level.percentage}% of position`);
                const partialAmount = (this.openTrade.amount * level.percentage) / 100;
                const partialPnl = (level.price - this.openTrade.entryPrice) * partialAmount;
                console.log(`   Partial close: ${partialAmount.toFixed(4)} ${this.openTrade.symbol}`);
                console.log(`   Partial profit: $${partialPnl.toFixed(2)}`);
                level.executed = true;
                level.executedAt = new Date();
                action = 'PARTIAL_CLOSE';
            }
        }
        // Check if all levels executed
        const allExecuted = this.openTrade.takeProfitLevels.every(level => level.executed);
        if (allExecuted) {
            console.log('✅ ALL TAKE PROFIT LEVELS EXECUTED - POSITION CLOSED');
            action = 'FULLY_CLOSED';
        }
        return action;
    }
    async monitorTrade() {
        console.log('🚀 STARTING TRADE MONITORING');
        console.log(`📊 Monitoring ${this.openTrade.symbol} position`);
        console.log(`💼 Position: ${this.openTrade.side} ${this.openTrade.amount} @ $${this.openTrade.entryPrice}`);
        console.log('');
        let monitoring = true;
        let iteration = 1;
        while (monitoring) {
            console.log(`\n🔄 Monitoring Iteration ${iteration}`);
            console.log('─'.repeat(40));
            const action = await this.manageOpenTrade();
            if (action === 'STOP_LOSS_HIT' || action === 'FULLY_CLOSED') {
                monitoring = false;
                console.log('\n🏁 Trade monitoring completed');
            }
            if (monitoring) {
                console.log('⏳ Waiting 10 seconds for next check...');
                await new Promise(resolve => setTimeout(resolve, 10000));
            }
            iteration++;
            // Safety limit
            if (iteration > 100) {
                console.log('⚠️ Reached maximum monitoring iterations, stopping');
                monitoring = false;
            }
        }
    }
}
// Run the trade manager
async function main() {
    try {
        const manager = new TradeManager();
        await manager.monitorTrade();
    }
    catch (error) {
        console.error('❌ Error in trade management:', error);
    }
}
main();
//# sourceMappingURL=manage-open-trades.js.map