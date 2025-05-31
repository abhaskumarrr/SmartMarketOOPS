const { onTradeExecuted, analyticsEmitter } = require('../services/analytics');
function setupAnalyticsSocket(io) {
    io.on('connection', (socket) => {
        socket.on('trade_executed', async (trade) => {
            await onTradeExecuted(trade);
        });
        analyticsEmitter.on('metric', (metric) => {
            socket.emit('analytics_update', metric);
        });
    });
}
module.exports = { setupAnalyticsSocket };
//# sourceMappingURL=analytics.js.map