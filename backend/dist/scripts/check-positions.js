"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const DeltaExchangeUnified_1 = require("../services/DeltaExchangeUnified");
const logger_1 = require("../utils/logger");
const dotenv_1 = __importDefault(require("dotenv"));
// Load environment variables
dotenv_1.default.config();
async function checkPositions() {
    try {
        logger_1.logger.info('🔍 Checking current Delta Exchange positions...');
        const config = {
            apiKey: process.env.DELTA_API_KEY,
            apiSecret: process.env.DELTA_API_SECRET,
            testnet: true
        };
        const deltaService = new DeltaExchangeUnified_1.DeltaExchangeUnified(config);
        await deltaService.initialize();
        // Get positions
        const positions = await deltaService.getPositions();
        logger_1.logger.info(`📊 Found ${positions.length} positions:`);
        for (const position of positions) {
            if (Math.abs(position.size) > 0) {
                logger_1.logger.info(`\n🔥 ACTIVE POSITION:`);
                logger_1.logger.info(`   Symbol: ${position.product?.symbol || 'Unknown'}`);
                logger_1.logger.info(`   Size: ${position.size}`);
                logger_1.logger.info(`   Entry Price: $${position.entry_price}`);
                logger_1.logger.info(`   Mark Price: $${position.mark_price || 'N/A'}`);
                logger_1.logger.info(`   Unrealized PnL: $${position.unrealized_pnl || 'N/A'}`);
                logger_1.logger.info(`   Margin: $${position.margin || 'N/A'}`);
                logger_1.logger.info(`   Liquidation Price: $${position.liquidation_price || 'N/A'}`);
            }
        }
        if (positions.length === 0 || positions.every(p => Math.abs(p.size) === 0)) {
            logger_1.logger.info('📭 No active positions found');
        }
    }
    catch (error) {
        logger_1.logger.error('❌ Error checking positions:', error);
    }
}
checkPositions();
//# sourceMappingURL=check-positions.js.map