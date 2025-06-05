import { DeltaExchangeUnified } from '../services/DeltaExchangeUnified';
import { logger } from '../utils/logger';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

async function checkPositions() {
  try {
    logger.info('🔍 Checking current Delta Exchange positions...');

    const config = {
      apiKey: process.env.DELTA_API_KEY!,
      apiSecret: process.env.DELTA_API_SECRET!,
      testnet: true
    };

    const deltaService = new DeltaExchangeUnified(config);
    await deltaService.initialize();
    
    // Get positions
    const positions = await deltaService.getPositions();
    
    logger.info(`📊 Found ${positions.length} positions:`);
    
    for (const position of positions) {
      if (Math.abs(position.size) > 0) {
        logger.info(`\n🔥 ACTIVE POSITION:`);
        logger.info(`   Symbol: ${position.product?.symbol || 'Unknown'}`);
        logger.info(`   Size: ${position.size}`);
        logger.info(`   Entry Price: $${position.entry_price}`);
        logger.info(`   Mark Price: $${position.mark_price || 'N/A'}`);
        logger.info(`   Unrealized PnL: $${position.unrealized_pnl || 'N/A'}`);
        logger.info(`   Margin: $${position.margin || 'N/A'}`);
        logger.info(`   Liquidation Price: $${position.liquidation_price || 'N/A'}`);
      }
    }
    
    if (positions.length === 0 || positions.every(p => Math.abs(p.size) === 0)) {
      logger.info('📭 No active positions found');
    }
    
  } catch (error) {
    logger.error('❌ Error checking positions:', error);
  }
}

checkPositions();
