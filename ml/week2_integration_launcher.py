#!/usr/bin/env python3
"""
Week 2 Integration Launcher for Enhanced SmartMarketOOPS System
Launches all Week 2 components: Real market data, Multi-symbol trading, 
Advanced risk management, Live validation, and Automated retraining
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Dict, Any
import json

from src.data.real_market_data_service import get_market_data_service
from src.trading.multi_symbol_manager import MultiSymbolTradingManager
from src.risk.advanced_risk_manager import AdvancedRiskManager
from src.validation.live_performance_validator import LivePerformanceValidator
from src.training.automated_retraining_pipeline import AutomatedRetrainingPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('week2_integration.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class Week2IntegrationManager:
    """Manager for all Week 2 enhanced components"""
    
    def __init__(self):
        """Initialize the Week 2 integration manager"""
        self.components = {}
        self.is_running = False
        self.start_time = None
        
        # Component status tracking
        self.component_status = {
            'market_data_service': 'stopped',
            'multi_symbol_manager': 'stopped',
            'risk_manager': 'stopped',
            'performance_validator': 'stopped',
            'retraining_pipeline': 'stopped'
        }
        
        logger.info("Week 2 Integration Manager initialized")
    
    async def initialize_components(self):
        """Initialize all Week 2 components"""
        logger.info("🚀 Initializing Week 2 Enhanced Components...")
        
        try:
            # 1. Real Market Data Service
            logger.info("📊 Initializing Real Market Data Service...")
            self.components['market_data_service'] = await get_market_data_service()
            self.component_status['market_data_service'] = 'running'
            logger.info("✅ Real Market Data Service initialized")
            
            # 2. Multi-Symbol Trading Manager
            logger.info("🎯 Initializing Multi-Symbol Trading Manager...")
            self.components['multi_symbol_manager'] = MultiSymbolTradingManager()
            await self.components['multi_symbol_manager'].initialize()
            self.component_status['multi_symbol_manager'] = 'running'
            logger.info("✅ Multi-Symbol Trading Manager initialized")
            
            # 3. Advanced Risk Manager
            logger.info("🛡️  Initializing Advanced Risk Manager...")
            self.components['risk_manager'] = AdvancedRiskManager(portfolio_value=100000.0)
            self.component_status['risk_manager'] = 'running'
            logger.info("✅ Advanced Risk Manager initialized")
            
            # 4. Live Performance Validator
            logger.info("📈 Initializing Live Performance Validator...")
            self.components['performance_validator'] = LivePerformanceValidator()
            self.component_status['performance_validator'] = 'running'
            logger.info("✅ Live Performance Validator initialized")
            
            # 5. Automated Retraining Pipeline
            logger.info("🔄 Initializing Automated Retraining Pipeline...")
            self.components['retraining_pipeline'] = AutomatedRetrainingPipeline()
            await self.components['retraining_pipeline'].start_pipeline()
            self.component_status['retraining_pipeline'] = 'running'
            logger.info("✅ Automated Retraining Pipeline initialized")
            
            logger.info("🎉 All Week 2 components initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error initializing components: {e}")
            raise
    
    async def start_integrated_system(self):
        """Start the integrated Week 2 system"""
        logger.info("🚀 Starting Week 2 Enhanced SmartMarketOOPS System")
        logger.info("="*80)
        
        self.start_time = datetime.now()
        self.is_running = True
        
        # Initialize all components
        await self.initialize_components()
        
        # Start main trading loop
        await self._run_main_trading_loop()
    
    async def _run_main_trading_loop(self):
        """Run the main integrated trading loop"""
        logger.info("🔄 Starting main trading loop...")
        
        cycle_count = 0
        
        try:
            while self.is_running:
                cycle_count += 1
                cycle_start = datetime.now()
                
                logger.info(f"\n--- Trading Cycle {cycle_count} ---")
                
                # 1. Generate multi-symbol signals
                signals = await self.components['multi_symbol_manager'].generate_multi_symbol_signals()
                logger.info(f"📊 Generated {len(signals)} signals")
                
                # 2. Apply advanced risk management
                if signals:
                    risk_approved_signals = await self._apply_risk_management(signals)
                    logger.info(f"🛡️  Risk approved {len(risk_approved_signals)}/{len(signals)} signals")
                else:
                    risk_approved_signals = {}
                
                # 3. Execute trades
                if risk_approved_signals:
                    executed_trades = await self.components['multi_symbol_manager'].execute_multi_symbol_trades(risk_approved_signals)
                    logger.info(f"🚀 Executed {len(executed_trades)} trades")
                
                # 4. Monitor existing positions
                closed_positions = await self.components['multi_symbol_manager'].monitor_positions()
                if closed_positions:
                    logger.info(f"🔄 Closed {len(closed_positions)} positions")
                
                # 5. Update performance metrics
                await self._update_performance_metrics(closed_positions)
                
                # 6. Generate status report
                if cycle_count % 10 == 0:  # Every 10 cycles
                    await self._generate_status_report(cycle_count)
                
                # Wait before next cycle
                cycle_duration = (datetime.now() - cycle_start).total_seconds()
                wait_time = max(30 - cycle_duration, 5)  # At least 5 seconds between cycles
                await asyncio.sleep(wait_time)
                
        except Exception as e:
            logger.error(f"❌ Error in main trading loop: {e}")
            raise
    
    async def _apply_risk_management(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Apply advanced risk management to signals"""
        risk_approved_signals = {}
        
        # Get current positions
        current_positions = list(self.components['multi_symbol_manager'].active_positions.values())
        
        for symbol, signal_data in signals.items():
            try:
                prediction = signal_data['prediction']
                config = signal_data['config']
                
                # Calculate position sizing
                position_sizing = self.components['risk_manager'].calculate_confidence_based_position_size(
                    symbol=symbol,
                    confidence=prediction.get('confidence', 0),
                    quality_score=prediction.get('quality_score', 0),
                    market_volatility=0.02,  # Would be calculated from real data
                    base_position_pct=config.position_size_pct
                )
                
                # Create mock position for risk checking
                mock_position = {
                    'symbol': symbol,
                    'position_value': position_sizing.recommended_size,
                    'risk_amount': position_sizing.recommended_size * 0.02  # 2% risk
                }
                
                # Check risk limits
                risk_approved, reason = self.components['risk_manager'].check_position_risk_limits(
                    mock_position, current_positions
                )
                
                if risk_approved:
                    # Update signal with risk-adjusted position size
                    signal_data['position_sizing'] = position_sizing
                    risk_approved_signals[symbol] = signal_data
                    logger.debug(f"✅ Risk approved signal for {symbol}: ${position_sizing.recommended_size:.2f}")
                else:
                    logger.debug(f"❌ Risk rejected signal for {symbol}: {reason}")
                
            except Exception as e:
                logger.error(f"Error applying risk management to {symbol}: {e}")
        
        return risk_approved_signals
    
    async def _update_performance_metrics(self, closed_positions: List[Dict[str, Any]]):
        """Update performance metrics from closed positions"""
        try:
            for position in closed_positions:
                # Update risk manager portfolio value
                pnl_amount = position.get('pnl', 0)
                self.components['risk_manager'].portfolio_value += pnl_amount
                
                # Log performance update
                symbol = position.get('symbol', 'UNKNOWN')
                pnl_pct = position.get('pnl_pct', 0)
                logger.info(f"📊 Performance update for {symbol}: {pnl_pct:+.2f}% "
                           f"(Portfolio: ${self.components['risk_manager'].portfolio_value:.2f})")
        
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    async def _generate_status_report(self, cycle_count: int):
        """Generate comprehensive status report"""
        try:
            logger.info("\n" + "="*60)
            logger.info(f"WEEK 2 SYSTEM STATUS REPORT - Cycle {cycle_count}")
            logger.info("="*60)
            
            # System uptime
            uptime = datetime.now() - self.start_time
            logger.info(f"System Uptime: {uptime}")
            
            # Component status
            logger.info("\nComponent Status:")
            for component, status in self.component_status.items():
                status_icon = "✅" if status == "running" else "❌"
                logger.info(f"  {status_icon} {component}: {status}")
            
            # Portfolio summary
            portfolio_summary = self.components['multi_symbol_manager'].get_portfolio_summary()
            logger.info(f"\nPortfolio Summary:")
            logger.info(f"  Balance: ${portfolio_summary['portfolio_balance']:.2f}")
            logger.info(f"  Active Positions: {portfolio_summary['active_positions']}")
            logger.info(f"  Total Trades: {portfolio_summary['total_trades']}")
            
            # Symbol performance
            logger.info(f"\nSymbol Performance:")
            for symbol, perf in portfolio_summary['symbol_performance'].items():
                if perf['total_trades'] > 0:
                    logger.info(f"  {symbol}: {perf['total_trades']} trades, "
                               f"{perf['win_rate']:.1%} win rate, "
                               f"${perf['total_pnl']:.2f} P&L")
            
            # Risk metrics
            risk_report = self.components['risk_manager'].generate_risk_report(
                list(self.components['multi_symbol_manager'].active_positions.values())
            )
            logger.info(f"\nRisk Metrics:")
            logger.info(f"  Total Exposure: ${risk_report['total_exposure']:.2f}")
            logger.info(f"  Total Risk: ${risk_report['total_risk']:.2f}")
            logger.info(f"  Risk Percentage: {risk_report['risk_percentage']:.2f}%")
            
            # Retraining status
            retrain_status = self.components['retraining_pipeline'].get_retraining_status()
            logger.info(f"\nRetraining Status:")
            logger.info(f"  Pipeline Running: {retrain_status['is_running']}")
            logger.info(f"  Symbols Retraining: {retrain_status['symbols_retraining']}")
            logger.info(f"  Queue Size: {retrain_status['queue_size']}")
            
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Error generating status report: {e}")
    
    async def stop_system(self):
        """Stop the integrated system gracefully"""
        logger.info("🛑 Stopping Week 2 Enhanced System...")
        
        self.is_running = False
        
        # Stop components in reverse order
        try:
            if 'retraining_pipeline' in self.components:
                await self.components['retraining_pipeline'].stop_pipeline()
                self.component_status['retraining_pipeline'] = 'stopped'
            
            if 'performance_validator' in self.components:
                # Performance validator doesn't need explicit stopping
                self.component_status['performance_validator'] = 'stopped'
            
            if 'multi_symbol_manager' in self.components:
                # Multi-symbol manager doesn't need explicit stopping
                self.component_status['multi_symbol_manager'] = 'stopped'
            
            if 'risk_manager' in self.components:
                # Risk manager doesn't need explicit stopping
                self.component_status['risk_manager'] = 'stopped'
            
            if 'market_data_service' in self.components:
                await self.components['market_data_service'].stop()
                self.component_status['market_data_service'] = 'stopped'
            
            logger.info("✅ Week 2 Enhanced System stopped gracefully")
            
        except Exception as e:
            logger.error(f"Error stopping system: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'is_running': self.is_running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'uptime': str(datetime.now() - self.start_time) if self.start_time else None,
            'component_status': self.component_status,
            'portfolio_summary': self.components.get('multi_symbol_manager', {}).get_portfolio_summary() if 'multi_symbol_manager' in self.components else {}
        }


# Global instance
integration_manager = None

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    if integration_manager:
        asyncio.create_task(integration_manager.stop_system())
    sys.exit(0)

async def main():
    """Main function to run Week 2 integration"""
    global integration_manager
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("🚀 Starting Week 2 Enhanced SmartMarketOOPS Integration")
    logger.info("="*80)
    logger.info("Components:")
    logger.info("  📊 Real Market Data Integration")
    logger.info("  🎯 Multi-Symbol Trading Expansion")
    logger.info("  🛡️  Advanced Risk Management")
    logger.info("  📈 Live Performance Validation")
    logger.info("  🔄 Automated Model Retraining")
    logger.info("="*80)
    
    try:
        # Create and start integration manager
        integration_manager = Week2IntegrationManager()
        await integration_manager.start_integrated_system()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error in Week 2 integration: {e}")
        raise
    finally:
        if integration_manager:
            await integration_manager.stop_system()


if __name__ == "__main__":
    asyncio.run(main())
