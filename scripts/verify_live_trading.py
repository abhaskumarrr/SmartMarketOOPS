#!/usr/bin/env python3
"""
Live Trading Verification Script
Validates that the production system is executing real trades correctly
"""

import asyncio
import aiohttp
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LiveTradingVerifier:
    """Verifies live trading operations in production"""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = None
        self.verification_results = {
            'timestamp': datetime.now().isoformat(),
            'system_health': {},
            'trading_verification': {},
            'performance_metrics': {},
            'risk_compliance': {},
            'data_integrity': {},
            'overall_status': 'unknown'
        }
    
    async def create_session(self):
        """Create aiohttp session with authentication"""
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        connector = aiohttp.TCPConnector(limit=10)
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(
            headers=headers,
            connector=connector,
            timeout=timeout
        )
    
    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
    
    async def verify_system_health(self) -> Dict[str, Any]:
        """Verify overall system health"""
        logger.info("Verifying system health...")
        
        health_checks = {
            'api_health': f'{self.base_url}/health',
            'ml_system': f'{self.base_url}/ml/health',
            'bridge_health': f'{self.base_url}/bridge/health',
            'database_health': f'{self.base_url}/db/health',
            'websocket_health': f'{self.base_url}/ws/health'
        }
        
        results = {}
        
        for service, endpoint in health_checks.items():
            try:
                async with self.session.get(endpoint) as response:
                    if response.status == 200:
                        data = await response.json()
                        results[service] = {
                            'status': 'healthy',
                            'response_time': response.headers.get('X-Response-Time', 'unknown'),
                            'details': data
                        }
                    else:
                        results[service] = {
                            'status': 'unhealthy',
                            'error': f'HTTP {response.status}'
                        }
            except Exception as e:
                results[service] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Check if all critical services are healthy
        critical_services = ['api_health', 'ml_system', 'bridge_health']
        all_healthy = all(
            results.get(service, {}).get('status') == 'healthy'
            for service in critical_services
        )
        
        results['overall_health'] = 'healthy' if all_healthy else 'degraded'
        
        return results
    
    async def verify_live_trading(self) -> Dict[str, Any]:
        """Verify live trading operations"""
        logger.info("Verifying live trading operations...")
        
        results = {}
        
        # Check recent trades
        try:
            async with self.session.get(f'{self.base_url}/api/trades/recent') as response:
                if response.status == 200:
                    trades = await response.json()
                    
                    # Analyze recent trading activity
                    recent_trades = [
                        trade for trade in trades
                        if datetime.fromisoformat(trade['timestamp']) > 
                        datetime.now() - timedelta(hours=24)
                    ]
                    
                    results['recent_trades'] = {
                        'count_24h': len(recent_trades),
                        'total_volume': sum(trade.get('volume', 0) for trade in recent_trades),
                        'successful_trades': len([t for t in recent_trades if t.get('status') == 'filled']),
                        'failed_trades': len([t for t in recent_trades if t.get('status') == 'failed'])
                    }
                    
                    # Check trade execution quality
                    if recent_trades:
                        execution_times = [
                            trade.get('execution_time', 0) for trade in recent_trades
                            if trade.get('execution_time')
                        ]
                        
                        results['execution_quality'] = {
                            'avg_execution_time': sum(execution_times) / len(execution_times) if execution_times else 0,
                            'max_execution_time': max(execution_times) if execution_times else 0,
                            'trades_under_1s': len([t for t in execution_times if t < 1000])
                        }
                else:
                    results['recent_trades'] = {'error': f'HTTP {response.status}'}
        
        except Exception as e:
            results['recent_trades'] = {'error': str(e)}
        
        # Check active positions
        try:
            async with self.session.get(f'{self.base_url}/api/positions') as response:
                if response.status == 200:
                    positions = await response.json()
                    
                    active_positions = [p for p in positions if p.get('status') == 'open']
                    
                    results['positions'] = {
                        'total_positions': len(positions),
                        'active_positions': len(active_positions),
                        'total_exposure': sum(
                            abs(pos.get('size', 0) * pos.get('current_price', 0))
                            for pos in active_positions
                        ),
                        'unrealized_pnl': sum(pos.get('unrealized_pnl', 0) for pos in active_positions)
                    }
                else:
                    results['positions'] = {'error': f'HTTP {response.status}'}
        
        except Exception as e:
            results['positions'] = {'error': str(e)}
        
        # Check ML signal generation
        try:
            async with self.session.get(f'{self.base_url}/api/signals/recent') as response:
                if response.status == 200:
                    signals = await response.json()
                    
                    recent_signals = [
                        signal for signal in signals
                        if datetime.fromisoformat(signal['timestamp']) >
                        datetime.now() - timedelta(hours=1)
                    ]
                    
                    results['ml_signals'] = {
                        'signals_1h': len(recent_signals),
                        'avg_confidence': sum(s.get('confidence', 0) for s in recent_signals) / len(recent_signals) if recent_signals else 0,
                        'executed_signals': len([s for s in recent_signals if s.get('executed', False)])
                    }
                else:
                    results['ml_signals'] = {'error': f'HTTP {response.status}'}
        
        except Exception as e:
            results['ml_signals'] = {'error': str(e)}
        
        return results
    
    async def verify_performance_metrics(self) -> Dict[str, Any]:
        """Verify trading performance metrics"""
        logger.info("Verifying performance metrics...")
        
        try:
            async with self.session.get(f'{self.base_url}/api/performance') as response:
                if response.status == 200:
                    performance = await response.json()
                    
                    # Validate against acceptance criteria
                    validation = {
                        'win_rate': {
                            'value': performance.get('winRate', 0),
                            'threshold': 60.0,
                            'passed': performance.get('winRate', 0) >= 60.0
                        },
                        'sharpe_ratio': {
                            'value': performance.get('sharpeRatio', 0),
                            'threshold': 1.5,
                            'passed': performance.get('sharpeRatio', 0) >= 1.5
                        },
                        'max_drawdown': {
                            'value': abs(performance.get('maxDrawdown', 0)),
                            'threshold': 20.0,
                            'passed': abs(performance.get('maxDrawdown', 0)) <= 20.0
                        },
                        'daily_return': {
                            'value': performance.get('dailyReturn', 0),
                            'threshold': 0.5,  # 0.5% minimum daily return
                            'passed': performance.get('dailyReturn', 0) >= 0.5
                        }
                    }
                    
                    return {
                        'raw_metrics': performance,
                        'validation': validation,
                        'overall_performance': all(v['passed'] for v in validation.values())
                    }
                else:
                    return {'error': f'HTTP {response.status}'}
        
        except Exception as e:
            return {'error': str(e)}
    
    async def verify_risk_compliance(self) -> Dict[str, Any]:
        """Verify risk management compliance"""
        logger.info("Verifying risk compliance...")
        
        try:
            async with self.session.get(f'{self.base_url}/api/risk/metrics') as response:
                if response.status == 200:
                    risk_data = await response.json()
                    
                    compliance_checks = {
                        'position_sizing': {
                            'value': risk_data.get('position_size_compliance', 0),
                            'threshold': 95.0,
                            'passed': risk_data.get('position_size_compliance', 0) >= 95.0
                        },
                        'leverage_limits': {
                            'value': risk_data.get('leverage_compliance', 0),
                            'threshold': 100.0,
                            'passed': risk_data.get('leverage_compliance', 0) >= 100.0
                        },
                        'stop_loss_execution': {
                            'value': risk_data.get('stop_loss_execution_rate', 0),
                            'threshold': 99.0,
                            'passed': risk_data.get('stop_loss_execution_rate', 0) >= 99.0
                        },
                        'risk_score': {
                            'value': risk_data.get('overall_risk_score', 0),
                            'threshold': 70.0,  # Maximum acceptable risk score
                            'passed': risk_data.get('overall_risk_score', 0) <= 70.0
                        }
                    }
                    
                    return {
                        'raw_metrics': risk_data,
                        'compliance_checks': compliance_checks,
                        'overall_compliance': all(c['passed'] for c in compliance_checks.values())
                    }
                else:
                    return {'error': f'HTTP {response.status}'}
        
        except Exception as e:
            return {'error': str(e)}
    
    async def verify_data_integrity(self) -> Dict[str, Any]:
        """Verify data integrity and quality"""
        logger.info("Verifying data integrity...")
        
        results = {}
        
        # Check market data freshness
        try:
            async with self.session.get(f'{self.base_url}/api/market-data/BTCUSD/latest') as response:
                if response.status == 200:
                    data = await response.json()
                    
                    last_update = datetime.fromisoformat(data['timestamp'])
                    data_age = (datetime.now() - last_update).total_seconds()
                    
                    results['market_data'] = {
                        'last_update': data['timestamp'],
                        'data_age_seconds': data_age,
                        'is_fresh': data_age < 60,  # Data should be less than 1 minute old
                        'price': data.get('price', 0)
                    }
                else:
                    results['market_data'] = {'error': f'HTTP {response.status}'}
        
        except Exception as e:
            results['market_data'] = {'error': str(e)}
        
        # Check database connectivity
        try:
            async with self.session.get(f'{self.base_url}/api/db/status') as response:
                if response.status == 200:
                    db_status = await response.json()
                    results['database'] = {
                        'connected': db_status.get('connected', False),
                        'latency': db_status.get('latency', 0),
                        'active_connections': db_status.get('active_connections', 0)
                    }
                else:
                    results['database'] = {'error': f'HTTP {response.status}'}
        
        except Exception as e:
            results['database'] = {'error': str(e)}
        
        return results
    
    async def run_comprehensive_verification(self) -> Dict[str, Any]:
        """Run complete live trading verification"""
        logger.info("Starting comprehensive live trading verification...")
        
        await self.create_session()
        
        try:
            # Run all verification checks
            self.verification_results['system_health'] = await self.verify_system_health()
            self.verification_results['trading_verification'] = await self.verify_live_trading()
            self.verification_results['performance_metrics'] = await self.verify_performance_metrics()
            self.verification_results['risk_compliance'] = await self.verify_risk_compliance()
            self.verification_results['data_integrity'] = await self.verify_data_integrity()
            
            # Determine overall status
            health_ok = self.verification_results['system_health'].get('overall_health') == 'healthy'
            performance_ok = self.verification_results['performance_metrics'].get('overall_performance', False)
            compliance_ok = self.verification_results['risk_compliance'].get('overall_compliance', False)
            
            if health_ok and performance_ok and compliance_ok:
                self.verification_results['overall_status'] = 'verified'
            elif health_ok:
                self.verification_results['overall_status'] = 'operational'
            else:
                self.verification_results['overall_status'] = 'issues_detected'
            
            # Save verification report
            report_filename = f'live_trading_verification_{int(time.time())}.json'
            with open(report_filename, 'w') as f:
                json.dump(self.verification_results, f, indent=2, default=str)
            
            logger.info(f"Verification completed. Report saved to {report_filename}")
            
            return self.verification_results
        
        finally:
            await self.close_session()

async def main():
    """Main verification function"""
    parser = argparse.ArgumentParser(description='Verify live trading operations')
    parser.add_argument('--host', default='https://api.smartmarket.com', help='API host URL')
    parser.add_argument('--api-key', help='API key for authentication')
    parser.add_argument('--continuous', action='store_true', help='Run continuous verification')
    parser.add_argument('--interval', type=int, default=300, help='Verification interval in seconds')
    
    args = parser.parse_args()
    
    verifier = LiveTradingVerifier(args.host, args.api_key)
    
    if args.continuous:
        logger.info(f"Starting continuous verification every {args.interval} seconds...")
        
        while True:
            try:
                report = await verifier.run_comprehensive_verification()
                
                status = report['overall_status']
                logger.info(f"Verification completed: {status}")
                
                if status == 'issues_detected':
                    logger.error("Issues detected in live trading system!")
                
                await asyncio.sleep(args.interval)
                
            except KeyboardInterrupt:
                logger.info("Continuous verification stopped by user")
                break
            except Exception as e:
                logger.error(f"Verification failed: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    else:
        try:
            report = await verifier.run_comprehensive_verification()
            
            print("\n" + "="*60)
            print("LIVE TRADING VERIFICATION RESULTS")
            print("="*60)
            
            status = report['overall_status']
            status_emoji = {
                'verified': '✅',
                'operational': '⚠️',
                'issues_detected': '❌'
            }.get(status, '❓')
            
            print(f"Overall Status: {status_emoji} {status.upper()}")
            
            # Print summary
            health = report['system_health']
            print(f"System Health: {health.get('overall_health', 'unknown')}")
            
            trading = report['trading_verification']
            if 'recent_trades' in trading:
                trades = trading['recent_trades']
                print(f"24h Trades: {trades.get('count_24h', 0)}")
                print(f"Success Rate: {trades.get('successful_trades', 0) / max(trades.get('count_24h', 1), 1) * 100:.1f}%")
            
            performance = report['performance_metrics']
            if 'validation' in performance:
                val = performance['validation']
                print(f"Win Rate: {val.get('win_rate', {}).get('value', 0):.1f}%")
                print(f"Sharpe Ratio: {val.get('sharpe_ratio', {}).get('value', 0):.2f}")
            
            print(f"\nDetailed report saved to live_trading_verification_*.json")
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            exit(1)

if __name__ == "__main__":
    asyncio.run(main())
