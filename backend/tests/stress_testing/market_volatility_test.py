"""
Market Volatility Stress Testing
Simulates extreme market conditions to test system resilience
"""

import asyncio
import aiohttp
import numpy as np
import pandas as pd
import json
import time
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import websockets
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketVolatilitySimulator:
    """Simulates extreme market volatility scenarios"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.ws_url = base_url.replace("http", "ws") + "/ws"
        self.session = None
        self.results = {
            'response_times': [],
            'errors': [],
            'successful_requests': 0,
            'failed_requests': 0,
            'start_time': None,
            'end_time': None
        }
    
    async def create_session(self):
        """Create aiohttp session"""
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=50)
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    
    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
    
    def generate_volatile_price_data(self, base_price=45000, volatility=0.1, duration_minutes=60):
        """Generate highly volatile price data"""
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(minutes=duration_minutes),
            end=datetime.now(),
            freq='1S'  # 1-second intervals for high volatility
        )
        
        prices = []
        current_price = base_price
        
        for i, timestamp in enumerate(timestamps):
            # Simulate extreme volatility with occasional flash crashes/spikes
            if i % 300 == 0:  # Every 5 minutes, simulate extreme event
                shock = np.random.choice([-0.15, 0.15])  # ±15% shock
                current_price *= (1 + shock)
            else:
                # Normal high volatility
                change = np.random.normal(0, volatility / 100)
                current_price *= (1 + change)
            
            # Ensure price doesn't go negative
            current_price = max(current_price, 1)
            prices.append(current_price)
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'volatility': volatility
        })
    
    async def send_market_data_burst(self, symbol, data_points=1000):
        """Send burst of market data updates"""
        if not self.session:
            await self.create_session()
        
        start_time = time.time()
        tasks = []
        
        for i in range(data_points):
            price_data = {
                'symbol': symbol,
                'price': 45000 + np.random.normal(0, 1000),
                'timestamp': int(time.time() * 1000) + i,
                'volume': np.random.randint(100, 10000)
            }
            
            task = self.send_price_update(price_data)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        successful = sum(1 for r in results if not isinstance(r, Exception))
        failed = len(results) - successful
        
        logger.info(f"Market data burst completed: {successful}/{len(results)} successful in {duration:.2f}s")
        
        return {
            'successful': successful,
            'failed': failed,
            'duration': duration,
            'throughput': len(results) / duration
        }
    
    async def send_price_update(self, price_data):
        """Send individual price update"""
        try:
            start_time = time.time()
            async with self.session.post(
                f"{self.base_url}/api/market-data/update",
                json=price_data
            ) as response:
                response_time = (time.time() - start_time) * 1000
                self.results['response_times'].append(response_time)
                
                if response.status == 200:
                    self.results['successful_requests'] += 1
                    return await response.json()
                else:
                    self.results['failed_requests'] += 1
                    self.results['errors'].append(f"HTTP {response.status}")
                    return None
        
        except Exception as e:
            self.results['failed_requests'] += 1
            self.results['errors'].append(str(e))
            return None
    
    async def simulate_flash_crash(self, symbol="BTCUSD", crash_percentage=0.3):
        """Simulate flash crash scenario"""
        logger.info(f"Simulating flash crash for {symbol} (-{crash_percentage*100}%)")
        
        # Generate flash crash data
        base_price = 45000
        crash_price = base_price * (1 - crash_percentage)
        recovery_price = base_price * 0.95  # Partial recovery
        
        # Flash crash sequence
        crash_sequence = [
            {'price': base_price, 'delay': 0},
            {'price': base_price * 0.9, 'delay': 1},
            {'price': base_price * 0.7, 'delay': 2},
            {'price': crash_price, 'delay': 3},
            {'price': crash_price * 1.1, 'delay': 5},
            {'price': recovery_price, 'delay': 10}
        ]
        
        tasks = []
        for point in crash_sequence:
            await asyncio.sleep(point['delay'])
            
            price_data = {
                'symbol': symbol,
                'price': point['price'],
                'timestamp': int(time.time() * 1000),
                'volume': np.random.randint(50000, 200000)  # High volume during crash
            }
            
            task = self.send_price_update(price_data)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(f"Flash crash simulation completed: {len([r for r in results if not isinstance(r, Exception)])} updates sent")
        
        return results
    
    async def test_system_under_load(self, concurrent_users=100, duration_seconds=300):
        """Test system under sustained high load"""
        logger.info(f"Starting load test: {concurrent_users} users for {duration_seconds}s")
        
        self.results['start_time'] = time.time()
        
        async def user_simulation():
            """Simulate individual user behavior"""
            while time.time() - self.results['start_time'] < duration_seconds:
                # Random API calls
                endpoints = [
                    '/api/portfolio',
                    '/api/positions',
                    '/api/signals',
                    '/api/performance'
                ]
                
                endpoint = np.random.choice(endpoints)
                
                try:
                    async with self.session.get(f"{self.base_url}{endpoint}") as response:
                        if response.status == 200:
                            self.results['successful_requests'] += 1
                        else:
                            self.results['failed_requests'] += 1
                
                except Exception as e:
                    self.results['failed_requests'] += 1
                    self.results['errors'].append(str(e))
                
                # Random delay between requests
                await asyncio.sleep(np.random.exponential(0.5))
        
        # Create concurrent user tasks
        user_tasks = [user_simulation() for _ in range(concurrent_users)]
        
        # Also simulate market data updates
        market_data_task = self.continuous_market_updates(duration_seconds)
        
        # Run all tasks concurrently
        await asyncio.gather(*user_tasks, market_data_task)
        
        self.results['end_time'] = time.time()
        
        return self.generate_test_report()
    
    async def continuous_market_updates(self, duration_seconds):
        """Send continuous market data updates"""
        symbols = ['BTCUSD', 'ETHUSD', 'SOLUSD']
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            for symbol in symbols:
                price_data = {
                    'symbol': symbol,
                    'price': 45000 + np.random.normal(0, 500),
                    'timestamp': int(time.time() * 1000),
                    'volume': np.random.randint(100, 5000)
                }
                
                await self.send_price_update(price_data)
            
            await asyncio.sleep(0.1)  # 10 updates per second per symbol
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        total_requests = self.results['successful_requests'] + self.results['failed_requests']
        duration = self.results['end_time'] - self.results['start_time']
        
        report = {
            'summary': {
                'total_requests': total_requests,
                'successful_requests': self.results['successful_requests'],
                'failed_requests': self.results['failed_requests'],
                'success_rate': (self.results['successful_requests'] / total_requests * 100) if total_requests > 0 else 0,
                'duration_seconds': duration,
                'requests_per_second': total_requests / duration if duration > 0 else 0
            },
            'response_times': {
                'mean': np.mean(self.results['response_times']) if self.results['response_times'] else 0,
                'median': np.median(self.results['response_times']) if self.results['response_times'] else 0,
                'p95': np.percentile(self.results['response_times'], 95) if self.results['response_times'] else 0,
                'p99': np.percentile(self.results['response_times'], 99) if self.results['response_times'] else 0,
                'max': np.max(self.results['response_times']) if self.results['response_times'] else 0
            },
            'errors': {
                'total_errors': len(self.results['errors']),
                'unique_errors': len(set(self.results['errors'])),
                'error_types': dict(pd.Series(self.results['errors']).value_counts()) if self.results['errors'] else {}
            }
        }
        
        return report
    
    async def run_comprehensive_stress_test(self):
        """Run all stress test scenarios"""
        logger.info("Starting comprehensive stress test suite")
        
        await self.create_session()
        
        test_results = {}
        
        try:
            # Test 1: Market data burst
            logger.info("Test 1: Market data burst test")
            test_results['market_data_burst'] = await self.send_market_data_burst('BTCUSD', 5000)
            
            # Test 2: Flash crash simulation
            logger.info("Test 2: Flash crash simulation")
            test_results['flash_crash'] = await self.simulate_flash_crash('BTCUSD', 0.25)
            
            # Test 3: Sustained load test
            logger.info("Test 3: Sustained load test")
            test_results['sustained_load'] = await self.test_system_under_load(50, 180)
            
            # Test 4: Extreme volatility
            logger.info("Test 4: Extreme volatility simulation")
            volatile_data = self.generate_volatile_price_data(volatility=0.2, duration_minutes=30)
            test_results['extreme_volatility'] = await self.send_market_data_burst('BTCUSD', len(volatile_data))
            
        finally:
            await self.close_session()
        
        # Generate final report
        final_report = {
            'test_timestamp': datetime.now().isoformat(),
            'test_results': test_results,
            'overall_summary': self.generate_test_report()
        }
        
        # Save report to file
        with open(f'stress_test_report_{int(time.time())}.json', 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        logger.info("Comprehensive stress test completed")
        return final_report

async def main():
    """Main stress testing function"""
    simulator = MarketVolatilitySimulator()
    
    try:
        report = await simulator.run_comprehensive_stress_test()
        
        print("\n" + "="*50)
        print("STRESS TEST RESULTS SUMMARY")
        print("="*50)
        
        summary = report['overall_summary']['summary']
        print(f"Total Requests: {summary['total_requests']}")
        print(f"Success Rate: {summary['success_rate']:.2f}%")
        print(f"Requests/Second: {summary['requests_per_second']:.2f}")
        
        response_times = report['overall_summary']['response_times']
        print(f"Average Response Time: {response_times['mean']:.2f}ms")
        print(f"95th Percentile: {response_times['p95']:.2f}ms")
        print(f"99th Percentile: {response_times['p99']:.2f}ms")
        
        errors = report['overall_summary']['errors']
        print(f"Total Errors: {errors['total_errors']}")
        
        print("\nDetailed report saved to stress_test_report_*.json")
        
    except Exception as e:
        logger.error(f"Stress test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
