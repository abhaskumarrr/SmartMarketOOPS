"""
Load Testing for SmartMarketOOPS Trading System
Tests system performance under high load and stress conditions
"""

from locust import HttpUser, task, between, events
import json
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingSystemUser(HttpUser):
    """Simulates a trading system user making API calls"""
    
    wait_time = between(0.1, 2.0)  # Wait 0.1-2 seconds between requests
    
    def on_start(self):
        """Initialize user session"""
        self.symbols = ['BTCUSD', 'ETHUSD', 'SOLUSD']
        self.timeframes = ['5m', '15m', '1h', '4h']
        self.user_id = f"user_{random.randint(1000, 9999)}"
        logger.info(f"Starting session for {self.user_id}")
    
    @task(3)
    def get_portfolio_data(self):
        """Test portfolio data endpoint - high frequency"""
        with self.client.get(
            "/api/portfolio",
            headers={"User-Agent": f"LoadTest-{self.user_id}"},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if 'totalValue' in data and 'positions' in data:
                    response.success()
                else:
                    response.failure("Invalid portfolio data structure")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(2)
    def get_positions(self):
        """Test positions endpoint - medium frequency"""
        with self.client.get(
            "/api/positions",
            headers={"User-Agent": f"LoadTest-{self.user_id}"},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    response.success()
                else:
                    response.failure("Invalid positions data structure")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(4)
    def get_market_data(self):
        """Test market data endpoint - highest frequency"""
        symbol = random.choice(self.symbols)
        timeframe = random.choice(self.timeframes)
        
        with self.client.get(
            f"/api/market-data/{symbol}",
            params={"timeframe": timeframe, "limit": 100},
            headers={"User-Agent": f"LoadTest-{self.user_id}"},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    # Validate data structure
                    required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    if all(field in data[0] for field in required_fields):
                        response.success()
                    else:
                        response.failure("Invalid market data structure")
                else:
                    response.failure("Empty market data response")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def get_trading_signals(self):
        """Test trading signals endpoint - low frequency"""
        with self.client.get(
            "/api/signals",
            headers={"User-Agent": f"LoadTest-{self.user_id}"},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    response.success()
                else:
                    response.failure("Invalid signals data structure")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def execute_signal(self):
        """Test signal execution endpoint - low frequency"""
        signal_id = f"signal_{random.randint(1, 100)}"
        
        with self.client.post(
            f"/api/signals/{signal_id}/execute",
            json={"confirm": True},
            headers={"User-Agent": f"LoadTest-{self.user_id}"},
            catch_response=True
        ) as response:
            if response.status_code in [200, 201, 404]:  # 404 is acceptable for non-existent signals
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(2)
    def get_performance_metrics(self):
        """Test performance metrics endpoint"""
        with self.client.get(
            "/api/performance",
            headers={"User-Agent": f"LoadTest-{self.user_id}"},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                required_metrics = ['totalTrades', 'winRate', 'sharpeRatio']
                if all(metric in data for metric in required_metrics):
                    response.success()
                else:
                    response.failure("Invalid performance metrics structure")
            else:
                response.failure(f"Status code: {response.status_code}")

class WebSocketUser(HttpUser):
    """Simulates WebSocket connections for real-time data"""
    
    wait_time = between(1, 5)
    
    def on_start(self):
        """Initialize WebSocket connection simulation"""
        self.user_id = f"ws_user_{random.randint(1000, 9999)}"
        logger.info(f"Starting WebSocket simulation for {self.user_id}")
    
    @task
    def simulate_websocket_connection(self):
        """Simulate WebSocket connection load"""
        # Since Locust doesn't natively support WebSocket load testing,
        # we simulate the HTTP upgrade request and connection overhead
        with self.client.get(
            "/ws/connect",
            headers={
                "Upgrade": "websocket",
                "Connection": "Upgrade",
                "User-Agent": f"WSLoadTest-{self.user_id}"
            },
            catch_response=True
        ) as response:
            # Accept any response as WebSocket endpoints may not be HTTP-testable
            response.success()

class HighFrequencyTradingUser(HttpUser):
    """Simulates high-frequency trading scenarios"""
    
    wait_time = between(0.01, 0.1)  # Very fast requests
    
    def on_start(self):
        self.user_id = f"hft_user_{random.randint(1000, 9999)}"
        self.request_count = 0
        logger.info(f"Starting HFT simulation for {self.user_id}")
    
    @task
    def rapid_market_data_requests(self):
        """Simulate rapid market data requests"""
        symbol = random.choice(['BTCUSD', 'ETHUSD'])
        
        with self.client.get(
            f"/api/market-data/{symbol}/latest",
            headers={"User-Agent": f"HFT-{self.user_id}"},
            catch_response=True
        ) as response:
            self.request_count += 1
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
            
            # Log every 100 requests
            if self.request_count % 100 == 0:
                logger.info(f"HFT User {self.user_id}: {self.request_count} requests completed")

# Event handlers for custom metrics
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, context, **kwargs):
    """Custom request handler for detailed metrics"""
    if exception:
        logger.error(f"Request failed: {name} - {exception}")
    elif response_time > 1000:  # Log slow requests (>1s)
        logger.warning(f"Slow request: {name} - {response_time}ms")

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Test start handler"""
    logger.info("Load test starting...")
    logger.info(f"Target host: {environment.host}")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Test stop handler with summary"""
    logger.info("Load test completed")
    
    # Calculate and log summary statistics
    stats = environment.stats
    total_requests = stats.total.num_requests
    total_failures = stats.total.num_failures
    avg_response_time = stats.total.avg_response_time
    
    logger.info(f"Total requests: {total_requests}")
    logger.info(f"Total failures: {total_failures}")
    logger.info(f"Failure rate: {(total_failures/total_requests)*100:.2f}%")
    logger.info(f"Average response time: {avg_response_time:.2f}ms")
    
    # Performance thresholds
    if avg_response_time > 500:
        logger.warning("Average response time exceeds 500ms threshold")
    if (total_failures/total_requests) > 0.01:  # 1% failure rate
        logger.error("Failure rate exceeds 1% threshold")

# Custom load test scenarios
class StressTestUser(HttpUser):
    """Stress test with extreme load patterns"""
    
    wait_time = between(0.001, 0.01)  # Extreme frequency
    
    @task
    def stress_test_portfolio(self):
        """Stress test portfolio endpoint"""
        with self.client.get("/api/portfolio", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Stress test failure: {response.status_code}")
            else:
                response.success()

# Test configuration for different scenarios
if __name__ == "__main__":
    import subprocess
    import sys
    
    # Run different test scenarios
    scenarios = {
        "normal_load": {
            "users": 50,
            "spawn_rate": 5,
            "duration": "5m",
            "user_class": "TradingSystemUser"
        },
        "high_load": {
            "users": 200,
            "spawn_rate": 10,
            "duration": "10m",
            "user_class": "TradingSystemUser"
        },
        "stress_test": {
            "users": 500,
            "spawn_rate": 20,
            "duration": "15m",
            "user_class": "StressTestUser"
        },
        "websocket_load": {
            "users": 100,
            "spawn_rate": 10,
            "duration": "5m",
            "user_class": "WebSocketUser"
        }
    }
    
    print("Available load test scenarios:")
    for name, config in scenarios.items():
        print(f"  {name}: {config['users']} users, {config['duration']}")
    
    print("\nRun with: locust -f locustfile.py --host=http://localhost:8000")
