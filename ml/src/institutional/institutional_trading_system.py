#!/usr/bin/env python3
"""
Institutional-Grade Trading System for Enhanced SmartMarketOOPS
Implements TWAP, VWAP, iceberg orders, and enterprise features
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Advanced order types"""
    MARKET = "market"
    LIMIT = "limit"
    TWAP = "twap"
    VWAP = "vwap"
    ICEBERG = "iceberg"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    BRACKET = "bracket"


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    ACTIVE = "active"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class InstitutionalOrder:
    """Institutional order structure"""
    order_id: str
    client_id: str
    symbol: str
    order_type: OrderType
    side: str  # 'buy' or 'sell'
    quantity: float
    price: Optional[float] = None
    
    # TWAP/VWAP parameters
    duration_minutes: Optional[int] = None
    slice_size: Optional[float] = None
    participation_rate: Optional[float] = None
    
    # Iceberg parameters
    visible_quantity: Optional[float] = None
    
    # Status and tracking
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Execution tracking
    child_orders: List[str] = field(default_factory=list)
    execution_log: List[Dict[str, Any]] = field(default_factory=list)


class TWAPExecutor:
    """Time-Weighted Average Price execution algorithm"""
    
    def __init__(self):
        """Initialize TWAP executor"""
        self.active_orders = {}
        self.execution_schedules = {}
        
    async def execute_twap_order(self, order: InstitutionalOrder, 
                               market_data_feed) -> Dict[str, Any]:
        """Execute TWAP order"""
        logger.info(f"Starting TWAP execution for order {order.order_id}")
        
        # Calculate execution parameters
        total_duration = timedelta(minutes=order.duration_minutes)
        slice_interval = total_duration / 10  # 10 slices by default
        slice_quantity = order.quantity / 10
        
        if order.slice_size:
            slice_quantity = min(slice_quantity, order.slice_size)
        
        execution_plan = []
        remaining_quantity = order.quantity
        current_time = datetime.now()
        
        # Create execution schedule
        for i in range(10):
            if remaining_quantity <= 0:
                break
                
            slice_qty = min(slice_quantity, remaining_quantity)
            execution_time = current_time + (slice_interval * i)
            
            execution_plan.append({
                'execution_time': execution_time,
                'quantity': slice_qty,
                'slice_number': i + 1
            })
            
            remaining_quantity -= slice_qty
        
        # Store execution plan
        self.execution_schedules[order.order_id] = execution_plan
        self.active_orders[order.order_id] = order
        
        # Start execution
        await self._execute_twap_slices(order, execution_plan, market_data_feed)
        
        return {
            'order_id': order.order_id,
            'execution_plan': execution_plan,
            'status': 'started'
        }
    
    async def _execute_twap_slices(self, order: InstitutionalOrder, 
                                 execution_plan: List[Dict[str, Any]], 
                                 market_data_feed):
        """Execute TWAP slices according to schedule"""
        
        for slice_info in execution_plan:
            # Wait until execution time
            wait_time = (slice_info['execution_time'] - datetime.now()).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            
            # Get current market data
            market_data = await market_data_feed.get_current_data(order.symbol)
            
            # Determine execution price (use mid-price for TWAP)
            if market_data:
                execution_price = (market_data['bid'] + market_data['ask']) / 2
            else:
                execution_price = order.price or 0
            
            # Execute slice
            fill_result = await self._execute_slice(
                order, slice_info['quantity'], execution_price, slice_info['slice_number']
            )
            
            # Update order status
            order.filled_quantity += fill_result['filled_quantity']
            order.average_fill_price = self._calculate_average_price(order)
            order.updated_at = datetime.now()
            
            # Log execution
            order.execution_log.append({
                'timestamp': datetime.now(),
                'slice_number': slice_info['slice_number'],
                'quantity': slice_info['quantity'],
                'filled_quantity': fill_result['filled_quantity'],
                'price': execution_price,
                'type': 'twap_slice'
            })
            
            logger.info(f"TWAP slice {slice_info['slice_number']} executed: "
                       f"{fill_result['filled_quantity']}@{execution_price}")
        
        # Mark order as completed
        if order.filled_quantity >= order.quantity * 0.99:  # 99% filled threshold
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
    
    async def _execute_slice(self, order: InstitutionalOrder, quantity: float, 
                           price: float, slice_number: int) -> Dict[str, Any]:
        """Execute individual slice (simulated)"""
        # In real implementation, this would send orders to exchange
        
        # Simulate partial fills and market impact
        market_impact = np.random.uniform(0.0001, 0.001)  # 0.01% to 0.1% impact
        execution_price = price * (1 + market_impact if order.side == 'buy' else 1 - market_impact)
        
        # Simulate fill ratio (90-100% fill rate)
        fill_ratio = np.random.uniform(0.9, 1.0)
        filled_quantity = quantity * fill_ratio
        
        return {
            'filled_quantity': filled_quantity,
            'execution_price': execution_price,
            'market_impact': market_impact,
            'slice_number': slice_number
        }
    
    def _calculate_average_price(self, order: InstitutionalOrder) -> float:
        """Calculate volume-weighted average fill price"""
        if not order.execution_log:
            return 0.0
        
        total_value = 0.0
        total_quantity = 0.0
        
        for execution in order.execution_log:
            quantity = execution['filled_quantity']
            price = execution['price']
            total_value += quantity * price
            total_quantity += quantity
        
        return total_value / total_quantity if total_quantity > 0 else 0.0


class VWAPExecutor:
    """Volume-Weighted Average Price execution algorithm"""
    
    def __init__(self):
        """Initialize VWAP executor"""
        self.active_orders = {}
        self.volume_profiles = {}
    
    async def execute_vwap_order(self, order: InstitutionalOrder, 
                               market_data_feed) -> Dict[str, Any]:
        """Execute VWAP order"""
        logger.info(f"Starting VWAP execution for order {order.order_id}")
        
        # Get historical volume profile
        volume_profile = await self._get_volume_profile(order.symbol, market_data_feed)
        
        # Calculate execution schedule based on volume profile
        execution_plan = self._create_vwap_schedule(order, volume_profile)
        
        # Store and execute
        self.active_orders[order.order_id] = order
        await self._execute_vwap_slices(order, execution_plan, market_data_feed)
        
        return {
            'order_id': order.order_id,
            'execution_plan': execution_plan,
            'volume_profile': volume_profile,
            'status': 'started'
        }
    
    async def _get_volume_profile(self, symbol: str, market_data_feed) -> List[float]:
        """Get historical volume profile for VWAP calculation"""
        # In real implementation, this would fetch historical volume data
        
        # Simulate typical intraday volume profile (U-shaped)
        hours = 24
        volume_profile = []
        
        for hour in range(hours):
            if hour < 2 or hour > 22:  # High volume at market open/close
                volume_weight = 1.5
            elif 6 <= hour <= 18:  # Normal trading hours
                volume_weight = 1.0
            else:  # Low volume overnight
                volume_weight = 0.3
            
            # Add some randomness
            volume_weight *= np.random.uniform(0.8, 1.2)
            volume_profile.append(volume_weight)
        
        # Normalize to sum to 1
        total_volume = sum(volume_profile)
        return [v / total_volume for v in volume_profile]
    
    def _create_vwap_schedule(self, order: InstitutionalOrder, 
                            volume_profile: List[float]) -> List[Dict[str, Any]]:
        """Create VWAP execution schedule"""
        execution_plan = []
        participation_rate = order.participation_rate or 0.1  # 10% default
        
        total_duration = timedelta(minutes=order.duration_minutes)
        slice_duration = total_duration / len(volume_profile)
        
        current_time = datetime.now()
        
        for i, volume_weight in enumerate(volume_profile):
            # Calculate slice quantity based on volume profile
            slice_quantity = order.quantity * volume_weight * participation_rate
            
            if slice_quantity > 0:
                execution_plan.append({
                    'execution_time': current_time + (slice_duration * i),
                    'quantity': slice_quantity,
                    'volume_weight': volume_weight,
                    'slice_number': i + 1
                })
        
        return execution_plan
    
    async def _execute_vwap_slices(self, order: InstitutionalOrder, 
                                 execution_plan: List[Dict[str, Any]], 
                                 market_data_feed):
        """Execute VWAP slices"""
        
        for slice_info in execution_plan:
            # Wait until execution time
            wait_time = (slice_info['execution_time'] - datetime.now()).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            
            # Get current market data and volume
            market_data = await market_data_feed.get_current_data(order.symbol)
            
            if market_data:
                # Calculate VWAP-adjusted price
                current_vwap = await self._calculate_current_vwap(order.symbol, market_data_feed)
                execution_price = current_vwap
            else:
                execution_price = order.price or 0
            
            # Execute slice
            fill_result = await self._execute_vwap_slice(
                order, slice_info['quantity'], execution_price, slice_info
            )
            
            # Update order
            order.filled_quantity += fill_result['filled_quantity']
            order.average_fill_price = self._calculate_vwap_average_price(order)
            order.updated_at = datetime.now()
            
            # Log execution
            order.execution_log.append({
                'timestamp': datetime.now(),
                'slice_number': slice_info['slice_number'],
                'quantity': slice_info['quantity'],
                'filled_quantity': fill_result['filled_quantity'],
                'price': execution_price,
                'volume_weight': slice_info['volume_weight'],
                'type': 'vwap_slice'
            })
    
    async def _calculate_current_vwap(self, symbol: str, market_data_feed) -> float:
        """Calculate current VWAP"""
        # Simplified VWAP calculation
        market_data = await market_data_feed.get_current_data(symbol)
        if market_data:
            return (market_data['bid'] + market_data['ask']) / 2
        return 0.0
    
    async def _execute_vwap_slice(self, order: InstitutionalOrder, quantity: float, 
                                price: float, slice_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute VWAP slice"""
        # Simulate execution with volume-based market impact
        volume_impact = slice_info['volume_weight'] * 0.0005  # Higher volume = lower impact
        execution_price = price * (1 + volume_impact if order.side == 'buy' else 1 - volume_impact)
        
        # Better fill rates during high volume periods
        base_fill_rate = 0.95
        volume_bonus = slice_info['volume_weight'] * 0.05
        fill_ratio = min(base_fill_rate + volume_bonus, 1.0)
        
        filled_quantity = quantity * fill_ratio
        
        return {
            'filled_quantity': filled_quantity,
            'execution_price': execution_price,
            'volume_impact': volume_impact
        }
    
    def _calculate_vwap_average_price(self, order: InstitutionalOrder) -> float:
        """Calculate VWAP average price"""
        if not order.execution_log:
            return 0.0
        
        total_value = 0.0
        total_quantity = 0.0
        
        for execution in order.execution_log:
            if execution['type'] == 'vwap_slice':
                quantity = execution['filled_quantity']
                price = execution['price']
                total_value += quantity * price
                total_quantity += quantity
        
        return total_value / total_quantity if total_quantity > 0 else 0.0


class IcebergExecutor:
    """Iceberg order execution algorithm"""
    
    def __init__(self):
        """Initialize iceberg executor"""
        self.active_orders = {}
    
    async def execute_iceberg_order(self, order: InstitutionalOrder, 
                                  market_data_feed) -> Dict[str, Any]:
        """Execute iceberg order"""
        logger.info(f"Starting Iceberg execution for order {order.order_id}")
        
        visible_qty = order.visible_quantity or (order.quantity * 0.1)  # 10% visible by default
        remaining_qty = order.quantity
        
        self.active_orders[order.order_id] = order
        
        slice_number = 1
        while remaining_qty > 0 and order.status not in [OrderStatus.CANCELLED, OrderStatus.FILLED]:
            # Calculate current slice size
            current_slice = min(visible_qty, remaining_qty)
            
            # Get market data
            market_data = await market_data_feed.get_current_data(order.symbol)
            execution_price = order.price
            
            if not execution_price and market_data:
                # Use best bid/ask for market orders
                execution_price = market_data['ask'] if order.side == 'buy' else market_data['bid']
            
            # Execute slice
            fill_result = await self._execute_iceberg_slice(
                order, current_slice, execution_price, slice_number
            )
            
            # Update order
            filled_qty = fill_result['filled_quantity']
            order.filled_quantity += filled_qty
            remaining_qty -= filled_qty
            order.average_fill_price = self._calculate_iceberg_average_price(order)
            order.updated_at = datetime.now()
            
            # Log execution
            order.execution_log.append({
                'timestamp': datetime.now(),
                'slice_number': slice_number,
                'quantity': current_slice,
                'filled_quantity': filled_qty,
                'price': execution_price,
                'remaining_quantity': remaining_qty,
                'type': 'iceberg_slice'
            })
            
            logger.info(f"Iceberg slice {slice_number} executed: "
                       f"{filled_qty}@{execution_price} (remaining: {remaining_qty})")
            
            slice_number += 1
            
            # Wait before next slice (to avoid detection)
            await asyncio.sleep(np.random.uniform(1, 5))  # 1-5 second delay
        
        # Update final status
        if remaining_qty <= 0:
            order.status = OrderStatus.FILLED
        elif order.filled_quantity > 0:
            order.status = OrderStatus.PARTIALLY_FILLED
        
        return {
            'order_id': order.order_id,
            'total_slices': slice_number - 1,
            'filled_quantity': order.filled_quantity,
            'remaining_quantity': remaining_qty,
            'status': order.status.value
        }
    
    async def _execute_iceberg_slice(self, order: InstitutionalOrder, quantity: float, 
                                   price: float, slice_number: int) -> Dict[str, Any]:
        """Execute individual iceberg slice"""
        # Simulate execution with concealment effectiveness
        concealment_factor = 1.0 - (slice_number * 0.01)  # Slightly worse fills as slices increase
        concealment_factor = max(concealment_factor, 0.9)  # Minimum 90% effectiveness
        
        # Market impact increases slightly with each slice
        market_impact = 0.0002 * slice_number  # 0.02% per slice
        execution_price = price * (1 + market_impact if order.side == 'buy' else 1 - market_impact)
        
        # Fill ratio based on concealment
        fill_ratio = 0.95 * concealment_factor
        filled_quantity = quantity * fill_ratio
        
        return {
            'filled_quantity': filled_quantity,
            'execution_price': execution_price,
            'concealment_factor': concealment_factor,
            'market_impact': market_impact
        }
    
    def _calculate_iceberg_average_price(self, order: InstitutionalOrder) -> float:
        """Calculate iceberg average price"""
        if not order.execution_log:
            return 0.0
        
        total_value = 0.0
        total_quantity = 0.0
        
        for execution in order.execution_log:
            if execution['type'] == 'iceberg_slice':
                quantity = execution['filled_quantity']
                price = execution['price']
                total_value += quantity * price
                total_quantity += quantity
        
        return total_value / total_quantity if total_quantity > 0 else 0.0


class InstitutionalOrderManager:
    """Complete institutional order management system"""
    
    def __init__(self):
        """Initialize institutional order manager"""
        self.orders = {}
        self.client_orders = {}
        
        # Execution engines
        self.twap_executor = TWAPExecutor()
        self.vwap_executor = VWAPExecutor()
        self.iceberg_executor = IcebergExecutor()
        
        # Performance tracking
        self.execution_statistics = {
            'total_orders': 0,
            'completed_orders': 0,
            'total_volume': 0.0,
            'average_execution_time': 0.0,
            'slippage_statistics': []
        }
        
        logger.info("Institutional Order Manager initialized")
    
    async def submit_order(self, order: InstitutionalOrder, 
                         market_data_feed) -> Dict[str, Any]:
        """Submit institutional order for execution"""
        
        # Validate order
        validation_result = self._validate_order(order)
        if not validation_result['valid']:
            order.status = OrderStatus.REJECTED
            return validation_result
        
        # Store order
        self.orders[order.order_id] = order
        
        if order.client_id not in self.client_orders:
            self.client_orders[order.client_id] = []
        self.client_orders[order.client_id].append(order.order_id)
        
        # Route to appropriate executor
        try:
            if order.order_type == OrderType.TWAP:
                result = await self.twap_executor.execute_twap_order(order, market_data_feed)
            elif order.order_type == OrderType.VWAP:
                result = await self.vwap_executor.execute_vwap_order(order, market_data_feed)
            elif order.order_type == OrderType.ICEBERG:
                result = await self.iceberg_executor.execute_iceberg_order(order, market_data_feed)
            else:
                result = await self._execute_simple_order(order, market_data_feed)
            
            order.status = OrderStatus.ACTIVE
            self.execution_statistics['total_orders'] += 1
            
            return {
                'success': True,
                'order_id': order.order_id,
                'status': order.status.value,
                'execution_result': result
            }
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            logger.error(f"Error executing order {order.order_id}: {e}")
            return {
                'success': False,
                'order_id': order.order_id,
                'error': str(e)
            }
    
    def _validate_order(self, order: InstitutionalOrder) -> Dict[str, Any]:
        """Validate institutional order"""
        
        # Basic validation
        if order.quantity <= 0:
            return {'valid': False, 'error': 'Invalid quantity'}
        
        if order.order_type in [OrderType.TWAP, OrderType.VWAP] and not order.duration_minutes:
            return {'valid': False, 'error': 'Duration required for TWAP/VWAP orders'}
        
        if order.order_type == OrderType.ICEBERG and not order.visible_quantity:
            order.visible_quantity = order.quantity * 0.1  # Default 10%
        
        if order.order_type in [OrderType.LIMIT, OrderType.ICEBERG] and not order.price:
            return {'valid': False, 'error': 'Price required for limit/iceberg orders'}
        
        return {'valid': True}
    
    async def _execute_simple_order(self, order: InstitutionalOrder, 
                                  market_data_feed) -> Dict[str, Any]:
        """Execute simple market/limit orders"""
        
        market_data = await market_data_feed.get_current_data(order.symbol)
        
        if order.order_type == OrderType.MARKET:
            execution_price = market_data['ask'] if order.side == 'buy' else market_data['bid']
        else:
            execution_price = order.price
        
        # Simulate execution
        fill_ratio = np.random.uniform(0.95, 1.0)
        filled_quantity = order.quantity * fill_ratio
        
        order.filled_quantity = filled_quantity
        order.average_fill_price = execution_price
        order.status = OrderStatus.FILLED if fill_ratio > 0.99 else OrderStatus.PARTIALLY_FILLED
        
        return {
            'filled_quantity': filled_quantity,
            'execution_price': execution_price,
            'fill_ratio': fill_ratio
        }
    
    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order status and execution details"""
        if order_id not in self.orders:
            return None
        
        order = self.orders[order_id]
        
        return {
            'order_id': order_id,
            'client_id': order.client_id,
            'symbol': order.symbol,
            'order_type': order.order_type.value,
            'side': order.side,
            'quantity': order.quantity,
            'filled_quantity': order.filled_quantity,
            'average_fill_price': order.average_fill_price,
            'status': order.status.value,
            'created_at': order.created_at.isoformat(),
            'updated_at': order.updated_at.isoformat(),
            'execution_log': order.execution_log
        }
    
    def get_client_orders(self, client_id: str) -> List[Dict[str, Any]]:
        """Get all orders for a client"""
        if client_id not in self.client_orders:
            return []
        
        client_order_ids = self.client_orders[client_id]
        return [self.get_order_status(order_id) for order_id in client_order_ids]
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution performance statistics"""
        completed_orders = sum(1 for order in self.orders.values() 
                             if order.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED])
        
        if completed_orders > 0:
            total_volume = sum(order.filled_quantity for order in self.orders.values())
            avg_execution_time = sum(
                (order.updated_at - order.created_at).total_seconds() 
                for order in self.orders.values() if order.status != OrderStatus.PENDING
            ) / completed_orders
        else:
            total_volume = 0.0
            avg_execution_time = 0.0
        
        return {
            'total_orders': len(self.orders),
            'completed_orders': completed_orders,
            'completion_rate': completed_orders / max(len(self.orders), 1),
            'total_volume': total_volume,
            'average_execution_time_seconds': avg_execution_time,
            'order_types': {
                order_type.value: sum(1 for order in self.orders.values() 
                                    if order.order_type == order_type)
                for order_type in OrderType
            }
        }


# Mock market data feed for testing
class MockMarketDataFeed:
    """Mock market data feed for testing"""
    
    async def get_current_data(self, symbol: str) -> Dict[str, Any]:
        """Get current market data"""
        base_price = 45000 if 'BTC' in symbol else 2500
        spread = base_price * 0.001  # 0.1% spread
        
        return {
            'symbol': symbol,
            'bid': base_price - spread/2,
            'ask': base_price + spread/2,
            'volume': np.random.uniform(100, 1000),
            'timestamp': datetime.now()
        }


async def main():
    """Test institutional trading system"""
    # Initialize system
    order_manager = InstitutionalOrderManager()
    market_feed = MockMarketDataFeed()
    
    # Create test orders
    orders = [
        InstitutionalOrder(
            order_id=str(uuid.uuid4()),
            client_id="client_001",
            symbol="BTCUSDT",
            order_type=OrderType.TWAP,
            side="buy",
            quantity=10.0,
            duration_minutes=5,
            slice_size=2.0
        ),
        InstitutionalOrder(
            order_id=str(uuid.uuid4()),
            client_id="client_001",
            symbol="ETHUSDT",
            order_type=OrderType.VWAP,
            side="sell",
            quantity=50.0,
            duration_minutes=3,
            participation_rate=0.15
        ),
        InstitutionalOrder(
            order_id=str(uuid.uuid4()),
            client_id="client_002",
            symbol="BTCUSDT",
            order_type=OrderType.ICEBERG,
            side="buy",
            quantity=25.0,
            price=44500,
            visible_quantity=5.0
        )
    ]
    
    # Submit orders
    for order in orders:
        result = await order_manager.submit_order(order, market_feed)
        print(f"Order {order.order_type.value} submitted: {result['success']}")
    
    # Wait for execution
    await asyncio.sleep(10)
    
    # Get statistics
    stats = order_manager.get_execution_statistics()
    print(f"\n🏛️ Institutional Trading Statistics:")
    print(f"Total Orders: {stats['total_orders']}")
    print(f"Completed Orders: {stats['completed_orders']}")
    print(f"Completion Rate: {stats['completion_rate']:.1%}")
    print(f"Total Volume: {stats['total_volume']:.2f}")
    print(f"Average Execution Time: {stats['average_execution_time_seconds']:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
