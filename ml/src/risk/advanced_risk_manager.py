#!/usr/bin/env python3
"""
Advanced Risk Management System for Enhanced SmartMarketOOPS
Implements sophisticated position sizing and portfolio-level risk management
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Risk metrics for a position or portfolio"""
    var_95: float  # Value at Risk (95% confidence)
    var_99: float  # Value at Risk (99% confidence)
    expected_shortfall: float  # Expected Shortfall (CVaR)
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    volatility: float
    beta: float = 1.0
    correlation_risk: float = 0.0


@dataclass
class PositionSizing:
    """Position sizing recommendation"""
    recommended_size: float
    max_size: float
    risk_adjusted_size: float
    confidence_multiplier: float
    volatility_adjustment: float
    correlation_adjustment: float
    kelly_fraction: float


class AdvancedRiskManager:
    """Advanced risk management with confidence-based position sizing"""
    
    def __init__(self, portfolio_value: float = 100000.0):
        """Initialize the advanced risk manager"""
        self.portfolio_value = portfolio_value
        self.max_portfolio_risk = 0.02  # 2% max daily portfolio risk
        self.max_position_risk = 0.005  # 0.5% max risk per position
        self.max_correlation_exposure = 0.3  # 30% max in correlated positions
        
        # Risk-free rate (annualized)
        self.risk_free_rate = 0.02
        
        # Historical data for risk calculations
        self.price_history = {}
        self.return_history = {}
        self.volatility_history = {}
        
        logger.info("Advanced Risk Manager initialized")
    
    def calculate_confidence_based_position_size(
        self,
        symbol: str,
        confidence: float,
        quality_score: float,
        market_volatility: float,
        correlation_risk: float = 0.0,
        base_position_pct: float = 2.0
    ) -> PositionSizing:
        """
        Calculate position size based on confidence, quality, and risk metrics
        
        Args:
            symbol: Trading symbol
            confidence: Model confidence (0-1)
            quality_score: Signal quality score (0-1)
            market_volatility: Current market volatility
            correlation_risk: Correlation risk with existing positions
            base_position_pct: Base position size percentage
            
        Returns:
            PositionSizing object with recommendations
        """
        
        # Confidence multiplier (higher confidence = larger position)
        confidence_multiplier = self._calculate_confidence_multiplier(confidence, quality_score)
        
        # Volatility adjustment (higher volatility = smaller position)
        volatility_adjustment = self._calculate_volatility_adjustment(market_volatility)
        
        # Correlation adjustment (higher correlation = smaller position)
        correlation_adjustment = self._calculate_correlation_adjustment(correlation_risk)
        
        # Kelly Criterion calculation
        kelly_fraction = self._calculate_kelly_fraction(confidence, quality_score, market_volatility)
        
        # Base position size
        base_size = self.portfolio_value * (base_position_pct / 100)
        
        # Apply adjustments
        confidence_adjusted = base_size * confidence_multiplier
        volatility_adjusted = confidence_adjusted * volatility_adjustment
        correlation_adjusted = volatility_adjusted * correlation_adjustment
        kelly_adjusted = min(correlation_adjusted, self.portfolio_value * kelly_fraction)
        
        # Risk-based maximum
        max_risk_amount = self.portfolio_value * self.max_position_risk
        risk_adjusted_size = min(kelly_adjusted, max_risk_amount / (market_volatility * 2))
        
        # Final recommended size
        recommended_size = min(risk_adjusted_size, self.portfolio_value * 0.1)  # Max 10% per position
        
        return PositionSizing(
            recommended_size=recommended_size,
            max_size=self.portfolio_value * 0.1,
            risk_adjusted_size=risk_adjusted_size,
            confidence_multiplier=confidence_multiplier,
            volatility_adjustment=volatility_adjustment,
            correlation_adjustment=correlation_adjustment,
            kelly_fraction=kelly_fraction
        )
    
    def _calculate_confidence_multiplier(self, confidence: float, quality_score: float) -> float:
        """Calculate position size multiplier based on confidence and quality"""
        
        # Combined confidence score
        combined_score = (confidence * 0.7) + (quality_score * 0.3)
        
        # Non-linear scaling for confidence
        if combined_score >= 0.9:
            return 2.0  # Double size for very high confidence
        elif combined_score >= 0.8:
            return 1.5  # 50% larger for high confidence
        elif combined_score >= 0.7:
            return 1.2  # 20% larger for good confidence
        elif combined_score >= 0.6:
            return 1.0  # Normal size for moderate confidence
        elif combined_score >= 0.5:
            return 0.7  # Smaller size for low confidence
        else:
            return 0.3  # Much smaller for very low confidence
    
    def _calculate_volatility_adjustment(self, volatility: float) -> float:
        """Calculate position size adjustment based on volatility"""
        
        # Normalize volatility (assuming 0.02 = 2% daily volatility as baseline)
        normalized_vol = volatility / 0.02
        
        # Inverse relationship: higher volatility = smaller position
        if normalized_vol <= 0.5:
            return 1.3  # Low volatility: increase position
        elif normalized_vol <= 1.0:
            return 1.0  # Normal volatility: no adjustment
        elif normalized_vol <= 1.5:
            return 0.8  # High volatility: reduce position
        elif normalized_vol <= 2.0:
            return 0.6  # Very high volatility: significantly reduce
        else:
            return 0.4  # Extreme volatility: minimal position
    
    def _calculate_correlation_adjustment(self, correlation_risk: float) -> float:
        """Calculate position size adjustment based on correlation risk"""
        
        # Reduce position size based on correlation with existing positions
        if correlation_risk <= 0.3:
            return 1.0  # Low correlation: no adjustment
        elif correlation_risk <= 0.5:
            return 0.9  # Moderate correlation: slight reduction
        elif correlation_risk <= 0.7:
            return 0.7  # High correlation: significant reduction
        else:
            return 0.5  # Very high correlation: major reduction
    
    def _calculate_kelly_fraction(self, confidence: float, quality_score: float, volatility: float) -> float:
        """Calculate Kelly Criterion fraction for optimal position sizing"""
        
        # Estimate win probability from confidence and quality
        win_probability = (confidence * 0.6) + (quality_score * 0.4)
        
        # Estimate average win/loss ratio (simplified)
        # Higher quality signals tend to have better win/loss ratios
        avg_win_loss_ratio = 1.0 + (quality_score * 1.5)
        
        # Kelly fraction = (bp - q) / b
        # where b = odds received on the wager (avg_win_loss_ratio)
        #       p = probability of winning (win_probability)
        #       q = probability of losing (1 - win_probability)
        
        if win_probability > 0.5 and avg_win_loss_ratio > 1.0:
            kelly_fraction = (
                (avg_win_loss_ratio * win_probability - (1 - win_probability)) / 
                avg_win_loss_ratio
            )
            
            # Cap Kelly fraction to prevent excessive leverage
            kelly_fraction = min(kelly_fraction, 0.25)  # Max 25% of portfolio
            
            # Adjust for volatility
            volatility_adjusted_kelly = kelly_fraction * (0.02 / max(volatility, 0.01))
            
            return max(0.01, min(volatility_adjusted_kelly, 0.15))  # Between 1% and 15%
        else:
            return 0.01  # Minimal position for low-probability trades
    
    def calculate_portfolio_risk_metrics(self, positions: List[Dict[str, Any]]) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        
        if not positions:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0)
        
        # Calculate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(positions)
        
        if len(portfolio_returns) < 30:  # Need sufficient data
            return RiskMetrics(0, 0, 0, 0, 0, 0, np.std(portfolio_returns) if portfolio_returns else 0)
        
        # Value at Risk calculations
        var_95 = np.percentile(portfolio_returns, 5)
        var_99 = np.percentile(portfolio_returns, 1)
        
        # Expected Shortfall (Conditional VaR)
        expected_shortfall = np.mean(portfolio_returns[portfolio_returns <= var_95])
        
        # Sharpe Ratio
        excess_returns = portfolio_returns - (self.risk_free_rate / 252)  # Daily risk-free rate
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
        
        # Sortino Ratio (using downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else np.std(portfolio_returns)
        sortino_ratio = np.mean(excess_returns) / downside_deviation if downside_deviation > 0 else 0
        
        # Maximum Drawdown
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Portfolio volatility
        volatility = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=expected_shortfall,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility
        )
    
    def _calculate_portfolio_returns(self, positions: List[Dict[str, Any]]) -> np.ndarray:
        """Calculate historical portfolio returns"""
        # This is a simplified implementation
        # In practice, you would use actual historical position data
        
        # Generate synthetic portfolio returns for demonstration
        # In production, this would use real historical data
        returns = []
        
        for position in positions:
            if 'pnl_pct' in position:
                returns.append(position['pnl_pct'] / 100)
        
        # If we don't have enough real data, generate synthetic returns
        if len(returns) < 30:
            # Generate synthetic returns based on portfolio characteristics
            mean_return = 0.0005  # 0.05% daily return
            volatility = 0.02     # 2% daily volatility
            
            synthetic_returns = np.random.normal(mean_return, volatility, 100)
            returns.extend(synthetic_returns)
        
        return np.array(returns)
    
    def check_position_risk_limits(self, new_position: Dict[str, Any], existing_positions: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Check if a new position would violate risk limits"""
        
        # Check individual position risk
        position_risk = new_position.get('risk_amount', 0)
        max_position_risk_amount = self.portfolio_value * self.max_position_risk
        
        if position_risk > max_position_risk_amount:
            return False, f"Position risk ${position_risk:.2f} exceeds limit ${max_position_risk_amount:.2f}"
        
        # Check portfolio risk
        total_risk = position_risk + sum(pos.get('risk_amount', 0) for pos in existing_positions)
        max_portfolio_risk_amount = self.portfolio_value * self.max_portfolio_risk
        
        if total_risk > max_portfolio_risk_amount:
            return False, f"Portfolio risk ${total_risk:.2f} exceeds limit ${max_portfolio_risk_amount:.2f}"
        
        # Check correlation exposure
        correlation_exposure = self._calculate_correlation_exposure(new_position, existing_positions)
        if correlation_exposure > self.max_correlation_exposure:
            return False, f"Correlation exposure {correlation_exposure:.1%} exceeds limit {self.max_correlation_exposure:.1%}"
        
        return True, "Position approved"
    
    def _calculate_correlation_exposure(self, new_position: Dict[str, Any], existing_positions: List[Dict[str, Any]]) -> float:
        """Calculate correlation exposure for a new position"""
        
        # Simplified correlation calculation
        # In practice, this would use actual correlation matrices
        
        new_symbol = new_position.get('symbol', '')
        new_value = new_position.get('position_value', 0)
        
        correlated_value = 0
        
        for position in existing_positions:
            existing_symbol = position.get('symbol', '')
            existing_value = position.get('position_value', 0)
            
            # Simplified correlation rules
            correlation = self._get_symbol_correlation(new_symbol, existing_symbol)
            
            if correlation > 0.7:  # High correlation threshold
                correlated_value += existing_value
        
        total_correlated_value = correlated_value + new_value
        return total_correlated_value / self.portfolio_value
    
    def _get_symbol_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols (simplified)"""
        
        # Simplified correlation matrix
        correlations = {
            ('BTCUSDT', 'ETHUSDT'): 0.8,
            ('BTCUSDT', 'SOLUSDT'): 0.7,
            ('BTCUSDT', 'ADAUSDT'): 0.6,
            ('ETHUSDT', 'SOLUSDT'): 0.75,
            ('ETHUSDT', 'ADAUSDT'): 0.65,
            ('SOLUSDT', 'ADAUSDT'): 0.6
        }
        
        # Check both directions
        correlation = correlations.get((symbol1, symbol2), correlations.get((symbol2, symbol1), 0.0))
        return correlation
    
    def optimize_portfolio_allocation(self, signals: Dict[str, Any]) -> Dict[str, float]:
        """Optimize portfolio allocation across multiple signals"""
        
        if not signals:
            return {}
        
        # Extract signal metrics
        symbols = list(signals.keys())
        confidences = [signals[s]['prediction'].get('confidence', 0) for s in symbols]
        quality_scores = [signals[s]['prediction'].get('quality_score', 0) for s in symbols]
        
        # Calculate combined scores
        combined_scores = [(c * 0.7 + q * 0.3) for c, q in zip(confidences, quality_scores)]
        
        # Normalize scores to sum to 1
        total_score = sum(combined_scores)
        if total_score == 0:
            return {}
        
        normalized_scores = [score / total_score for score in combined_scores]
        
        # Apply risk constraints
        max_allocation = 0.3  # Max 30% per symbol
        risk_adjusted_allocations = {}
        
        for symbol, allocation in zip(symbols, normalized_scores):
            # Apply maximum allocation constraint
            constrained_allocation = min(allocation, max_allocation)
            
            # Apply correlation constraints
            correlation_adjustment = self._get_correlation_adjustment_for_allocation(symbol, symbols)
            final_allocation = constrained_allocation * correlation_adjustment
            
            risk_adjusted_allocations[symbol] = final_allocation
        
        # Renormalize to ensure allocations sum to reasonable total
        total_allocation = sum(risk_adjusted_allocations.values())
        max_total_allocation = 0.8  # Max 80% of portfolio allocated
        
        if total_allocation > max_total_allocation:
            scaling_factor = max_total_allocation / total_allocation
            risk_adjusted_allocations = {
                symbol: allocation * scaling_factor 
                for symbol, allocation in risk_adjusted_allocations.items()
            }
        
        return risk_adjusted_allocations
    
    def _get_correlation_adjustment_for_allocation(self, symbol: str, all_symbols: List[str]) -> float:
        """Get correlation adjustment factor for portfolio allocation"""
        
        # Count highly correlated symbols
        high_correlation_count = 0
        
        for other_symbol in all_symbols:
            if other_symbol != symbol:
                correlation = self._get_symbol_correlation(symbol, other_symbol)
                if correlation > 0.7:
                    high_correlation_count += 1
        
        # Reduce allocation if many correlated symbols
        if high_correlation_count >= 3:
            return 0.5  # 50% reduction
        elif high_correlation_count >= 2:
            return 0.7  # 30% reduction
        elif high_correlation_count >= 1:
            return 0.85  # 15% reduction
        else:
            return 1.0  # No reduction
    
    def generate_risk_report(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        
        risk_metrics = self.calculate_portfolio_risk_metrics(positions)
        
        # Calculate current exposures
        total_exposure = sum(pos.get('position_value', 0) for pos in positions)
        total_risk = sum(pos.get('risk_amount', 0) for pos in positions)
        
        # Symbol exposures
        symbol_exposures = {}
        for position in positions:
            symbol = position.get('symbol', 'UNKNOWN')
            value = position.get('position_value', 0)
            symbol_exposures[symbol] = symbol_exposures.get(symbol, 0) + value
        
        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': self.portfolio_value,
            'total_exposure': total_exposure,
            'total_risk': total_risk,
            'exposure_percentage': (total_exposure / self.portfolio_value) * 100,
            'risk_percentage': (total_risk / self.portfolio_value) * 100,
            'risk_metrics': {
                'var_95': risk_metrics.var_95,
                'var_99': risk_metrics.var_99,
                'expected_shortfall': risk_metrics.expected_shortfall,
                'sharpe_ratio': risk_metrics.sharpe_ratio,
                'sortino_ratio': risk_metrics.sortino_ratio,
                'max_drawdown': risk_metrics.max_drawdown,
                'volatility': risk_metrics.volatility
            },
            'symbol_exposures': symbol_exposures,
            'risk_limits': {
                'max_portfolio_risk': self.max_portfolio_risk,
                'max_position_risk': self.max_position_risk,
                'max_correlation_exposure': self.max_correlation_exposure
            },
            'limit_utilization': {
                'portfolio_risk': (total_risk / self.portfolio_value) / self.max_portfolio_risk,
                'exposure': (total_exposure / self.portfolio_value) / 0.8  # Assuming 80% max exposure
            }
        }


def main():
    """Test the advanced risk manager"""
    risk_manager = AdvancedRiskManager(portfolio_value=100000)
    
    # Test position sizing
    position_sizing = risk_manager.calculate_confidence_based_position_size(
        symbol='BTCUSDT',
        confidence=0.85,
        quality_score=0.75,
        market_volatility=0.025,
        correlation_risk=0.3
    )
    
    print(f"Position Sizing Recommendation:")
    print(f"  Recommended Size: ${position_sizing.recommended_size:.2f}")
    print(f"  Confidence Multiplier: {position_sizing.confidence_multiplier:.2f}")
    print(f"  Kelly Fraction: {position_sizing.kelly_fraction:.3f}")
    
    # Test risk report
    sample_positions = [
        {'symbol': 'BTCUSDT', 'position_value': 5000, 'risk_amount': 100, 'pnl_pct': 2.5},
        {'symbol': 'ETHUSDT', 'position_value': 3000, 'risk_amount': 75, 'pnl_pct': -1.2}
    ]
    
    risk_report = risk_manager.generate_risk_report(sample_positions)
    print(f"\nRisk Report:")
    print(f"  Total Exposure: ${risk_report['total_exposure']:.2f}")
    print(f"  Total Risk: ${risk_report['total_risk']:.2f}")
    print(f"  Sharpe Ratio: {risk_report['risk_metrics']['sharpe_ratio']:.3f}")


if __name__ == "__main__":
    main()
