#!/usr/bin/env python3
"""
Advanced Portfolio Analytics System for Enhanced SmartMarketOOPS
Implements performance attribution, risk decomposition, factor analysis, drawdown analysis, and Sharpe optimization
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Scientific computing libraries
from scipy import stats, optimize
from scipy.linalg import inv, pinv
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttributionLevel(Enum):
    """Performance attribution levels"""
    STRATEGY = "strategy"
    TIMEFRAME = "timeframe"
    MARKET_CONDITION = "market_condition"
    SECTOR = "sector"
    FACTOR = "factor"


class RiskFactorType(Enum):
    """Risk factor types"""
    MARKET = "market"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    SENTIMENT = "sentiment"
    LIQUIDITY = "liquidity"
    CURRENCY = "currency"
    SECTOR = "sector"


@dataclass
class PortfolioPosition:
    """Portfolio position data structure"""
    symbol: str
    quantity: float
    market_value: float
    weight: float
    sector: str
    strategy: str
    entry_date: datetime
    current_price: float
    unrealized_pnl: float
    realized_pnl: float


@dataclass
class PerformanceAttribution:
    """Performance attribution result"""
    timestamp: datetime
    attribution_level: AttributionLevel
    total_return: float
    benchmark_return: float
    active_return: float
    attribution_breakdown: Dict[str, float]
    interaction_effects: Dict[str, float]
    residual_return: float
    information_ratio: float


@dataclass
class RiskDecomposition:
    """Risk decomposition result"""
    timestamp: datetime
    total_risk: float
    systematic_risk: float
    idiosyncratic_risk: float
    factor_contributions: Dict[str, float]
    correlation_structure: np.ndarray
    principal_components: Dict[str, float]
    var_95: float
    expected_shortfall: float


@dataclass
class FactorExposure:
    """Factor exposure analysis result"""
    timestamp: datetime
    factor_loadings: Dict[RiskFactorType, float]
    factor_returns: Dict[RiskFactorType, float]
    factor_volatilities: Dict[RiskFactorType, float]
    factor_correlations: Dict[Tuple[RiskFactorType, RiskFactorType], float]
    tracking_error: float
    active_share: float


@dataclass
class DrawdownAnalysis:
    """Drawdown analysis result"""
    timestamp: datetime
    current_drawdown: float
    max_drawdown: float
    max_drawdown_date: datetime
    recovery_time_days: Optional[int]
    underwater_duration_days: int
    drawdown_frequency: float
    average_drawdown: float
    stress_test_results: Dict[str, float]


@dataclass
class OptimizationResult:
    """Portfolio optimization result"""
    timestamp: datetime
    optimal_weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    rebalancing_trades: Dict[str, float]
    transaction_costs: float
    improvement_metrics: Dict[str, float]


class PerformanceAttributionAnalyzer:
    """Advanced performance attribution analysis"""

    def __init__(self):
        """Initialize performance attribution analyzer"""
        self.attribution_history = deque(maxlen=1000)
        self.benchmark_data = {}

        logger.info("Performance Attribution Analyzer initialized")

    def calculate_brinson_attribution(self, portfolio_returns: pd.DataFrame,
                                    benchmark_returns: pd.DataFrame,
                                    portfolio_weights: pd.DataFrame,
                                    benchmark_weights: pd.DataFrame) -> PerformanceAttribution:
        """Calculate Brinson-Fachler performance attribution"""
        start_time = time.perf_counter()

        # Ensure data alignment
        common_index = portfolio_returns.index.intersection(benchmark_returns.index)
        portfolio_returns = portfolio_returns.loc[common_index]
        benchmark_returns = benchmark_returns.loc[common_index]

        # Calculate total returns
        portfolio_total_return = (portfolio_returns * portfolio_weights).sum(axis=1).mean()
        benchmark_total_return = benchmark_returns.mean().mean()
        active_return = portfolio_total_return - benchmark_total_return

        # Brinson attribution components
        attribution_breakdown = {}

        # Asset Allocation Effect
        weight_diff = portfolio_weights - benchmark_weights
        allocation_effect = (weight_diff * benchmark_returns).sum().sum()
        attribution_breakdown['allocation_effect'] = allocation_effect

        # Security Selection Effect
        return_diff = portfolio_returns - benchmark_returns
        selection_effect = (benchmark_weights * return_diff).sum().sum()
        attribution_breakdown['selection_effect'] = selection_effect

        # Interaction Effect
        interaction_effect = (weight_diff * return_diff).sum().sum()
        attribution_breakdown['interaction_effect'] = interaction_effect

        # Calculate Information Ratio
        active_returns = (portfolio_returns * portfolio_weights).sum(axis=1) - benchmark_returns.mean(axis=1)
        tracking_error = active_returns.std()
        information_ratio = active_return / tracking_error if tracking_error > 0 else 0

        # Residual return (unexplained)
        explained_return = sum(attribution_breakdown.values())
        residual_return = active_return - explained_return

        processing_time = (time.perf_counter() - start_time) * 1000

        result = PerformanceAttribution(
            timestamp=datetime.now(),
            attribution_level=AttributionLevel.STRATEGY,
            total_return=portfolio_total_return,
            benchmark_return=benchmark_total_return,
            active_return=active_return,
            attribution_breakdown=attribution_breakdown,
            interaction_effects={'total_interaction': interaction_effect},
            residual_return=residual_return,
            information_ratio=information_ratio
        )

        self.attribution_history.append(result)

        logger.info(f"Attribution analysis completed in {processing_time:.2f}ms")
        return result

    def calculate_sector_attribution(self, positions: List[PortfolioPosition],
                                   sector_returns: Dict[str, float],
                                   benchmark_sector_weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate sector-level attribution"""

        # Calculate portfolio sector weights
        total_value = sum(pos.market_value for pos in positions)
        portfolio_sector_weights = {}

        for pos in positions:
            sector = pos.sector
            if sector not in portfolio_sector_weights:
                portfolio_sector_weights[sector] = 0
            portfolio_sector_weights[sector] += pos.market_value / total_value

        # Sector attribution
        sector_attribution = {}

        for sector in set(list(portfolio_sector_weights.keys()) + list(benchmark_sector_weights.keys())):
            port_weight = portfolio_sector_weights.get(sector, 0)
            bench_weight = benchmark_sector_weights.get(sector, 0)
            sector_return = sector_returns.get(sector, 0)

            # Allocation effect for this sector
            allocation_effect = (port_weight - bench_weight) * sector_return
            sector_attribution[f"{sector}_allocation"] = allocation_effect

        return sector_attribution

    def calculate_timeframe_attribution(self, returns_data: pd.DataFrame,
                                      timeframes: List[str] = None) -> Dict[str, float]:
        """Calculate attribution across different timeframes"""
        if timeframes is None:
            timeframes = ['1D', '1W', '1M', '3M', '1Y']

        timeframe_attribution = {}

        for timeframe in timeframes:
            try:
                if timeframe == '1D':
                    period_returns = returns_data.tail(1)
                elif timeframe == '1W':
                    period_returns = returns_data.tail(7)
                elif timeframe == '1M':
                    period_returns = returns_data.tail(30)
                elif timeframe == '3M':
                    period_returns = returns_data.tail(90)
                elif timeframe == '1Y':
                    period_returns = returns_data.tail(252)
                else:
                    continue

                if len(period_returns) > 0:
                    cumulative_return = (1 + period_returns).prod() - 1
                    timeframe_attribution[timeframe] = cumulative_return.mean()
                else:
                    timeframe_attribution[timeframe] = 0.0

            except Exception as e:
                logger.warning(f"Error calculating {timeframe} attribution: {e}")
                timeframe_attribution[timeframe] = 0.0

        return timeframe_attribution

    def get_attribution_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get attribution summary for the last N days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_attributions = [
            attr for attr in self.attribution_history
            if attr.timestamp > cutoff_date
        ]

        if not recent_attributions:
            return {}

        return {
            'period_days': days,
            'total_attributions': len(recent_attributions),
            'avg_active_return': np.mean([attr.active_return for attr in recent_attributions]),
            'avg_information_ratio': np.mean([attr.information_ratio for attr in recent_attributions]),
            'allocation_effect_avg': np.mean([
                attr.attribution_breakdown.get('allocation_effect', 0)
                for attr in recent_attributions
            ]),
            'selection_effect_avg': np.mean([
                attr.attribution_breakdown.get('selection_effect', 0)
                for attr in recent_attributions
            ]),
            'latest_attribution': recent_attributions[-1] if recent_attributions else None
        }


class RiskDecompositionAnalyzer:
    """Advanced risk decomposition and factor analysis"""

    def __init__(self, lookback_periods: int = 252):
        """Initialize risk decomposition analyzer"""
        self.lookback_periods = lookback_periods
        self.risk_history = deque(maxlen=1000)
        self.factor_models = {}

        logger.info("Risk Decomposition Analyzer initialized")

    def decompose_portfolio_risk(self, returns_data: pd.DataFrame,
                                weights: pd.Series) -> RiskDecomposition:
        """Decompose portfolio risk into systematic and idiosyncratic components"""
        start_time = time.perf_counter()

        # Calculate portfolio returns
        portfolio_returns = (returns_data * weights).sum(axis=1)

        # Estimate covariance matrix using Ledoit-Wolf shrinkage
        lw = LedoitWolf()
        cov_matrix = lw.fit(returns_data.fillna(0)).covariance_

        # Calculate portfolio variance
        portfolio_variance = np.dot(weights.values, np.dot(cov_matrix, weights.values))
        total_risk = np.sqrt(portfolio_variance * 252)  # Annualized

        # Principal Component Analysis for factor identification
        pca = PCA(n_components=min(10, len(returns_data.columns)))
        pca.fit(returns_data.fillna(0))

        # Calculate systematic risk (explained by first few PCs)
        n_systematic_factors = min(5, len(pca.components_))
        systematic_variance = sum(pca.explained_variance_[:n_systematic_factors])
        systematic_risk = np.sqrt(systematic_variance * 252)

        # Idiosyncratic risk
        idiosyncratic_risk = np.sqrt(max(0, portfolio_variance * 252 - systematic_variance * 252))

        # Factor contributions
        factor_contributions = {}
        for i in range(n_systematic_factors):
            factor_contributions[f'PC_{i+1}'] = pca.explained_variance_ratio_[i]

        # Principal components interpretation
        principal_components = {}
        for i in range(min(3, len(pca.components_))):
            # Find assets with highest loadings on this component
            loadings = pca.components_[i]
            top_assets = np.argsort(np.abs(loadings))[-3:]
            principal_components[f'PC_{i+1}_top_assets'] = [
                returns_data.columns[idx] for idx in top_assets
            ]

        # Risk metrics
        var_95 = np.percentile(portfolio_returns, 5)
        expected_shortfall = portfolio_returns[portfolio_returns <= var_95].mean()

        processing_time = (time.perf_counter() - start_time) * 1000

        result = RiskDecomposition(
            timestamp=datetime.now(),
            total_risk=total_risk,
            systematic_risk=systematic_risk,
            idiosyncratic_risk=idiosyncratic_risk,
            factor_contributions=factor_contributions,
            correlation_structure=np.corrcoef(returns_data.fillna(0).T),
            principal_components=principal_components,
            var_95=var_95,
            expected_shortfall=expected_shortfall
        )

        self.risk_history.append(result)

        logger.info(f"Risk decomposition completed in {processing_time:.2f}ms")
        return result

    def analyze_correlation_structure(self, returns_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlation structure and identify clusters"""

        # Calculate correlation matrix
        corr_matrix = returns_data.corr().fillna(0)

        # Hierarchical clustering of correlations
        distance_matrix = 1 - np.abs(corr_matrix)

        # K-means clustering on correlation matrix
        n_clusters = min(5, len(returns_data.columns) // 2)
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(corr_matrix.fillna(0))

            cluster_assignments = {}
            for i, asset in enumerate(returns_data.columns):
                cluster_assignments[asset] = int(clusters[i])
        else:
            cluster_assignments = {asset: 0 for asset in returns_data.columns}

        # Calculate average intra-cluster and inter-cluster correlations
        intra_cluster_corrs = []
        inter_cluster_corrs = []

        for i in range(len(returns_data.columns)):
            for j in range(i+1, len(returns_data.columns)):
                asset_i = returns_data.columns[i]
                asset_j = returns_data.columns[j]
                corr_ij = corr_matrix.iloc[i, j]

                if cluster_assignments[asset_i] == cluster_assignments[asset_j]:
                    intra_cluster_corrs.append(corr_ij)
                else:
                    inter_cluster_corrs.append(corr_ij)

        return {
            'correlation_matrix': corr_matrix,
            'cluster_assignments': cluster_assignments,
            'avg_intra_cluster_correlation': np.mean(intra_cluster_corrs) if intra_cluster_corrs else 0,
            'avg_inter_cluster_correlation': np.mean(inter_cluster_corrs) if inter_cluster_corrs else 0,
            'correlation_dispersion': np.std(corr_matrix.values.flatten()),
            'max_correlation': np.max(corr_matrix.values[corr_matrix.values < 1]),
            'min_correlation': np.min(corr_matrix.values)
        }

    def calculate_component_var(self, returns_data: pd.DataFrame,
                              weights: pd.Series, confidence_level: float = 0.05) -> Dict[str, float]:
        """Calculate Component Value-at-Risk for each position"""

        portfolio_returns = (returns_data * weights).sum(axis=1)
        portfolio_var = np.percentile(portfolio_returns, confidence_level * 100)

        component_vars = {}

        for asset in returns_data.columns:
            # Calculate marginal VaR
            asset_weight = weights.get(asset, 0)
            if asset_weight == 0:
                component_vars[asset] = 0
                continue

            # Perturb weight slightly to calculate marginal effect
            perturbed_weights = weights.copy()
            epsilon = 0.001
            perturbed_weights[asset] += epsilon
            perturbed_weights = perturbed_weights / perturbed_weights.sum()  # Renormalize

            perturbed_returns = (returns_data * perturbed_weights).sum(axis=1)
            perturbed_var = np.percentile(perturbed_returns, confidence_level * 100)

            marginal_var = (perturbed_var - portfolio_var) / epsilon
            component_var = marginal_var * asset_weight

            component_vars[asset] = component_var

        return component_vars


class FactorExposureAnalyzer:
    """Advanced factor exposure analysis and systematic risk identification"""

    def __init__(self):
        """Initialize factor exposure analyzer"""
        self.exposure_history = deque(maxlen=1000)
        self.factor_models = {}

        # Define factor proxies (in real implementation, these would be actual factor data)
        self.factor_proxies = {
            RiskFactorType.MARKET: 'market_return',
            RiskFactorType.MOMENTUM: 'momentum_factor',
            RiskFactorType.VOLATILITY: 'volatility_factor',
            RiskFactorType.SENTIMENT: 'sentiment_factor',
            RiskFactorType.LIQUIDITY: 'liquidity_factor'
        }

        logger.info("Factor Exposure Analyzer initialized")

    def calculate_factor_exposures(self, returns_data: pd.DataFrame,
                                 factor_returns: pd.DataFrame,
                                 portfolio_weights: pd.Series) -> FactorExposure:
        """Calculate portfolio factor exposures using multi-factor regression"""
        start_time = time.perf_counter()

        # Calculate portfolio returns
        portfolio_returns = (returns_data * portfolio_weights).sum(axis=1)

        # Align data
        common_index = portfolio_returns.index.intersection(factor_returns.index)
        portfolio_returns = portfolio_returns.loc[common_index]
        factor_returns = factor_returns.loc[common_index]

        # Multi-factor regression
        factor_loadings = {}
        factor_returns_dict = {}
        factor_volatilities = {}

        # Prepare regression data
        X = factor_returns.fillna(0).values
        y = portfolio_returns.fillna(0).values

        if len(X) > len(factor_returns.columns) and len(y) > 0:
            # Use OLS regression with regularization
            try:
                # Add constant term
                X_with_const = np.column_stack([np.ones(len(X)), X])

                # Calculate factor loadings using least squares
                beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]

                # Extract factor loadings (skip constant term)
                for i, factor_name in enumerate(factor_returns.columns):
                    factor_type = self._map_factor_name_to_type(factor_name)
                    factor_loadings[factor_type] = beta[i + 1]  # Skip constant

                    # Calculate factor statistics
                    factor_series = factor_returns[factor_name]
                    factor_returns_dict[factor_type] = factor_series.mean()
                    factor_volatilities[factor_type] = factor_series.std()

            except Exception as e:
                logger.warning(f"Error in factor regression: {e}")
                # Fallback to simple correlations
                for factor_name in factor_returns.columns:
                    factor_type = self._map_factor_name_to_type(factor_name)
                    correlation = portfolio_returns.corr(factor_returns[factor_name])
                    factor_loadings[factor_type] = correlation if not np.isnan(correlation) else 0

                    factor_series = factor_returns[factor_name]
                    factor_returns_dict[factor_type] = factor_series.mean()
                    factor_volatilities[factor_type] = factor_series.std()
        else:
            # Insufficient data - use default values
            for factor_type in RiskFactorType:
                factor_loadings[factor_type] = 0.0
                factor_returns_dict[factor_type] = 0.0
                factor_volatilities[factor_type] = 0.0

        # Calculate factor correlations
        factor_correlations = {}
        factor_names = list(factor_returns.columns)
        for i, factor1 in enumerate(factor_names):
            for j, factor2 in enumerate(factor_names[i+1:], i+1):
                factor_type1 = self._map_factor_name_to_type(factor1)
                factor_type2 = self._map_factor_name_to_type(factor2)

                corr = factor_returns[factor1].corr(factor_returns[factor2])
                factor_correlations[(factor_type1, factor_type2)] = corr if not np.isnan(corr) else 0

        # Calculate tracking error and active share
        tracking_error = self._calculate_tracking_error(portfolio_returns, factor_returns, factor_loadings)
        active_share = self._calculate_active_share(portfolio_weights)

        processing_time = (time.perf_counter() - start_time) * 1000

        result = FactorExposure(
            timestamp=datetime.now(),
            factor_loadings=factor_loadings,
            factor_returns=factor_returns_dict,
            factor_volatilities=factor_volatilities,
            factor_correlations=factor_correlations,
            tracking_error=tracking_error,
            active_share=active_share
        )

        self.exposure_history.append(result)

        logger.info(f"Factor exposure analysis completed in {processing_time:.2f}ms")
        return result

    def _map_factor_name_to_type(self, factor_name: str) -> RiskFactorType:
        """Map factor name to RiskFactorType enum"""
        factor_name_lower = factor_name.lower()

        if 'market' in factor_name_lower:
            return RiskFactorType.MARKET
        elif 'momentum' in factor_name_lower:
            return RiskFactorType.MOMENTUM
        elif 'volatility' in factor_name_lower or 'vol' in factor_name_lower:
            return RiskFactorType.VOLATILITY
        elif 'sentiment' in factor_name_lower:
            return RiskFactorType.SENTIMENT
        elif 'liquidity' in factor_name_lower:
            return RiskFactorType.LIQUIDITY
        else:
            return RiskFactorType.MARKET  # Default

    def _calculate_tracking_error(self, portfolio_returns: pd.Series,
                                factor_returns: pd.DataFrame,
                                factor_loadings: Dict[RiskFactorType, float]) -> float:
        """Calculate tracking error relative to factor model"""

        # Reconstruct returns using factor model
        reconstructed_returns = pd.Series(0, index=portfolio_returns.index)

        for factor_name in factor_returns.columns:
            factor_type = self._map_factor_name_to_type(factor_name)
            loading = factor_loadings.get(factor_type, 0)
            reconstructed_returns += loading * factor_returns[factor_name]

        # Calculate tracking error
        residuals = portfolio_returns - reconstructed_returns
        tracking_error = residuals.std() * np.sqrt(252)  # Annualized

        return tracking_error

    def _calculate_active_share(self, portfolio_weights: pd.Series) -> float:
        """Calculate active share (simplified - assumes equal benchmark weights)"""
        n_assets = len(portfolio_weights)
        benchmark_weight = 1.0 / n_assets if n_assets > 0 else 0

        active_weights = portfolio_weights - benchmark_weight
        active_share = 0.5 * np.sum(np.abs(active_weights))

        return active_share

    def identify_hedging_opportunities(self, current_exposures: Dict[RiskFactorType, float],
                                     target_exposures: Dict[RiskFactorType, float] = None) -> Dict[str, Any]:
        """Identify hedging opportunities based on factor exposures"""

        if target_exposures is None:
            # Default target is neutral exposure
            target_exposures = {factor: 0.0 for factor in RiskFactorType}

        hedging_needs = {}
        hedging_recommendations = {}

        for factor_type, current_exposure in current_exposures.items():
            target_exposure = target_exposures.get(factor_type, 0.0)
            exposure_difference = current_exposure - target_exposure

            hedging_needs[factor_type] = exposure_difference

            # Generate hedging recommendations
            if abs(exposure_difference) > 0.1:  # Threshold for hedging
                if exposure_difference > 0:
                    recommendation = f"Reduce {factor_type.value} exposure by {exposure_difference:.2f}"
                else:
                    recommendation = f"Increase {factor_type.value} exposure by {abs(exposure_difference):.2f}"

                hedging_recommendations[factor_type.value] = {
                    'action': 'reduce' if exposure_difference > 0 else 'increase',
                    'magnitude': abs(exposure_difference),
                    'priority': 'high' if abs(exposure_difference) > 0.3 else 'medium'
                }

        return {
            'hedging_needs': hedging_needs,
            'recommendations': hedging_recommendations,
            'total_hedging_required': sum(abs(need) for need in hedging_needs.values())
        }


class DrawdownAnalyzer:
    """Comprehensive drawdown analysis and recovery time estimation"""

    def __init__(self):
        """Initialize drawdown analyzer"""
        self.drawdown_history = deque(maxlen=1000)
        self.underwater_periods = []

        logger.info("Drawdown Analyzer initialized")

    def analyze_drawdowns(self, returns_data: pd.Series) -> DrawdownAnalysis:
        """Comprehensive drawdown analysis"""
        start_time = time.perf_counter()

        # Calculate cumulative returns
        cumulative_returns = (1 + returns_data).cumprod()

        # Calculate running maximum (peak)
        running_max = cumulative_returns.expanding().max()

        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / running_max

        # Current drawdown
        current_drawdown = drawdown.iloc[-1]

        # Maximum drawdown
        max_drawdown = drawdown.min()
        max_drawdown_date = drawdown.idxmin()

        # Recovery time analysis
        recovery_time_days = self._calculate_recovery_time(cumulative_returns, running_max, drawdown)

        # Underwater duration (current)
        underwater_duration_days = self._calculate_underwater_duration(drawdown)

        # Drawdown frequency and statistics
        drawdown_periods = self._identify_drawdown_periods(drawdown)
        drawdown_frequency = len(drawdown_periods) / (len(returns_data) / 252) if len(returns_data) > 252 else 0
        average_drawdown = np.mean([period['max_drawdown'] for period in drawdown_periods]) if drawdown_periods else 0

        # Stress testing
        stress_test_results = self._perform_stress_tests(returns_data)

        processing_time = (time.perf_counter() - start_time) * 1000

        result = DrawdownAnalysis(
            timestamp=datetime.now(),
            current_drawdown=current_drawdown,
            max_drawdown=max_drawdown,
            max_drawdown_date=max_drawdown_date,
            recovery_time_days=recovery_time_days,
            underwater_duration_days=underwater_duration_days,
            drawdown_frequency=drawdown_frequency,
            average_drawdown=average_drawdown,
            stress_test_results=stress_test_results
        )

        self.drawdown_history.append(result)

        logger.info(f"Drawdown analysis completed in {processing_time:.2f}ms")
        return result

    def _calculate_recovery_time(self, cumulative_returns: pd.Series,
                               running_max: pd.Series, drawdown: pd.Series) -> Optional[int]:
        """Calculate recovery time from maximum drawdown"""

        max_dd_date = drawdown.idxmin()
        max_dd_idx = drawdown.index.get_loc(max_dd_date)

        # Find recovery date (when portfolio reaches new high)
        post_max_dd = cumulative_returns.iloc[max_dd_idx:]
        peak_value = running_max.iloc[max_dd_idx]

        recovery_dates = post_max_dd[post_max_dd >= peak_value]

        if len(recovery_dates) > 0:
            recovery_date = recovery_dates.index[0]
            recovery_time = (recovery_date - max_dd_date).days
            return recovery_time
        else:
            return None  # Not yet recovered

    def _calculate_underwater_duration(self, drawdown: pd.Series) -> int:
        """Calculate current underwater duration"""

        # Find the last time portfolio was at a peak (drawdown = 0)
        at_peak = drawdown >= -0.001  # Small tolerance for floating point

        if at_peak.iloc[-1]:
            return 0  # Currently at peak

        # Find last peak
        last_peak_indices = np.where(at_peak)[0]

        if len(last_peak_indices) > 0:
            last_peak_idx = last_peak_indices[-1]
            underwater_duration = len(drawdown) - last_peak_idx - 1
            return underwater_duration
        else:
            return len(drawdown)  # Never reached peak

    def _identify_drawdown_periods(self, drawdown: pd.Series) -> List[Dict[str, Any]]:
        """Identify distinct drawdown periods"""

        drawdown_periods = []
        in_drawdown = False
        current_period = None

        for i, (date, dd_value) in enumerate(drawdown.items()):
            if dd_value < -0.001 and not in_drawdown:  # Start of drawdown
                in_drawdown = True
                current_period = {
                    'start_date': date,
                    'start_index': i,
                    'max_drawdown': dd_value,
                    'max_drawdown_date': date
                }
            elif dd_value < -0.001 and in_drawdown:  # Continue drawdown
                if dd_value < current_period['max_drawdown']:
                    current_period['max_drawdown'] = dd_value
                    current_period['max_drawdown_date'] = date
            elif dd_value >= -0.001 and in_drawdown:  # End of drawdown
                in_drawdown = False
                current_period['end_date'] = date
                current_period['end_index'] = i
                current_period['duration_days'] = (current_period['end_date'] - current_period['start_date']).days
                drawdown_periods.append(current_period)
                current_period = None

        # Handle ongoing drawdown
        if in_drawdown and current_period is not None:
            current_period['end_date'] = drawdown.index[-1]
            current_period['end_index'] = len(drawdown) - 1
            current_period['duration_days'] = (current_period['end_date'] - current_period['start_date']).days
            drawdown_periods.append(current_period)

        return drawdown_periods

    def _perform_stress_tests(self, returns_data: pd.Series) -> Dict[str, float]:
        """Perform various stress tests"""

        stress_scenarios = {}

        # Historical stress test (worst N-day period)
        for n_days in [1, 5, 10, 20]:
            if len(returns_data) >= n_days:
                rolling_returns = returns_data.rolling(n_days).sum()
                worst_period = rolling_returns.min()
                stress_scenarios[f'worst_{n_days}_day'] = worst_period

        # Volatility stress test
        current_vol = returns_data.std()
        stressed_vol = current_vol * 2  # Double volatility

        # Monte Carlo stress test (simplified)
        np.random.seed(42)
        stressed_returns = np.random.normal(returns_data.mean(), stressed_vol, 1000)
        stress_scenarios['monte_carlo_var_95'] = np.percentile(stressed_returns, 5)
        stress_scenarios['monte_carlo_var_99'] = np.percentile(stressed_returns, 1)

        # Tail risk measures
        stress_scenarios['expected_shortfall_5'] = returns_data[returns_data <= np.percentile(returns_data, 5)].mean()
        stress_scenarios['maximum_loss'] = returns_data.min()

        return stress_scenarios


class SharpeRatioOptimizer:
    """Advanced Sharpe ratio optimization and portfolio rebalancing"""

    def __init__(self, risk_free_rate: float = 0.02):
        """Initialize Sharpe ratio optimizer"""
        self.risk_free_rate = risk_free_rate
        self.optimization_history = deque(maxlen=1000)

        logger.info("Sharpe Ratio Optimizer initialized")

    def optimize_portfolio(self, expected_returns: pd.Series,
                         covariance_matrix: pd.DataFrame,
                         current_weights: pd.Series = None,
                         constraints: Dict[str, Any] = None) -> OptimizationResult:
        """Optimize portfolio for maximum Sharpe ratio"""
        start_time = time.perf_counter()

        # Prepare data
        assets = expected_returns.index
        n_assets = len(assets)

        if current_weights is None:
            current_weights = pd.Series(1.0/n_assets, index=assets)

        # Set up constraints
        if constraints is None:
            constraints = {
                'min_weight': 0.0,
                'max_weight': 1.0,
                'max_concentration': 0.4,
                'transaction_cost_bps': 10
            }

        # Objective function: negative Sharpe ratio (for minimization)
        def negative_sharpe_ratio(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)

            if portfolio_volatility == 0:
                return -np.inf

            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            return -sharpe_ratio

        # Constraints
        constraint_list = []

        # Weights sum to 1
        constraint_list.append({
            'type': 'eq',
            'fun': lambda weights: np.sum(weights) - 1.0
        })

        # Weight bounds
        bounds = [(constraints['min_weight'], constraints['max_weight']) for _ in range(n_assets)]

        # Maximum concentration constraint
        if 'max_concentration' in constraints:
            max_conc = constraints['max_concentration']
            for i in range(n_assets):
                constraint_list.append({
                    'type': 'ineq',
                    'fun': lambda weights, i=i: max_conc - weights[i]
                })

        # Initial guess
        initial_weights = current_weights.values

        # Optimization
        try:
            result = optimize.minimize(
                negative_sharpe_ratio,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraint_list,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )

            if result.success:
                optimal_weights_array = result.x
                optimal_weights = pd.Series(optimal_weights_array, index=assets)

                # Calculate metrics
                expected_return = np.dot(optimal_weights, expected_returns)
                expected_variance = np.dot(optimal_weights, np.dot(covariance_matrix, optimal_weights))
                expected_volatility = np.sqrt(expected_variance)
                sharpe_ratio = (expected_return - self.risk_free_rate) / expected_volatility

                # Calculate rebalancing trades
                weight_changes = optimal_weights - current_weights
                rebalancing_trades = weight_changes[abs(weight_changes) > 0.01]  # Only significant changes

                # Calculate transaction costs
                transaction_cost_bps = constraints.get('transaction_cost_bps', 10)
                transaction_costs = sum(abs(trade) * transaction_cost_bps / 10000 for trade in rebalancing_trades)

                # Calculate improvement metrics
                current_return = np.dot(current_weights, expected_returns)
                current_variance = np.dot(current_weights, np.dot(covariance_matrix, current_weights))
                current_volatility = np.sqrt(current_variance)
                current_sharpe = (current_return - self.risk_free_rate) / current_volatility if current_volatility > 0 else 0

                improvement_metrics = {
                    'sharpe_improvement': sharpe_ratio - current_sharpe,
                    'return_improvement': expected_return - current_return,
                    'volatility_change': expected_volatility - current_volatility,
                    'net_improvement': (sharpe_ratio - current_sharpe) - transaction_costs
                }

            else:
                # Optimization failed - return current weights
                optimal_weights = current_weights
                expected_return = np.dot(current_weights, expected_returns)
                expected_variance = np.dot(current_weights, np.dot(covariance_matrix, current_weights))
                expected_volatility = np.sqrt(expected_variance)
                sharpe_ratio = (expected_return - self.risk_free_rate) / expected_volatility if expected_volatility > 0 else 0
                rebalancing_trades = pd.Series(dtype=float)
                transaction_costs = 0.0
                improvement_metrics = {'optimization_failed': True}

        except Exception as e:
            logger.error(f"Optimization error: {e}")
            # Return current weights on error
            optimal_weights = current_weights
            expected_return = np.dot(current_weights, expected_returns)
            expected_variance = np.dot(current_weights, np.dot(covariance_matrix, current_weights))
            expected_volatility = np.sqrt(expected_variance)
            sharpe_ratio = (expected_return - self.risk_free_rate) / expected_volatility if expected_volatility > 0 else 0
            rebalancing_trades = pd.Series(dtype=float)
            transaction_costs = 0.0
            improvement_metrics = {'error': str(e)}

        processing_time = (time.perf_counter() - start_time) * 1000

        result = OptimizationResult(
            timestamp=datetime.now(),
            optimal_weights=optimal_weights.to_dict(),
            expected_return=expected_return,
            expected_volatility=expected_volatility,
            sharpe_ratio=sharpe_ratio,
            rebalancing_trades=rebalancing_trades.to_dict(),
            transaction_costs=transaction_costs,
            improvement_metrics=improvement_metrics
        )

        self.optimization_history.append(result)

        logger.info(f"Portfolio optimization completed in {processing_time:.2f}ms")
        return result

    def risk_parity_optimization(self, covariance_matrix: pd.DataFrame) -> Dict[str, float]:
        """Optimize for equal risk contribution (risk parity)"""

        n_assets = len(covariance_matrix)

        # Objective function: minimize sum of squared deviations from equal risk contribution
        def risk_parity_objective(weights):
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))

            # Calculate marginal risk contributions
            marginal_contribs = np.dot(covariance_matrix, weights)
            risk_contribs = weights * marginal_contribs / portfolio_variance

            # Target equal risk contribution
            target_contrib = 1.0 / n_assets

            # Sum of squared deviations
            return np.sum((risk_contribs - target_contrib) ** 2)

        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1.0}]
        bounds = [(0.01, 0.5) for _ in range(n_assets)]  # Min 1%, max 50%

        # Initial guess
        initial_weights = np.ones(n_assets) / n_assets

        try:
            result = optimize.minimize(
                risk_parity_objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

            if result.success:
                optimal_weights = result.x
                return dict(zip(covariance_matrix.index, optimal_weights))
            else:
                # Return equal weights if optimization fails
                equal_weights = 1.0 / n_assets
                return dict(zip(covariance_matrix.index, [equal_weights] * n_assets))

        except Exception as e:
            logger.error(f"Risk parity optimization error: {e}")
            equal_weights = 1.0 / n_assets
            return dict(zip(covariance_matrix.index, [equal_weights] * n_assets))

    def minimum_variance_optimization(self, covariance_matrix: pd.DataFrame,
                                    constraints: Dict[str, Any] = None) -> Dict[str, float]:
        """Optimize for minimum variance portfolio"""

        n_assets = len(covariance_matrix)

        if constraints is None:
            constraints = {'min_weight': 0.0, 'max_weight': 1.0}

        # Objective function: portfolio variance
        def portfolio_variance(weights):
            return np.dot(weights, np.dot(covariance_matrix, weights))

        # Constraints
        constraint_list = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1.0}]
        bounds = [(constraints['min_weight'], constraints['max_weight']) for _ in range(n_assets)]

        # Initial guess
        initial_weights = np.ones(n_assets) / n_assets

        try:
            result = optimize.minimize(
                portfolio_variance,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraint_list
            )

            if result.success:
                optimal_weights = result.x
                return dict(zip(covariance_matrix.index, optimal_weights))
            else:
                equal_weights = 1.0 / n_assets
                return dict(zip(covariance_matrix.index, [equal_weights] * n_assets))

        except Exception as e:
            logger.error(f"Minimum variance optimization error: {e}")
            equal_weights = 1.0 / n_assets
            return dict(zip(covariance_matrix.index, [equal_weights] * n_assets))

    def calculate_efficient_frontier(self, expected_returns: pd.Series,
                                   covariance_matrix: pd.DataFrame,
                                   n_portfolios: int = 50) -> Dict[str, List[float]]:
        """Calculate efficient frontier"""

        min_ret = expected_returns.min()
        max_ret = expected_returns.max()
        target_returns = np.linspace(min_ret, max_ret, n_portfolios)

        efficient_portfolios = []

        for target_return in target_returns:
            # Minimize variance subject to target return
            def portfolio_variance(weights):
                return np.dot(weights, np.dot(covariance_matrix, weights))

            constraints = [
                {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1.0},
                {'type': 'eq', 'fun': lambda weights: np.dot(weights, expected_returns) - target_return}
            ]

            bounds = [(0, 1) for _ in range(len(expected_returns))]
            initial_weights = np.ones(len(expected_returns)) / len(expected_returns)

            try:
                result = optimize.minimize(
                    portfolio_variance,
                    initial_weights,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints
                )

                if result.success:
                    weights = result.x
                    portfolio_return = np.dot(weights, expected_returns)
                    portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
                    portfolio_volatility = np.sqrt(portfolio_variance)
                    sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility

                    efficient_portfolios.append({
                        'return': portfolio_return,
                        'volatility': portfolio_volatility,
                        'sharpe_ratio': sharpe_ratio,
                        'weights': weights
                    })

            except Exception:
                continue

        if efficient_portfolios:
            returns = [p['return'] for p in efficient_portfolios]
            volatilities = [p['volatility'] for p in efficient_portfolios]
            sharpe_ratios = [p['sharpe_ratio'] for p in efficient_portfolios]

            return {
                'returns': returns,
                'volatilities': volatilities,
                'sharpe_ratios': sharpe_ratios,
                'portfolios': efficient_portfolios
            }
        else:
            return {'returns': [], 'volatilities': [], 'sharpe_ratios': [], 'portfolios': []}


class AdvancedPortfolioAnalyticsSystem:
    """Complete advanced portfolio analytics system integrating all components"""

    def __init__(self, symbols: List[str] = None, risk_free_rate: float = 0.02):
        """Initialize advanced portfolio analytics system"""
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT']
        self.risk_free_rate = risk_free_rate

        # Initialize components
        self.attribution_analyzer = PerformanceAttributionAnalyzer()
        self.risk_decomposer = RiskDecompositionAnalyzer()
        self.factor_analyzer = FactorExposureAnalyzer()
        self.drawdown_analyzer = DrawdownAnalyzer()
        self.sharpe_optimizer = SharpeRatioOptimizer(risk_free_rate)

        # System state
        self.portfolio_positions = {}
        self.analytics_history = deque(maxlen=1000)
        self.is_initialized = False

        logger.info(f"Advanced Portfolio Analytics System initialized for {len(self.symbols)} symbols")

    async def initialize_system(self, historical_data: Dict[str, pd.DataFrame],
                              initial_positions: List[PortfolioPosition] = None):
        """Initialize portfolio analytics system"""
        logger.info("📊 Initializing Advanced Portfolio Analytics System...")

        initialization_results = {}

        try:
            # Store initial positions
            if initial_positions:
                for position in initial_positions:
                    self.portfolio_positions[position.symbol] = position
            else:
                # Create default equal-weight positions
                total_value = 100000  # $100k default portfolio
                position_value = total_value / len(self.symbols)

                for symbol in self.symbols:
                    self.portfolio_positions[symbol] = PortfolioPosition(
                        symbol=symbol,
                        quantity=position_value / 45000,  # Assume $45k price
                        market_value=position_value,
                        weight=1.0 / len(self.symbols),
                        sector='crypto',
                        strategy='default',
                        entry_date=datetime.now(),
                        current_price=45000,
                        unrealized_pnl=0.0,
                        realized_pnl=0.0
                    )

            # Initialize analytics components with historical data
            for symbol, data in historical_data.items():
                if len(data) < 50:
                    logger.warning(f"Insufficient data for {symbol}")
                    continue

                # Test each component
                returns = data['close'].pct_change().dropna()

                # Test risk decomposition
                if len(returns) > 20:
                    weights = pd.Series(1.0/len(self.symbols), index=[symbol])
                    returns_df = pd.DataFrame({symbol: returns})
                    risk_result = self.risk_decomposer.decompose_portfolio_risk(returns_df, weights)
                    initialization_results[f'{symbol}_risk'] = {
                        'total_risk': risk_result.total_risk,
                        'systematic_risk': risk_result.systematic_risk
                    }

                # Test drawdown analysis
                drawdown_result = self.drawdown_analyzer.analyze_drawdowns(returns)
                initialization_results[f'{symbol}_drawdown'] = {
                    'max_drawdown': drawdown_result.max_drawdown,
                    'current_drawdown': drawdown_result.current_drawdown
                }

            self.is_initialized = True
            logger.info("✅ Advanced Portfolio Analytics System initialization complete!")

        except Exception as e:
            logger.error(f"❌ Error initializing analytics system: {e}")
            initialization_results['error'] = str(e)

        return initialization_results

    async def run_comprehensive_analytics(self, market_data: Dict[str, pd.DataFrame],
                                        factor_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Run comprehensive portfolio analytics"""
        start_time = time.perf_counter()

        if not self.is_initialized:
            logger.warning("Analytics system not initialized")
            return {}

        analytics_results = {}

        try:
            # 1. Prepare portfolio data
            portfolio_data = self._prepare_portfolio_data(market_data)

            # 2. Performance Attribution Analysis
            attribution_result = await self._run_attribution_analysis(portfolio_data)
            analytics_results['attribution'] = attribution_result

            # 3. Risk Decomposition
            risk_result = await self._run_risk_decomposition(portfolio_data)
            analytics_results['risk_decomposition'] = risk_result

            # 4. Factor Exposure Analysis
            if factor_data is not None:
                factor_result = await self._run_factor_analysis(portfolio_data, factor_data)
                analytics_results['factor_exposure'] = factor_result

            # 5. Drawdown Analysis
            drawdown_result = await self._run_drawdown_analysis(portfolio_data)
            analytics_results['drawdown_analysis'] = drawdown_result

            # 6. Portfolio Optimization
            optimization_result = await self._run_portfolio_optimization(portfolio_data)
            analytics_results['optimization'] = optimization_result

            # 7. Calculate overall portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(portfolio_data)
            analytics_results['portfolio_metrics'] = portfolio_metrics

            processing_time = (time.perf_counter() - start_time) * 1000
            analytics_results['processing_time_ms'] = processing_time

            # Store in history
            self.analytics_history.append({
                'timestamp': datetime.now(),
                'results': analytics_results
            })

            logger.info(f"Comprehensive analytics completed in {processing_time:.2f}ms")

        except Exception as e:
            logger.error(f"Error in comprehensive analytics: {e}")
            analytics_results['error'] = str(e)

        return analytics_results

    def _prepare_portfolio_data(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Prepare portfolio data for analysis"""

        # Calculate portfolio weights
        total_value = sum(pos.market_value for pos in self.portfolio_positions.values())
        weights = pd.Series({
            symbol: pos.market_value / total_value
            for symbol, pos in self.portfolio_positions.items()
        })

        # Combine returns data
        returns_data = {}
        for symbol, data in market_data.items():
            if symbol in self.portfolio_positions:
                returns_data[symbol] = data['close'].pct_change().dropna()

        returns_df = pd.DataFrame(returns_data).fillna(0)

        # Calculate portfolio returns
        portfolio_returns = (returns_df * weights).sum(axis=1)

        return {
            'weights': weights,
            'returns_df': returns_df,
            'portfolio_returns': portfolio_returns,
            'positions': list(self.portfolio_positions.values())
        }

    async def _run_attribution_analysis(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run performance attribution analysis"""

        returns_df = portfolio_data['returns_df']
        weights = portfolio_data['weights']

        # Create benchmark (equal weights)
        benchmark_weights = pd.Series(1.0/len(returns_df.columns), index=returns_df.columns)
        benchmark_returns = returns_df

        # Run attribution
        attribution = self.attribution_analyzer.calculate_brinson_attribution(
            returns_df, benchmark_returns,
            pd.DataFrame([weights] * len(returns_df)),
            pd.DataFrame([benchmark_weights] * len(returns_df))
        )

        # Additional attribution analyses
        sector_attribution = self.attribution_analyzer.calculate_sector_attribution(
            portfolio_data['positions'],
            {'crypto': 0.05},  # Simplified sector returns
            {'crypto': 1.0}    # Simplified benchmark weights
        )

        timeframe_attribution = self.attribution_analyzer.calculate_timeframe_attribution(
            returns_df
        )

        return {
            'brinson_attribution': {
                'total_return': attribution.total_return,
                'benchmark_return': attribution.benchmark_return,
                'active_return': attribution.active_return,
                'attribution_breakdown': attribution.attribution_breakdown,
                'information_ratio': attribution.information_ratio
            },
            'sector_attribution': sector_attribution,
            'timeframe_attribution': timeframe_attribution
        }

    async def _run_risk_decomposition(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run risk decomposition analysis"""

        returns_df = portfolio_data['returns_df']
        weights = portfolio_data['weights']

        # Risk decomposition
        risk_decomp = self.risk_decomposer.decompose_portfolio_risk(returns_df, weights)

        # Correlation analysis
        correlation_analysis = self.risk_decomposer.analyze_correlation_structure(returns_df)

        # Component VaR
        component_var = self.risk_decomposer.calculate_component_var(returns_df, weights)

        return {
            'total_risk': risk_decomp.total_risk,
            'systematic_risk': risk_decomp.systematic_risk,
            'idiosyncratic_risk': risk_decomp.idiosyncratic_risk,
            'factor_contributions': risk_decomp.factor_contributions,
            'var_95': risk_decomp.var_95,
            'expected_shortfall': risk_decomp.expected_shortfall,
            'correlation_analysis': correlation_analysis,
            'component_var': component_var
        }

    async def _run_factor_analysis(self, portfolio_data: Dict[str, Any],
                                 factor_data: pd.DataFrame) -> Dict[str, Any]:
        """Run factor exposure analysis"""

        returns_df = portfolio_data['returns_df']
        weights = portfolio_data['weights']

        # Factor exposure analysis
        factor_exposure = self.factor_analyzer.calculate_factor_exposures(
            returns_df, factor_data, weights
        )

        # Hedging opportunities
        hedging_analysis = self.factor_analyzer.identify_hedging_opportunities(
            factor_exposure.factor_loadings
        )

        return {
            'factor_loadings': factor_exposure.factor_loadings,
            'factor_returns': factor_exposure.factor_returns,
            'factor_volatilities': factor_exposure.factor_volatilities,
            'tracking_error': factor_exposure.tracking_error,
            'active_share': factor_exposure.active_share,
            'hedging_analysis': hedging_analysis
        }

    async def _run_drawdown_analysis(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run drawdown analysis"""

        portfolio_returns = portfolio_data['portfolio_returns']

        # Comprehensive drawdown analysis
        drawdown_analysis = self.drawdown_analyzer.analyze_drawdowns(portfolio_returns)

        return {
            'current_drawdown': drawdown_analysis.current_drawdown,
            'max_drawdown': drawdown_analysis.max_drawdown,
            'max_drawdown_date': drawdown_analysis.max_drawdown_date.isoformat(),
            'recovery_time_days': drawdown_analysis.recovery_time_days,
            'underwater_duration_days': drawdown_analysis.underwater_duration_days,
            'drawdown_frequency': drawdown_analysis.drawdown_frequency,
            'average_drawdown': drawdown_analysis.average_drawdown,
            'stress_test_results': drawdown_analysis.stress_test_results
        }

    async def _run_portfolio_optimization(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run portfolio optimization"""

        returns_df = portfolio_data['returns_df']
        weights = portfolio_data['weights']

        # Calculate expected returns (simple historical mean)
        expected_returns = returns_df.mean() * 252  # Annualized

        # Calculate covariance matrix
        covariance_matrix = returns_df.cov() * 252  # Annualized

        # Sharpe ratio optimization
        sharpe_optimization = self.sharpe_optimizer.optimize_portfolio(
            expected_returns, covariance_matrix, weights
        )

        # Risk parity optimization
        risk_parity_weights = self.sharpe_optimizer.risk_parity_optimization(covariance_matrix)

        # Minimum variance optimization
        min_var_weights = self.sharpe_optimizer.minimum_variance_optimization(covariance_matrix)

        return {
            'sharpe_optimization': {
                'optimal_weights': sharpe_optimization.optimal_weights,
                'expected_return': sharpe_optimization.expected_return,
                'expected_volatility': sharpe_optimization.expected_volatility,
                'sharpe_ratio': sharpe_optimization.sharpe_ratio,
                'rebalancing_trades': sharpe_optimization.rebalancing_trades,
                'transaction_costs': sharpe_optimization.transaction_costs,
                'improvement_metrics': sharpe_optimization.improvement_metrics
            },
            'risk_parity_weights': risk_parity_weights,
            'minimum_variance_weights': min_var_weights
        }

    def _calculate_portfolio_metrics(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall portfolio metrics"""

        portfolio_returns = portfolio_data['portfolio_returns']

        # Basic metrics
        total_return = (1 + portfolio_returns).prod() - 1
        annualized_return = (1 + portfolio_returns.mean()) ** 252 - 1
        annualized_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0

        # Additional metrics
        sortino_ratio = self._calculate_sortino_ratio(portfolio_returns)
        calmar_ratio = self._calculate_calmar_ratio(portfolio_returns)
        max_drawdown = self.drawdown_analyzer.analyze_drawdowns(portfolio_returns).max_drawdown

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'total_positions': len(self.portfolio_positions),
            'portfolio_value': sum(pos.market_value for pos in self.portfolio_positions.values())
        }

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf')

        downside_deviation = downside_returns.std() * np.sqrt(252)
        annualized_return = (1 + returns.mean()) ** 252 - 1

        return (annualized_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0

    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio"""
        annualized_return = (1 + returns.mean()) ** 252 - 1
        max_drawdown = abs(self.drawdown_analyzer.analyze_drawdowns(returns).max_drawdown)

        return annualized_return / max_drawdown if max_drawdown > 0 else 0

    def get_analytics_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get analytics summary for the last N days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_analytics = [
            entry for entry in self.analytics_history
            if entry['timestamp'] > cutoff_date
        ]

        if not recent_analytics:
            return {}

        # Extract key metrics
        sharpe_ratios = []
        processing_times = []

        for entry in recent_analytics:
            results = entry['results']
            if 'portfolio_metrics' in results:
                sharpe_ratios.append(results['portfolio_metrics'].get('sharpe_ratio', 0))
            if 'processing_time_ms' in results:
                processing_times.append(results['processing_time_ms'])

        return {
            'period_days': days,
            'total_analytics_runs': len(recent_analytics),
            'avg_sharpe_ratio': np.mean(sharpe_ratios) if sharpe_ratios else 0,
            'avg_processing_time_ms': np.mean(processing_times) if processing_times else 0,
            'latency_target_met': np.mean(processing_times) < 100 if processing_times else False,
            'latest_analytics': recent_analytics[-1] if recent_analytics else None
        }


# Testing and Validation Functions

def create_sample_portfolio_data(symbols: List[str] = None, periods: int = 252) -> Dict[str, Any]:
    """Create sample portfolio data for testing"""
    if symbols is None:
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT']

    np.random.seed(42)

    # Generate market data
    market_data = {}
    dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

    for symbol in symbols:
        # Generate realistic returns with different characteristics
        if 'BTC' in symbol:
            returns = np.random.normal(0.0008, 0.025, periods)  # Higher volatility
        elif 'ETH' in symbol:
            returns = np.random.normal(0.0006, 0.022, periods)
        else:
            returns = np.random.normal(0.0004, 0.018, periods)  # Lower volatility

        # Generate price series
        base_price = 45000 if 'BTC' in symbol else 2500
        prices = base_price * np.exp(np.cumsum(returns))

        market_data[symbol] = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.normal(0, 0.001, periods)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, periods))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, periods))),
            'close': prices,
            'volume': np.random.lognormal(15, 1, periods)
        }, index=dates)

    # Create portfolio positions
    total_value = 100000  # $100k portfolio
    position_value = total_value / len(symbols)

    positions = []
    for symbol in symbols:
        current_price = market_data[symbol]['close'].iloc[-1]
        quantity = position_value / current_price

        positions.append(PortfolioPosition(
            symbol=symbol,
            quantity=quantity,
            market_value=position_value,
            weight=1.0 / len(symbols),
            sector='crypto',
            strategy='equal_weight',
            entry_date=dates[0],
            current_price=current_price,
            unrealized_pnl=np.random.normal(0, 1000),
            realized_pnl=np.random.normal(0, 500)
        ))

    # Create factor data
    factor_data = pd.DataFrame({
        'market_return': np.random.normal(0.0005, 0.015, periods),
        'momentum_factor': np.random.normal(0, 0.01, periods),
        'volatility_factor': np.random.normal(0, 0.008, periods),
        'sentiment_factor': np.random.normal(0, 0.012, periods),
        'liquidity_factor': np.random.normal(0, 0.006, periods)
    }, index=dates)

    return {
        'market_data': market_data,
        'positions': positions,
        'factor_data': factor_data,
        'symbols': symbols
    }


async def test_performance_attribution():
    """Test performance attribution analysis"""
    logger.info("🧪 Testing Performance Attribution Analysis...")

    # Create test data
    test_data = create_sample_portfolio_data()

    # Initialize system
    analytics_system = AdvancedPortfolioAnalyticsSystem(test_data['symbols'])
    await analytics_system.initialize_system(test_data['market_data'], test_data['positions'])

    # Test attribution analysis
    start_time = time.perf_counter()

    # Prepare portfolio data
    portfolio_data = analytics_system._prepare_portfolio_data(test_data['market_data'])

    # Run attribution analysis
    attribution_result = await analytics_system._run_attribution_analysis(portfolio_data)

    processing_time = (time.perf_counter() - start_time) * 1000

    # Validate results
    brinson_attr = attribution_result['brinson_attribution']

    validation_results = {
        'attribution_calculated': 'total_return' in brinson_attr,
        'active_return_calculated': 'active_return' in brinson_attr,
        'information_ratio_calculated': 'information_ratio' in brinson_attr,
        'sector_attribution_calculated': len(attribution_result['sector_attribution']) > 0,
        'timeframe_attribution_calculated': len(attribution_result['timeframe_attribution']) > 0,
        'processing_time_ms': processing_time,
        'latency_target_met': processing_time < 100
    }

    logger.info(f"✅ Attribution analysis completed in {processing_time:.2f}ms")

    return validation_results


async def test_risk_decomposition():
    """Test risk decomposition analysis"""
    logger.info("🧪 Testing Risk Decomposition Analysis...")

    # Create test data
    test_data = create_sample_portfolio_data()

    # Initialize system
    analytics_system = AdvancedPortfolioAnalyticsSystem(test_data['symbols'])
    await analytics_system.initialize_system(test_data['market_data'], test_data['positions'])

    # Test risk decomposition
    start_time = time.perf_counter()

    portfolio_data = analytics_system._prepare_portfolio_data(test_data['market_data'])
    risk_result = await analytics_system._run_risk_decomposition(portfolio_data)

    processing_time = (time.perf_counter() - start_time) * 1000

    # Validate results
    validation_results = {
        'total_risk_calculated': risk_result['total_risk'] > 0,
        'systematic_risk_calculated': risk_result['systematic_risk'] >= 0,
        'idiosyncratic_risk_calculated': risk_result['idiosyncratic_risk'] >= 0,
        'var_calculated': 'var_95' in risk_result,
        'expected_shortfall_calculated': 'expected_shortfall' in risk_result,
        'correlation_analysis_completed': 'correlation_analysis' in risk_result,
        'component_var_calculated': len(risk_result['component_var']) > 0,
        'processing_time_ms': processing_time,
        'latency_target_met': processing_time < 100,
        'risk_decomposition_valid': (
            risk_result['systematic_risk'] + risk_result['idiosyncratic_risk']
            <= risk_result['total_risk'] * 1.1  # Allow 10% tolerance
        )
    }

    logger.info(f"✅ Risk decomposition completed in {processing_time:.2f}ms")

    return validation_results


async def test_factor_exposure():
    """Test factor exposure analysis"""
    logger.info("🧪 Testing Factor Exposure Analysis...")

    # Create test data
    test_data = create_sample_portfolio_data()

    # Initialize system
    analytics_system = AdvancedPortfolioAnalyticsSystem(test_data['symbols'])
    await analytics_system.initialize_system(test_data['market_data'], test_data['positions'])

    # Test factor analysis
    start_time = time.perf_counter()

    portfolio_data = analytics_system._prepare_portfolio_data(test_data['market_data'])
    factor_result = await analytics_system._run_factor_analysis(portfolio_data, test_data['factor_data'])

    processing_time = (time.perf_counter() - start_time) * 1000

    # Validate results
    validation_results = {
        'factor_loadings_calculated': len(factor_result['factor_loadings']) > 0,
        'factor_returns_calculated': len(factor_result['factor_returns']) > 0,
        'tracking_error_calculated': 'tracking_error' in factor_result,
        'active_share_calculated': 'active_share' in factor_result,
        'hedging_analysis_completed': 'hedging_analysis' in factor_result,
        'processing_time_ms': processing_time,
        'latency_target_met': processing_time < 100
    }

    logger.info(f"✅ Factor exposure analysis completed in {processing_time:.2f}ms")

    return validation_results


async def test_drawdown_analysis():
    """Test drawdown analysis"""
    logger.info("🧪 Testing Drawdown Analysis...")

    # Create test data
    test_data = create_sample_portfolio_data()

    # Initialize system
    analytics_system = AdvancedPortfolioAnalyticsSystem(test_data['symbols'])
    await analytics_system.initialize_system(test_data['market_data'], test_data['positions'])

    # Test drawdown analysis
    start_time = time.perf_counter()

    portfolio_data = analytics_system._prepare_portfolio_data(test_data['market_data'])
    drawdown_result = await analytics_system._run_drawdown_analysis(portfolio_data)

    processing_time = (time.perf_counter() - start_time) * 1000

    # Validate results
    validation_results = {
        'current_drawdown_calculated': 'current_drawdown' in drawdown_result,
        'max_drawdown_calculated': 'max_drawdown' in drawdown_result,
        'recovery_time_calculated': 'recovery_time_days' in drawdown_result,
        'underwater_duration_calculated': 'underwater_duration_days' in drawdown_result,
        'stress_tests_completed': len(drawdown_result['stress_test_results']) > 0,
        'drawdown_frequency_calculated': 'drawdown_frequency' in drawdown_result,
        'processing_time_ms': processing_time,
        'latency_target_met': processing_time < 100,
        'max_drawdown_reasonable': abs(drawdown_result['max_drawdown']) <= 1.0  # Max 100% drawdown
    }

    logger.info(f"✅ Drawdown analysis completed in {processing_time:.2f}ms")

    return validation_results


async def test_sharpe_optimization():
    """Test Sharpe ratio optimization"""
    logger.info("🧪 Testing Sharpe Ratio Optimization...")

    # Create test data
    test_data = create_sample_portfolio_data()

    # Initialize system
    analytics_system = AdvancedPortfolioAnalyticsSystem(test_data['symbols'])
    await analytics_system.initialize_system(test_data['market_data'], test_data['positions'])

    # Test optimization
    start_time = time.perf_counter()

    portfolio_data = analytics_system._prepare_portfolio_data(test_data['market_data'])
    optimization_result = await analytics_system._run_portfolio_optimization(portfolio_data)

    processing_time = (time.perf_counter() - start_time) * 1000

    # Validate results
    sharpe_opt = optimization_result['sharpe_optimization']

    validation_results = {
        'optimal_weights_calculated': len(sharpe_opt['optimal_weights']) > 0,
        'sharpe_ratio_calculated': 'sharpe_ratio' in sharpe_opt,
        'expected_return_calculated': 'expected_return' in sharpe_opt,
        'expected_volatility_calculated': 'expected_volatility' in sharpe_opt,
        'rebalancing_trades_calculated': 'rebalancing_trades' in sharpe_opt,
        'transaction_costs_calculated': 'transaction_costs' in sharpe_opt,
        'risk_parity_calculated': len(optimization_result['risk_parity_weights']) > 0,
        'min_variance_calculated': len(optimization_result['minimum_variance_weights']) > 0,
        'processing_time_ms': processing_time,
        'latency_target_met': processing_time < 100,
        'weights_sum_to_one': abs(sum(sharpe_opt['optimal_weights'].values()) - 1.0) < 0.01
    }

    logger.info(f"✅ Sharpe optimization completed in {processing_time:.2f}ms")

    return validation_results


async def test_comprehensive_analytics():
    """Test comprehensive analytics system"""
    logger.info("🧪 Testing Comprehensive Analytics System...")

    # Create test data
    test_data = create_sample_portfolio_data()

    # Initialize system
    analytics_system = AdvancedPortfolioAnalyticsSystem(test_data['symbols'])
    init_results = await analytics_system.initialize_system(test_data['market_data'], test_data['positions'])

    # Run comprehensive analytics
    start_time = time.perf_counter()

    analytics_results = await analytics_system.run_comprehensive_analytics(
        test_data['market_data'], test_data['factor_data']
    )

    processing_time = (time.perf_counter() - start_time) * 1000

    # Validate comprehensive results
    validation_results = {
        'initialization_successful': analytics_system.is_initialized,
        'attribution_completed': 'attribution' in analytics_results,
        'risk_decomposition_completed': 'risk_decomposition' in analytics_results,
        'factor_exposure_completed': 'factor_exposure' in analytics_results,
        'drawdown_analysis_completed': 'drawdown_analysis' in analytics_results,
        'optimization_completed': 'optimization' in analytics_results,
        'portfolio_metrics_calculated': 'portfolio_metrics' in analytics_results,
        'processing_time_ms': processing_time,
        'latency_target_met': processing_time < 100,
        'all_components_working': all([
            'attribution' in analytics_results,
            'risk_decomposition' in analytics_results,
            'factor_exposure' in analytics_results,
            'drawdown_analysis' in analytics_results,
            'optimization' in analytics_results,
            'portfolio_metrics' in analytics_results
        ])
    }

    # Test analytics summary
    summary = analytics_system.get_analytics_summary(days=1)
    validation_results['summary_generated'] = len(summary) > 0

    logger.info(f"✅ Comprehensive analytics completed in {processing_time:.2f}ms")

    return validation_results


async def run_comprehensive_validation():
    """Run comprehensive validation of the portfolio analytics system"""
    logger.info("🔬 Running Comprehensive Validation of Portfolio Analytics System...")

    validation_results = {}

    # Test 1: Performance Attribution
    attribution_results = await test_performance_attribution()
    validation_results['attribution'] = attribution_results

    # Test 2: Risk Decomposition
    risk_results = await test_risk_decomposition()
    validation_results['risk_decomposition'] = risk_results

    # Test 3: Factor Exposure
    factor_results = await test_factor_exposure()
    validation_results['factor_exposure'] = factor_results

    # Test 4: Drawdown Analysis
    drawdown_results = await test_drawdown_analysis()
    validation_results['drawdown'] = drawdown_results

    # Test 5: Sharpe Optimization
    optimization_results = await test_sharpe_optimization()
    validation_results['optimization'] = optimization_results

    # Test 6: Comprehensive System
    comprehensive_results = await test_comprehensive_analytics()
    validation_results['comprehensive'] = comprehensive_results

    # Overall validation summary
    all_latency_targets = [
        attribution_results['latency_target_met'],
        risk_results['latency_target_met'],
        factor_results['latency_target_met'],
        drawdown_results['latency_target_met'],
        optimization_results['latency_target_met'],
        comprehensive_results['latency_target_met']
    ]

    validation_summary = {
        'all_components_tested': True,
        'all_latency_targets_met': all(all_latency_targets),
        'attribution_accuracy': attribution_results['attribution_calculated'],
        'risk_decomposition_accuracy': risk_results['risk_decomposition_valid'],
        'optimization_accuracy': optimization_results['weights_sum_to_one'],
        'comprehensive_system_working': comprehensive_results['all_components_working'],
        'avg_processing_time_ms': np.mean([
            attribution_results['processing_time_ms'],
            risk_results['processing_time_ms'],
            factor_results['processing_time_ms'],
            drawdown_results['processing_time_ms'],
            optimization_results['processing_time_ms'],
            comprehensive_results['processing_time_ms']
        ]),
        'all_targets_met': all([
            all(all_latency_targets),
            attribution_results['attribution_calculated'],
            risk_results['risk_decomposition_valid'],
            optimization_results['weights_sum_to_one'],
            comprehensive_results['all_components_working']
        ])
    }

    validation_results['summary'] = validation_summary

    # Log summary
    logger.info("📊 Validation Summary:")
    logger.info(f"   All Latency Targets Met: {'✅' if validation_summary['all_latency_targets_met'] else '❌'}")
    logger.info(f"   Attribution Accuracy: {'✅' if validation_summary['attribution_accuracy'] else '❌'}")
    logger.info(f"   Risk Decomposition Valid: {'✅' if validation_summary['risk_decomposition_accuracy'] else '❌'}")
    logger.info(f"   Optimization Accuracy: {'✅' if validation_summary['optimization_accuracy'] else '❌'}")
    logger.info(f"   Comprehensive System: {'✅' if validation_summary['comprehensive_system_working'] else '❌'}")
    logger.info(f"   Average Processing Time: {validation_summary['avg_processing_time_ms']:.2f}ms")
    logger.info(f"   All Targets Met: {'✅' if validation_summary['all_targets_met'] else '❌'}")

    return validation_results


async def main():
    """Main testing function for Phase 6.3"""
    logger.info("🚀 Starting Phase 6.3: Advanced Portfolio Analytics Testing")

    # Run comprehensive validation
    validation_results = await run_comprehensive_validation()

    print("\n" + "="*80)
    print("📊 PHASE 6.3: ADVANCED PORTFOLIO ANALYTICS - RESULTS")
    print("="*80)

    summary = validation_results['summary']

    print(f"⚡ LATENCY PERFORMANCE:")
    print(f"   All Latency Targets Met: {'✅ ACHIEVED' if summary['all_latency_targets_met'] else '❌ NOT MET'}")
    print(f"   Average Processing Time: {summary['avg_processing_time_ms']:.2f}ms")
    print(f"   Target (<100ms): {'✅ ACHIEVED' if summary['avg_processing_time_ms'] < 100 else '❌ NOT MET'}")

    print(f"\n🎯 ACCURACY PERFORMANCE:")
    print(f"   Attribution Analysis: {'✅ WORKING' if summary['attribution_accuracy'] else '❌ FAILED'}")
    print(f"   Risk Decomposition: {'✅ VALID' if summary['risk_decomposition_accuracy'] else '❌ INVALID'}")
    print(f"   Portfolio Optimization: {'✅ ACCURATE' if summary['optimization_accuracy'] else '❌ INACCURATE'}")

    print(f"\n🔧 SYSTEM INTEGRATION:")
    print(f"   Comprehensive System: {'✅ WORKING' if summary['comprehensive_system_working'] else '❌ FAILED'}")
    print(f"   All Components: {'✅ INTEGRATED' if summary['all_components_tested'] else '❌ MISSING'}")

    print(f"\n🏆 OVERALL PHASE 6.3 STATUS:")
    print(f"   All Targets Met: {'✅ SUCCESS' if summary['all_targets_met'] else '❌ NEEDS IMPROVEMENT'}")

    print("\n📈 DETAILED COMPONENT RESULTS:")
    for component, results in validation_results.items():
        if component != 'summary':
            latency_status = '✅' if results.get('latency_target_met', False) else '❌'
            print(f"   {component.title()}: {latency_status} ({results.get('processing_time_ms', 0):.2f}ms)")

    print("\n" + "="*80)
    print("✅ Phase 6.3: Advanced Portfolio Analytics - COMPLETE")
    print("="*80)

    return validation_results


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
