#!/usr/bin/env python3
"""
Enhanced Trading Predictions API

This module provides comprehensive trading predictions by integrating:
- ML model predictions with confidence scoring
- Smart Money Concepts (SMC) analysis
- Multi-timeframe confluence analysis
- Risk-adjusted signal generation
- Trading-specific directional predictions

Key Features:
- Enhanced ML predictions with SMC integration
- Multi-timeframe analysis and confluence scoring
- Confidence-based risk management
- Trading signal generation with entry/exit levels
- Real-time market structure analysis
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
import ta

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

# Import ML components
from ml.src.api.model_service import ModelService, get_model_service
from ml.src.models import ModelFactory

# Import SMC components
try:
    from ml.backend.src.strategy.smc_detection import SMCDector
    from ml.backend.src.strategy.multi_timeframe_confluence import (
        MultiTimeframeAnalyzer, TimeframeType, create_sample_multi_timeframe_data,
        get_enhanced_multi_timeframe_analysis
    )
    SMC_AVAILABLE = True
except ImportError as e:
    logging.warning(f"SMC components not available: {e}")
    SMC_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create API router
router = APIRouter()


class TradingPredictionInput(BaseModel):
    """Enhanced input for trading predictions"""
    symbol: str = Field(..., description="Trading symbol (e.g., 'BTC/USDT')")
    timeframe: str = Field("15m", description="Primary timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d)")
    ohlcv_data: Optional[List[Dict[str, float]]] = Field(None, description="Historical OHLCV data")
    include_smc: bool = Field(True, description="Include Smart Money Concepts analysis")
    include_confluence: bool = Field(True, description="Include multi-timeframe confluence")
    confidence_threshold: float = Field(0.6, description="Minimum confidence for signal generation")
    risk_level: str = Field("medium", description="Risk level: low, medium, high")


class TradingSignal(BaseModel):
    """Trading signal with entry/exit levels"""
    signal_type: str  # 'buy', 'sell', 'hold'
    confidence: float
    strength: str  # 'weak', 'moderate', 'strong', 'very_strong', 'extreme'
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    risk_reward_ratio: Optional[float]
    timeframe: str
    reasoning: List[str]


class TradingPredictionOutput(BaseModel):
    """Enhanced output for trading predictions"""
    symbol: str
    timestamp: str
    primary_timeframe: str

    # ML Predictions
    ml_prediction: Dict[str, Any]

    # SMC Analysis
    smc_analysis: Optional[Dict[str, Any]]

    # Multi-timeframe Confluence
    confluence_analysis: Optional[Dict[str, Any]]

    # Trading Signals
    primary_signal: TradingSignal
    alternative_signals: List[TradingSignal]

    # Risk Assessment
    risk_assessment: Dict[str, Any]

    # Market Context
    market_context: Dict[str, Any]


class EnhancedTradingPredictor:
    """
    Enhanced Trading Prediction Service

    Combines ML predictions with SMC analysis for comprehensive trading signals
    """

    def __init__(self, model_service: ModelService):
        """Initialize the enhanced trading predictor"""
        self.model_service = model_service
        self.smc_available = SMC_AVAILABLE

        # Risk level configurations
        self.risk_configs = {
            'low': {'confidence_threshold': 0.8, 'max_risk_per_trade': 0.01},
            'medium': {'confidence_threshold': 0.6, 'max_risk_per_trade': 0.02},
            'high': {'confidence_threshold': 0.4, 'max_risk_per_trade': 0.05}
        }

        logger.info(f"EnhancedTradingPredictor initialized (SMC available: {self.smc_available})")

    def predict_trading_signals(self, input_data: TradingPredictionInput) -> TradingPredictionOutput:
        """
        Generate comprehensive trading predictions and signals

        Args:
            input_data: Trading prediction input parameters

        Returns:
            Comprehensive trading prediction output
        """
        logger.info(f"Generating trading predictions for {input_data.symbol}")

        try:
            # Step 1: Get ML predictions
            ml_prediction = self._get_ml_prediction(input_data)

            # Step 2: Prepare OHLCV data
            ohlcv_data = self._prepare_ohlcv_data(input_data)

            # Step 3: Get SMC analysis if requested
            smc_analysis = None
            if input_data.include_smc and self.smc_available:
                smc_analysis = self._get_smc_analysis(ohlcv_data)

            # Step 4: Get multi-timeframe confluence if requested
            confluence_analysis = None
            if input_data.include_confluence and self.smc_available:
                confluence_analysis = self._get_confluence_analysis(ohlcv_data, input_data.timeframe)

            # Step 5: Generate trading signals
            signals = self._generate_trading_signals(
                ml_prediction, smc_analysis, confluence_analysis, input_data
            )

            # Step 6: Assess risk
            risk_assessment = self._assess_risk(signals, input_data)

            # Step 7: Generate market context
            market_context = self._generate_market_context(
                ml_prediction, smc_analysis, confluence_analysis, ohlcv_data
            )

            # Step 8: Create comprehensive output
            return TradingPredictionOutput(
                symbol=input_data.symbol,
                timestamp=datetime.now().isoformat(),
                primary_timeframe=input_data.timeframe,
                ml_prediction=ml_prediction,
                smc_analysis=smc_analysis,
                confluence_analysis=confluence_analysis,
                primary_signal=signals['primary'],
                alternative_signals=signals['alternatives'],
                risk_assessment=risk_assessment,
                market_context=market_context
            )

        except Exception as e:
            logger.error(f"Error generating trading predictions: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    def _get_ml_prediction(self, input_data: TradingPredictionInput) -> Dict[str, Any]:
        """Get ML model predictions"""
        try:
            # Prepare features for ML prediction
            if input_data.ohlcv_data and len(input_data.ohlcv_data) > 0:
                # Use provided OHLCV data
                latest_data = input_data.ohlcv_data[-1]
                features = {
                    'open': latest_data.get('open', 0),
                    'high': latest_data.get('high', 0),
                    'low': latest_data.get('low', 0),
                    'close': latest_data.get('close', 0),
                    'volume': latest_data.get('volume', 0)
                }
            else:
                # Use sample features (in production, fetch from data source)
                features = {
                    'open': 50000.0, 'high': 51000.0, 'low': 49500.0,
                    'close': 50500.0, 'volume': 1000000.0
                }

            # Get ML prediction
            ml_result = self.model_service.predict(
                symbol=input_data.symbol,
                features=features,
                sequence_length=60
            )

            # Enhance ML prediction with trading-specific analysis
            enhanced_prediction = {
                'raw_prediction': ml_result,
                'direction': ml_result.get('predicted_direction', 'neutral'),
                'confidence': ml_result.get('confidence', 0.5),
                'probabilities': ml_result.get('predictions', [0.33, 0.34, 0.33]),
                'model_version': ml_result.get('model_version', 'unknown'),
                'prediction_time': ml_result.get('prediction_time', datetime.now().isoformat())
            }

            return enhanced_prediction

        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
            # Return neutral prediction if ML fails
            return {
                'raw_prediction': None,
                'direction': 'neutral',
                'confidence': 0.5,
                'probabilities': [0.33, 0.34, 0.33],
                'model_version': 'fallback',
                'prediction_time': datetime.now().isoformat(),
                'error': str(e)
            }

    def _prepare_ohlcv_data(self, input_data: TradingPredictionInput) -> pd.DataFrame:
        """Prepare OHLCV data for analysis"""
        if input_data.ohlcv_data and len(input_data.ohlcv_data) > 0:
            # Convert provided data to DataFrame
            df = pd.DataFrame(input_data.ohlcv_data)

            # Ensure required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = 50000.0  # Default value

            # Add timestamp if not present
            if 'timestamp' not in df.columns:
                df['timestamp'] = pd.date_range(
                    start=datetime.now() - timedelta(minutes=len(df) * 15),
                    periods=len(df),
                    freq='15T'
                )

            return df
        else:
            # Generate sample data for demonstration
            logger.info("No OHLCV data provided, generating sample data")
            return self._generate_sample_ohlcv_data()

    def _generate_sample_ohlcv_data(self, num_candles: int = 500) -> pd.DataFrame:
        """Generate sample OHLCV data for testing"""
        np.random.seed(42)

        base_price = 50000
        data = []

        for i in range(num_candles):
            # Generate realistic price movement
            change = np.random.normal(0, 0.002)
            new_price = base_price * (1 + change)

            # Generate OHLC
            volatility = new_price * 0.001
            open_price = base_price + np.random.normal(0, volatility * 0.3)
            close_price = new_price

            high_price = max(open_price, close_price) + abs(np.random.normal(0, volatility * 0.4))
            low_price = min(open_price, close_price) - abs(np.random.normal(0, volatility * 0.4))

            volume = np.random.uniform(500000, 2000000)

            timestamp = datetime.now() - timedelta(minutes=(num_candles - i) * 15)

            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })

            base_price = close_price

        return pd.DataFrame(data)

    def _get_smc_analysis(self, ohlcv_data: pd.DataFrame) -> Dict[str, Any]:
        """Get Smart Money Concepts analysis"""
        try:
            # Initialize SMC detector
            smc_detector = SMCDector(ohlcv_data)

            # Get comprehensive SMC analysis
            smc_results = smc_detector.detect_all()

            # Extract key SMC insights for trading
            smc_insights = {
                'order_blocks': {
                    'total': len(smc_results.get('order_blocks', [])),
                    'bullish': len([ob for ob in smc_results.get('order_blocks', []) if ob.get('type') == 'bullish']),
                    'bearish': len([ob for ob in smc_results.get('order_blocks', []) if ob.get('type') == 'bearish']),
                    'recent_strength': self._get_recent_ob_strength(smc_results.get('order_blocks', []))
                },
                'fair_value_gaps': {
                    'total': len(smc_results.get('fvg', [])),
                    'active': len([fvg for fvg in smc_results.get('fvg', []) if fvg.get('filled', False) == False]),
                    'bullish': len([fvg for fvg in smc_results.get('fvg', []) if fvg.get('type') == 'bullish']),
                    'bearish': len([fvg for fvg in smc_results.get('fvg', []) if fvg.get('type') == 'bearish'])
                },
                'liquidity_levels': {
                    'total': len(smc_results.get('liquidity', [])),
                    'buy_side': len([liq for liq in smc_results.get('liquidity', []) if liq.get('type') == 'buy_side']),
                    'sell_side': len([liq for liq in smc_results.get('liquidity', []) if liq.get('type') == 'sell_side'])
                },
                'market_structure': smc_results.get('market_structure', {}),
                'multi_timeframe_confluence': smc_results.get('multi_timeframe_confluence', {}),
                'smc_bias': self._determine_smc_bias(smc_results),
                'key_levels': self._extract_key_levels(smc_results),
                'institutional_activity': self._assess_institutional_activity(smc_results)
            }

            return smc_insights

        except Exception as e:
            logger.warning(f"SMC analysis failed: {e}")
            return {
                'error': str(e),
                'smc_bias': 'neutral',
                'institutional_activity': 'low',
                'key_levels': []
            }

    def _get_confluence_analysis(self, ohlcv_data: pd.DataFrame, primary_timeframe: str) -> Dict[str, Any]:
        """Get multi-timeframe confluence analysis"""
        try:
            # Map timeframe string to TimeframeType
            timeframe_map = {
                '1m': TimeframeType.M1, '5m': TimeframeType.M5, '15m': TimeframeType.M15,
                '30m': TimeframeType.M30, '1h': TimeframeType.H1, '4h': TimeframeType.H4, '1d': TimeframeType.D1
            }

            primary_tf = timeframe_map.get(primary_timeframe, TimeframeType.M15)

            # Create multi-timeframe data sources
            data_sources = {
                primary_tf: ohlcv_data.copy(),
                TimeframeType.H1: self._resample_timeframe(ohlcv_data, '1H'),
                TimeframeType.H4: self._resample_timeframe(ohlcv_data, '4H')
            }

            # Get enhanced multi-timeframe analysis
            confluence_results = get_enhanced_multi_timeframe_analysis(
                data_sources, primary_timeframe=primary_tf
            )

            # Extract key confluence insights
            confluence_insights = {
                'best_signal': confluence_results.get('best_signal', {}),
                'htf_bias': self._extract_htf_bias(confluence_results.get('htf_biases', {})),
                'entry_zones': self._extract_entry_zones(confluence_results.get('discount_premium_zones', {})),
                'confluence_score': confluence_results.get('best_signal', {}).get('score', 0.5),
                'market_timing': confluence_results.get('market_timing_score', 0.5),
                'timeframe_alignment': self._assess_timeframe_alignment(confluence_results),
                'statistics': confluence_results.get('statistics', {})
            }

            return confluence_insights

        except Exception as e:
            logger.warning(f"Confluence analysis failed: {e}")
            return {
                'error': str(e),
                'best_signal': {'type': 'none', 'score': 0.5},
                'htf_bias': 'neutral',
                'confluence_score': 0.5,
                'market_timing': 0.5
            }

    def _generate_trading_signals(self, ml_prediction: Dict, smc_analysis: Optional[Dict],
                                confluence_analysis: Optional[Dict], input_data: TradingPredictionInput) -> Dict[str, Any]:
        """Generate comprehensive trading signals"""

        # Get risk configuration
        risk_config = self.risk_configs.get(input_data.risk_level, self.risk_configs['medium'])

        # Combine all analysis for signal generation
        combined_confidence = self._calculate_combined_confidence(
            ml_prediction, smc_analysis, confluence_analysis
        )

        # Determine primary signal direction
        primary_direction = self._determine_primary_direction(
            ml_prediction, smc_analysis, confluence_analysis
        )

        # Generate primary signal
        primary_signal = self._create_trading_signal(
            primary_direction, combined_confidence, ml_prediction,
            smc_analysis, confluence_analysis, input_data
        )

        # Generate alternative signals
        alternative_signals = self._generate_alternative_signals(
            ml_prediction, smc_analysis, confluence_analysis, input_data
        )

        return {
            'primary': primary_signal,
            'alternatives': alternative_signals
        }

    def _calculate_combined_confidence(self, ml_prediction: Dict, smc_analysis: Optional[Dict],
                                     confluence_analysis: Optional[Dict]) -> float:
        """Calculate combined confidence score from all analyses"""

        # Base ML confidence
        ml_confidence = ml_prediction.get('confidence', 0.5)

        # SMC confidence boost
        smc_boost = 0.0
        if smc_analysis:
            smc_bias = smc_analysis.get('smc_bias', 'neutral')
            institutional_activity = smc_analysis.get('institutional_activity', 'low')

            if smc_bias != 'neutral':
                smc_boost += 0.1
            if institutional_activity in ['medium', 'high']:
                smc_boost += 0.1

        # Confluence confidence boost
        confluence_boost = 0.0
        if confluence_analysis:
            confluence_score = confluence_analysis.get('confluence_score', 0.5)
            market_timing = confluence_analysis.get('market_timing', 0.5)

            confluence_boost = (confluence_score - 0.5) * 0.3
            confluence_boost += (market_timing - 0.5) * 0.2

        # Combine confidences (weighted average)
        combined = (ml_confidence * 0.5) + (smc_boost * 0.3) + (confluence_boost * 0.2)

        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, combined))

    def _determine_primary_direction(self, ml_prediction: Dict, smc_analysis: Optional[Dict],
                                   confluence_analysis: Optional[Dict]) -> str:
        """Determine primary trading direction"""

        # Get ML direction
        ml_direction = ml_prediction.get('direction', 'neutral')

        # Get SMC bias
        smc_bias = 'neutral'
        if smc_analysis:
            smc_bias = smc_analysis.get('smc_bias', 'neutral')

        # Get confluence direction
        confluence_direction = 'neutral'
        if confluence_analysis:
            best_signal = confluence_analysis.get('best_signal', {})
            confluence_direction = best_signal.get('type', 'neutral')
            if confluence_direction == 'none':
                confluence_direction = 'neutral'

        # Voting system for direction
        directions = [ml_direction, smc_bias, confluence_direction]

        # Count votes
        buy_votes = directions.count('up') + directions.count('bullish') + directions.count('buy')
        sell_votes = directions.count('down') + directions.count('bearish') + directions.count('sell')

        if buy_votes > sell_votes:
            return 'buy'
        elif sell_votes > buy_votes:
            return 'sell'
        else:
            return 'hold'

    def _create_trading_signal(self, direction: str, confidence: float, ml_prediction: Dict,
                             smc_analysis: Optional[Dict], confluence_analysis: Optional[Dict],
                             input_data: TradingPredictionInput) -> TradingSignal:
        """Create a comprehensive trading signal"""

        # Determine signal strength based on confidence
        if confidence >= 0.8:
            strength = 'very_strong'
        elif confidence >= 0.7:
            strength = 'strong'
        elif confidence >= 0.6:
            strength = 'moderate'
        else:
            strength = 'weak'

        # Calculate entry/exit levels (simplified for demo)
        current_price = 50500.0  # In production, get from latest OHLCV

        entry_price = None
        stop_loss = None
        take_profit = None
        risk_reward_ratio = None

        if direction in ['buy', 'sell']:
            entry_price = current_price

            if direction == 'buy':
                stop_loss = current_price * 0.98  # 2% stop loss
                take_profit = current_price * 1.04  # 4% take profit
            else:  # sell
                stop_loss = current_price * 1.02  # 2% stop loss
                take_profit = current_price * 0.96  # 4% take profit

            # Calculate risk/reward ratio
            if stop_loss and take_profit:
                risk = abs(entry_price - stop_loss)
                reward = abs(take_profit - entry_price)
                risk_reward_ratio = reward / risk if risk > 0 else 1.0

        # Generate reasoning
        reasoning = self._generate_signal_reasoning(
            direction, confidence, ml_prediction, smc_analysis, confluence_analysis
        )

        return TradingSignal(
            signal_type=direction,
            confidence=confidence,
            strength=strength,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=risk_reward_ratio,
            timeframe=input_data.timeframe,
            reasoning=reasoning
        )

    # Helper methods for SMC analysis
    def _get_recent_ob_strength(self, order_blocks: List) -> float:
        """Get average strength of recent order blocks"""
        if not order_blocks:
            return 0.0

        recent_blocks = order_blocks[-5:]  # Last 5 order blocks
        strengths = [ob.get('strength', 0.5) for ob in recent_blocks if isinstance(ob, dict)]

        if not strengths:
            return 0.0

        return sum(strengths) / len(strengths)

    def _determine_smc_bias(self, smc_results: Dict) -> str:
        """Determine overall SMC bias"""
        try:
            # Check market structure
            market_structure = smc_results.get('market_structure', {})
            if isinstance(market_structure, dict):
                current_trend = market_structure.get('current_trend', 'neutral')
                if current_trend in ['bullish', 'bearish']:
                    return current_trend

            # Check order block bias
            order_blocks = smc_results.get('order_blocks', [])
            if order_blocks:
                bullish_count = len([ob for ob in order_blocks if ob.get('type') == 'bullish'])
                bearish_count = len([ob for ob in order_blocks if ob.get('type') == 'bearish'])

                if bullish_count > bearish_count:
                    return 'bullish'
                elif bearish_count > bullish_count:
                    return 'bearish'

            return 'neutral'

        except Exception:
            return 'neutral'

    def _extract_key_levels(self, smc_results: Dict) -> List[float]:
        """Extract key price levels from SMC analysis"""
        key_levels = []

        try:
            # Extract from order blocks
            order_blocks = smc_results.get('order_blocks', [])
            for ob in order_blocks:
                if isinstance(ob, dict):
                    if 'top' in ob:
                        key_levels.append(ob['top'])
                    if 'bottom' in ob:
                        key_levels.append(ob['bottom'])

            # Extract from liquidity levels
            liquidity = smc_results.get('liquidity', [])
            for liq in liquidity:
                if isinstance(liq, dict) and 'price' in liq:
                    key_levels.append(liq['price'])

            # Remove duplicates and sort
            key_levels = sorted(set(key_levels))

            # Return most relevant levels (within reasonable range)
            current_price = 50500.0  # In production, get from latest data
            relevant_levels = []

            for level in key_levels:
                distance_pct = abs(level - current_price) / current_price
                if distance_pct <= 0.1:  # Within 10%
                    relevant_levels.append(level)

            return relevant_levels[:10]  # Top 10 levels

        except Exception:
            return []

    def _assess_institutional_activity(self, smc_results: Dict) -> str:
        """Assess level of institutional activity"""
        try:
            activity_score = 0

            # Check order block activity
            order_blocks = smc_results.get('order_blocks', [])
            if len(order_blocks) > 5:
                activity_score += 1

            # Check FVG activity
            fvgs = smc_results.get('fvg', [])
            active_fvgs = [fvg for fvg in fvgs if not fvg.get('filled', True)]
            if len(active_fvgs) > 3:
                activity_score += 1

            # Check liquidity levels
            liquidity = smc_results.get('liquidity', [])
            if len(liquidity) > 10:
                activity_score += 1

            # Check market structure quality
            market_structure = smc_results.get('market_structure', {})
            if isinstance(market_structure, dict):
                structure_quality = market_structure.get('structure_quality', 0)
                if structure_quality > 0.7:
                    activity_score += 1

            if activity_score >= 3:
                return 'high'
            elif activity_score >= 2:
                return 'medium'
            else:
                return 'low'

        except Exception:
            return 'low'

    # Helper methods for confluence analysis
    def _resample_timeframe(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample data to higher timeframe"""
        try:
            df = data.copy()
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)

            resampled = df.resample(timeframe).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            resampled.reset_index(inplace=True)
            return resampled

        except Exception:
            return data.copy()

    def _extract_htf_bias(self, htf_biases: Dict) -> str:
        """Extract overall higher timeframe bias"""
        if not htf_biases:
            return 'neutral'

        bullish_count = 0
        bearish_count = 0

        for timeframe, bias_info in htf_biases.items():
            if isinstance(bias_info, dict):
                direction = bias_info.get('direction', 'neutral')
                if direction == 'bullish':
                    bullish_count += 1
                elif direction == 'bearish':
                    bearish_count += 1

        if bullish_count > bearish_count:
            return 'bullish'
        elif bearish_count > bullish_count:
            return 'bearish'
        else:
            return 'neutral'

    def _extract_entry_zones(self, zones: Dict) -> Dict[str, Any]:
        """Extract entry zone information"""
        if not zones:
            return {'primary_zone': 'equilibrium', 'zone_strength': 0.5}

        # Get primary timeframe zone (usually 15m)
        primary_zone_info = zones.get('15m', {})

        if isinstance(primary_zone_info, dict):
            zone_type = primary_zone_info.get('zone_type', 'equilibrium')
            zone_strength = primary_zone_info.get('strength', 0.5)

            return {
                'primary_zone': zone_type,
                'zone_strength': zone_strength,
                'all_zones': zones
            }

        return {'primary_zone': 'equilibrium', 'zone_strength': 0.5}

    def _assess_timeframe_alignment(self, confluence_results: Dict) -> Dict[str, Any]:
        """Assess alignment across timeframes"""
        try:
            statistics = confluence_results.get('statistics', {})

            total_timeframes = statistics.get('total_timeframes', 0)
            bullish_timeframes = statistics.get('bullish_timeframes', 0)
            bearish_timeframes = statistics.get('bearish_timeframes', 0)

            if total_timeframes > 0:
                bullish_pct = bullish_timeframes / total_timeframes
                bearish_pct = bearish_timeframes / total_timeframes

                alignment_score = max(bullish_pct, bearish_pct)

                if alignment_score >= 0.8:
                    alignment_strength = 'very_strong'
                elif alignment_score >= 0.6:
                    alignment_strength = 'strong'
                elif alignment_score >= 0.4:
                    alignment_strength = 'moderate'
                else:
                    alignment_strength = 'weak'

                return {
                    'alignment_score': alignment_score,
                    'alignment_strength': alignment_strength,
                    'bullish_percentage': bullish_pct,
                    'bearish_percentage': bearish_pct,
                    'total_timeframes': total_timeframes
                }

            return {
                'alignment_score': 0.5,
                'alignment_strength': 'weak',
                'bullish_percentage': 0.0,
                'bearish_percentage': 0.0,
                'total_timeframes': 0
            }

        except Exception:
            return {
                'alignment_score': 0.5,
                'alignment_strength': 'weak',
                'bullish_percentage': 0.0,
                'bearish_percentage': 0.0,
                'total_timeframes': 0
            }

    def _generate_alternative_signals(self, ml_prediction: Dict, smc_analysis: Optional[Dict],
                                    confluence_analysis: Optional[Dict], input_data: TradingPredictionInput) -> List[TradingSignal]:
        """Generate alternative trading signals"""
        alternatives = []

        # Generate contrarian signal if confidence is low
        primary_direction = self._determine_primary_direction(ml_prediction, smc_analysis, confluence_analysis)
        primary_confidence = self._calculate_combined_confidence(ml_prediction, smc_analysis, confluence_analysis)

        if primary_confidence < 0.6:
            contrarian_direction = 'sell' if primary_direction == 'buy' else 'buy' if primary_direction == 'sell' else 'hold'
            contrarian_signal = self._create_trading_signal(
                contrarian_direction, 1.0 - primary_confidence, ml_prediction,
                smc_analysis, confluence_analysis, input_data
            )
            alternatives.append(contrarian_signal)

        # Generate conservative signal (always hold unless very confident)
        if primary_confidence < 0.8:
            conservative_signal = self._create_trading_signal(
                'hold', 0.8, ml_prediction, smc_analysis, confluence_analysis, input_data
            )
            alternatives.append(conservative_signal)

        return alternatives

    def _generate_signal_reasoning(self, direction: str, confidence: float, ml_prediction: Dict,
                                 smc_analysis: Optional[Dict], confluence_analysis: Optional[Dict]) -> List[str]:
        """Generate reasoning for the trading signal"""
        reasoning = []

        # ML reasoning
        ml_direction = ml_prediction.get('direction', 'neutral')
        ml_confidence = ml_prediction.get('confidence', 0.5)
        reasoning.append(f"ML model predicts {ml_direction} with {ml_confidence:.1%} confidence")

        # SMC reasoning
        if smc_analysis:
            smc_bias = smc_analysis.get('smc_bias', 'neutral')
            institutional_activity = smc_analysis.get('institutional_activity', 'low')
            reasoning.append(f"SMC analysis shows {smc_bias} bias with {institutional_activity} institutional activity")

            order_blocks = smc_analysis.get('order_blocks', {})
            if order_blocks.get('total', 0) > 0:
                reasoning.append(f"Detected {order_blocks['total']} order blocks ({order_blocks.get('bullish', 0)} bullish, {order_blocks.get('bearish', 0)} bearish)")

        # Confluence reasoning
        if confluence_analysis:
            confluence_score = confluence_analysis.get('confluence_score', 0.5)
            htf_bias = confluence_analysis.get('htf_bias', 'neutral')
            market_timing = confluence_analysis.get('market_timing', 0.5)

            reasoning.append(f"Multi-timeframe confluence score: {confluence_score:.1%}")
            reasoning.append(f"Higher timeframe bias: {htf_bias}")
            reasoning.append(f"Market timing score: {market_timing:.1%}")

        # Overall confidence reasoning
        if confidence >= 0.8:
            reasoning.append("High confidence signal - strong alignment across all analyses")
        elif confidence >= 0.6:
            reasoning.append("Moderate confidence signal - good alignment with some divergence")
        else:
            reasoning.append("Low confidence signal - mixed signals across analyses")

        return reasoning

    def _assess_risk(self, signals: Dict, input_data: TradingPredictionInput) -> Dict[str, Any]:
        """Assess risk for the trading signals"""
        primary_signal = signals['primary']
        risk_config = self.risk_configs.get(input_data.risk_level, self.risk_configs['medium'])

        # Calculate position size based on risk
        max_risk_per_trade = risk_config['max_risk_per_trade']

        risk_assessment = {
            'risk_level': input_data.risk_level,
            'max_risk_per_trade': max_risk_per_trade,
            'confidence_threshold': risk_config['confidence_threshold'],
            'signal_meets_threshold': primary_signal.confidence >= risk_config['confidence_threshold'],
            'recommended_position_size': self._calculate_position_size(primary_signal, max_risk_per_trade),
            'risk_factors': self._identify_risk_factors(primary_signal, input_data),
            'risk_mitigation': self._suggest_risk_mitigation(primary_signal)
        }

        return risk_assessment

    def _calculate_position_size(self, signal: TradingSignal, max_risk_per_trade: float) -> Dict[str, Any]:
        """Calculate recommended position size"""
        if signal.signal_type == 'hold' or not signal.entry_price or not signal.stop_loss:
            return {
                'percentage_of_portfolio': 0.0,
                'risk_amount': 0.0,
                'reasoning': 'No position recommended for hold signal or missing price levels'
            }

        # Calculate risk per unit
        risk_per_unit = abs(signal.entry_price - signal.stop_loss)

        # Calculate position size based on max risk
        if risk_per_unit > 0:
            # Adjust position size based on confidence
            confidence_multiplier = min(signal.confidence / 0.6, 1.0)  # Scale down if confidence < 60%
            adjusted_risk = max_risk_per_trade * confidence_multiplier

            return {
                'percentage_of_portfolio': adjusted_risk,
                'risk_per_unit': risk_per_unit,
                'confidence_multiplier': confidence_multiplier,
                'reasoning': f'Position sized for {adjusted_risk:.1%} portfolio risk based on {signal.confidence:.1%} confidence'
            }

        return {
            'percentage_of_portfolio': 0.0,
            'risk_amount': 0.0,
            'reasoning': 'Cannot calculate position size - invalid price levels'
        }

    def _identify_risk_factors(self, signal: TradingSignal, input_data: TradingPredictionInput) -> List[str]:
        """Identify potential risk factors"""
        risk_factors = []

        if signal.confidence < 0.6:
            risk_factors.append("Low confidence signal - higher probability of false signal")

        if signal.risk_reward_ratio and signal.risk_reward_ratio < 1.5:
            risk_factors.append("Poor risk/reward ratio - limited profit potential")

        if input_data.timeframe in ['1m', '5m']:
            risk_factors.append("Short timeframe trading - higher noise and false signals")

        if signal.strength == 'weak':
            risk_factors.append("Weak signal strength - consider waiting for better setup")

        return risk_factors

    def _suggest_risk_mitigation(self, signal: TradingSignal) -> List[str]:
        """Suggest risk mitigation strategies"""
        mitigation = []

        if signal.confidence < 0.7:
            mitigation.append("Consider reducing position size due to lower confidence")
            mitigation.append("Wait for additional confirmation before entry")

        if signal.risk_reward_ratio and signal.risk_reward_ratio < 2.0:
            mitigation.append("Consider tighter entry or wider profit target to improve R:R")

        mitigation.append("Use proper stop loss management")
        mitigation.append("Monitor price action for invalidation signals")

        if signal.signal_type in ['buy', 'sell']:
            mitigation.append("Consider scaling into position rather than full size entry")

        return mitigation

    def _generate_market_context(self, ml_prediction: Dict, smc_analysis: Optional[Dict],
                               confluence_analysis: Optional[Dict], ohlcv_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate market context information"""

        # Basic market metrics
        latest_data = ohlcv_data.iloc[-1] if len(ohlcv_data) > 0 else {}
        current_price = latest_data.get('close', 50500.0)

        # Calculate basic volatility
        if len(ohlcv_data) >= 20:
            returns = ohlcv_data['close'].pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)  # Annualized
        else:
            volatility = 0.2  # Default 20%

        # Market regime
        market_regime = self._determine_market_regime(ohlcv_data)

        context = {
            'current_price': current_price,
            'volatility': volatility,
            'market_regime': market_regime,
            'data_quality': {
                'data_points': len(ohlcv_data),
                'data_coverage': 'sufficient' if len(ohlcv_data) >= 100 else 'limited'
            },
            'analysis_summary': {
                'ml_available': ml_prediction.get('error') is None,
                'smc_available': smc_analysis is not None and smc_analysis.get('error') is None,
                'confluence_available': confluence_analysis is not None and confluence_analysis.get('error') is None
            }
        }

        return context

    def _determine_market_regime(self, ohlcv_data: pd.DataFrame) -> str:
        """Determine current market regime"""
        if len(ohlcv_data) < 50:
            return 'insufficient_data'

        # Calculate trend using moving averages
        data = ohlcv_data.copy()
        data['sma_20'] = data['close'].rolling(20).mean()
        data['sma_50'] = data['close'].rolling(50).mean()

        latest = data.iloc[-1]
        current_price = latest['close']
        sma_20 = latest['sma_20']
        sma_50 = latest['sma_50']

        # Determine regime
        if current_price > sma_20 > sma_50:
            return 'trending_up'
        elif current_price < sma_20 < sma_50:
            return 'trending_down'
        else:
            return 'ranging'


# Create enhanced trading predictor instance
enhanced_predictor = None


def get_enhanced_trading_predictor(model_service: ModelService = Depends(get_model_service)) -> EnhancedTradingPredictor:
    """Dependency for getting enhanced trading predictor instance"""
    global enhanced_predictor
    if enhanced_predictor is None:
        enhanced_predictor = EnhancedTradingPredictor(model_service)
    return enhanced_predictor


# API Endpoints
@router.post("/trading-predictions", response_model=TradingPredictionOutput)
async def get_trading_predictions(
    input_data: TradingPredictionInput,
    predictor: EnhancedTradingPredictor = Depends(get_enhanced_trading_predictor)
) -> TradingPredictionOutput:
    """
    Get comprehensive trading predictions with ML and SMC analysis
    """
    return predictor.predict_trading_signals(input_data)


@router.get("/trading-predictions/{symbol}")
async def get_quick_predictions(
    symbol: str,
    timeframe: str = "15m",
    include_smc: bool = True,
    include_confluence: bool = True,
    predictor: EnhancedTradingPredictor = Depends(get_enhanced_trading_predictor)
) -> TradingPredictionOutput:
    """
    Get quick trading predictions for a symbol
    """
    input_data = TradingPredictionInput(
        symbol=symbol,
        timeframe=timeframe,
        include_smc=include_smc,
        include_confluence=include_confluence
    )

    return predictor.predict_trading_signals(input_data)


@router.get("/health")
async def health_check():
    """Health check for enhanced trading predictions service"""
    return {
        "status": "healthy",
        "service": "Enhanced Trading Predictions",
        "smc_available": SMC_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    }
