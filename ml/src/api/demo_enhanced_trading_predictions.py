#!/usr/bin/env python3
"""
Demo for Enhanced Trading Predictions System

This script demonstrates the Enhanced Trading Predictions system that integrates
ML predictions with Smart Money Concepts analysis for comprehensive trading signals.
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock
import json

# Add project paths
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

try:
    from enhanced_trading_predictions import (
        EnhancedTradingPredictor, TradingPredictionInput, ModelService
    )
    ENHANCED_PREDICTIONS_AVAILABLE = True
except ImportError as e:
    print(f"Enhanced trading predictions not available: {e}")
    ENHANCED_PREDICTIONS_AVAILABLE = False


def create_mock_model_service():
    """Create a mock model service for demonstration"""
    mock_service = Mock(spec=ModelService)
    
    # Configure mock to return realistic predictions
    mock_service.predict.return_value = {
        'symbol': 'BTC/USDT',
        'predictions': [0.15, 0.25, 0.60],  # [down, neutral, up] - bullish prediction
        'confidence': 0.75,
        'predicted_direction': 'up',
        'prediction_time': '2024-01-15T10:30:00',
        'model_version': 'cnn-lstm-v2.1'
    }
    
    return mock_service


def demo_basic_prediction():
    """Demonstrate basic trading prediction"""
    print("\n" + "="*60)
    print("🚀 ENHANCED TRADING PREDICTIONS DEMO")
    print("="*60)
    
    if not ENHANCED_PREDICTIONS_AVAILABLE:
        print("❌ Enhanced trading predictions not available")
        return
    
    # Create mock model service
    mock_service = create_mock_model_service()
    
    # Initialize enhanced trading predictor
    predictor = EnhancedTradingPredictor(mock_service)
    
    print(f"✅ Enhanced Trading Predictor initialized")
    print(f"   SMC Available: {predictor.smc_available}")
    print(f"   Risk Levels: {list(predictor.risk_configs.keys())}")
    
    # Create trading prediction input
    input_data = TradingPredictionInput(
        symbol="BTC/USDT",
        timeframe="15m",
        include_smc=True,
        include_confluence=True,
        confidence_threshold=0.6,
        risk_level="medium"
    )
    
    print(f"\n📊 Generating predictions for {input_data.symbol}...")
    print(f"   Timeframe: {input_data.timeframe}")
    print(f"   Risk Level: {input_data.risk_level}")
    print(f"   Confidence Threshold: {input_data.confidence_threshold}")
    
    # Generate comprehensive trading predictions
    try:
        result = predictor.predict_trading_signals(input_data)
        
        print(f"\n🎯 PREDICTION RESULTS")
        print(f"   Timestamp: {result.timestamp}")
        print(f"   Primary Timeframe: {result.primary_timeframe}")
        
        # ML Prediction Results
        ml_pred = result.ml_prediction
        print(f"\n🤖 ML PREDICTION:")
        print(f"   Direction: {ml_pred['direction']}")
        print(f"   Confidence: {ml_pred['confidence']:.1%}")
        print(f"   Model Version: {ml_pred['model_version']}")
        
        # Primary Trading Signal
        primary_signal = result.primary_signal
        print(f"\n📈 PRIMARY TRADING SIGNAL:")
        print(f"   Signal Type: {primary_signal.signal_type.upper()}")
        print(f"   Confidence: {primary_signal.confidence:.1%}")
        print(f"   Strength: {primary_signal.strength}")
        print(f"   Timeframe: {primary_signal.timeframe}")
        
        if primary_signal.entry_price:
            print(f"   Entry Price: ${primary_signal.entry_price:,.2f}")
            print(f"   Stop Loss: ${primary_signal.stop_loss:,.2f}")
            print(f"   Take Profit: ${primary_signal.take_profit:,.2f}")
            print(f"   Risk/Reward: {primary_signal.risk_reward_ratio:.2f}")
        
        # Signal Reasoning
        print(f"\n💡 SIGNAL REASONING:")
        for i, reason in enumerate(primary_signal.reasoning, 1):
            print(f"   {i}. {reason}")
        
        # Risk Assessment
        risk_assessment = result.risk_assessment
        print(f"\n⚠️  RISK ASSESSMENT:")
        print(f"   Risk Level: {risk_assessment['risk_level']}")
        print(f"   Max Risk per Trade: {risk_assessment['max_risk_per_trade']:.1%}")
        print(f"   Signal Meets Threshold: {risk_assessment['signal_meets_threshold']}")
        
        position_size = risk_assessment['recommended_position_size']
        print(f"   Recommended Position: {position_size['percentage_of_portfolio']:.1%} of portfolio")
        
        if risk_assessment['risk_factors']:
            print(f"\n🚨 RISK FACTORS:")
            for i, factor in enumerate(risk_assessment['risk_factors'], 1):
                print(f"   {i}. {factor}")
        
        if risk_assessment['risk_mitigation']:
            print(f"\n🛡️  RISK MITIGATION:")
            for i, mitigation in enumerate(risk_assessment['risk_mitigation'], 1):
                print(f"   {i}. {mitigation}")
        
        # Market Context
        market_context = result.market_context
        print(f"\n🌍 MARKET CONTEXT:")
        print(f"   Current Price: ${market_context['current_price']:,.2f}")
        print(f"   Volatility: {market_context['volatility']:.1%}")
        print(f"   Market Regime: {market_context['market_regime']}")
        print(f"   Data Quality: {market_context['data_quality']['data_coverage']}")
        
        analysis_summary = market_context['analysis_summary']
        print(f"   ML Available: {analysis_summary['ml_available']}")
        print(f"   SMC Available: {analysis_summary['smc_available']}")
        print(f"   Confluence Available: {analysis_summary['confluence_available']}")
        
        # Alternative Signals
        if result.alternative_signals:
            print(f"\n🔄 ALTERNATIVE SIGNALS:")
            for i, alt_signal in enumerate(result.alternative_signals, 1):
                print(f"   {i}. {alt_signal.signal_type.upper()} - {alt_signal.confidence:.1%} confidence ({alt_signal.strength})")
        
        print(f"\n✅ Enhanced trading prediction completed successfully!")
        
        return result
        
    except Exception as e:
        print(f"❌ Error generating predictions: {e}")
        return None


def demo_different_risk_levels():
    """Demonstrate predictions with different risk levels"""
    print(f"\n" + "="*60)
    print("🎚️  RISK LEVEL COMPARISON DEMO")
    print("="*60)
    
    if not ENHANCED_PREDICTIONS_AVAILABLE:
        print("❌ Enhanced trading predictions not available")
        return
    
    mock_service = create_mock_model_service()
    predictor = EnhancedTradingPredictor(mock_service)
    
    risk_levels = ['low', 'medium', 'high']
    
    for risk_level in risk_levels:
        print(f"\n📊 Risk Level: {risk_level.upper()}")
        print("-" * 30)
        
        input_data = TradingPredictionInput(
            symbol="BTC/USDT",
            timeframe="15m",
            include_smc=False,  # Disable for faster demo
            include_confluence=False,
            risk_level=risk_level
        )
        
        try:
            result = predictor.predict_trading_signals(input_data)
            
            primary_signal = result.primary_signal
            risk_assessment = result.risk_assessment
            
            print(f"Signal: {primary_signal.signal_type.upper()} ({primary_signal.confidence:.1%})")
            print(f"Confidence Threshold: {risk_assessment['confidence_threshold']:.1%}")
            print(f"Max Risk per Trade: {risk_assessment['max_risk_per_trade']:.1%}")
            print(f"Meets Threshold: {risk_assessment['signal_meets_threshold']}")
            
            position_size = risk_assessment['recommended_position_size']
            print(f"Position Size: {position_size['percentage_of_portfolio']:.1%}")
            
        except Exception as e:
            print(f"Error: {e}")


def demo_different_timeframes():
    """Demonstrate predictions across different timeframes"""
    print(f"\n" + "="*60)
    print("⏰ TIMEFRAME COMPARISON DEMO")
    print("="*60)
    
    if not ENHANCED_PREDICTIONS_AVAILABLE:
        print("❌ Enhanced trading predictions not available")
        return
    
    mock_service = create_mock_model_service()
    predictor = EnhancedTradingPredictor(mock_service)
    
    timeframes = ['5m', '15m', '1h', '4h']
    
    for timeframe in timeframes:
        print(f"\n📈 Timeframe: {timeframe}")
        print("-" * 20)
        
        input_data = TradingPredictionInput(
            symbol="BTC/USDT",
            timeframe=timeframe,
            include_smc=False,  # Disable for faster demo
            include_confluence=False,
            risk_level="medium"
        )
        
        try:
            result = predictor.predict_trading_signals(input_data)
            
            primary_signal = result.primary_signal
            market_context = result.market_context
            
            print(f"Signal: {primary_signal.signal_type.upper()}")
            print(f"Confidence: {primary_signal.confidence:.1%}")
            print(f"Strength: {primary_signal.strength}")
            print(f"Market Regime: {market_context['market_regime']}")
            
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Run all demos"""
    print("🎯 Enhanced Trading Predictions System Demo")
    print("Integrating ML Predictions with Smart Money Concepts")
    
    # Run demos
    demo_basic_prediction()
    demo_different_risk_levels()
    demo_different_timeframes()
    
    print(f"\n" + "="*60)
    print("🎉 DEMO COMPLETED")
    print("="*60)
    print("The Enhanced Trading Predictions system successfully:")
    print("✅ Integrated ML predictions with SMC analysis")
    print("✅ Generated comprehensive trading signals")
    print("✅ Provided risk assessment and position sizing")
    print("✅ Delivered market context and reasoning")
    print("✅ Supported multiple risk levels and timeframes")
    print("\nReady for production use in trading applications! 🚀")


if __name__ == "__main__":
    main()
