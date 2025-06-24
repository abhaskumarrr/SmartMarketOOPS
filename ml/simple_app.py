#!/usr/bin/env python3
"""
Simple ML API for SmartMarketOOPS
Provides ML predictions for trading decisions
"""

from flask import Flask, jsonify, request
import numpy as np
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class SimpleTradingML:
    """Simple ML model for trading predictions"""
    
    def __init__(self):
        self.model_name = "SimpleTradingML"
        self.confidence_threshold = 0.65
        logger.info(f"Initialized {self.model_name}")
    
    def predict(self, market_data):
        """Generate trading prediction based on market data"""
        try:
            # Simple momentum-based prediction
            if len(market_data) < 10:
                return {
                    'action': 'hold',
                    'confidence': 0.5,
                    'reason': 'insufficient_data'
                }
            
            # Calculate simple indicators
            prices = [float(d.get('close', 0)) for d in market_data[-10:]]
            volumes = [float(d.get('volume', 0)) for d in market_data[-10:]]
            
            if not prices or all(p == 0 for p in prices):
                return {
                    'action': 'hold',
                    'confidence': 0.5,
                    'reason': 'invalid_data'
                }
            
            # Simple moving averages
            short_ma = np.mean(prices[-5:])
            long_ma = np.mean(prices[-10:])
            current_price = prices[-1]
            
            # Volume analysis
            avg_volume = np.mean(volumes[-5:]) if volumes else 1
            current_volume = volumes[-1] if volumes else 1
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Momentum calculation
            momentum = (short_ma / long_ma - 1) * 100 if long_ma > 0 else 0
            
            # Price trend
            price_trend = (current_price / prices[0] - 1) * 100 if prices[0] > 0 else 0
            
            # Generate prediction
            confidence = 0.5
            action = 'hold'
            
            # Buy signals
            if momentum > 1.0 and price_trend > 2.0 and volume_ratio > 1.2:
                action = 'buy'
                confidence = min(0.85, 0.6 + abs(momentum) * 0.05)
            
            # Sell signals
            elif momentum < -1.0 and price_trend < -2.0 and volume_ratio > 1.2:
                action = 'sell'
                confidence = min(0.85, 0.6 + abs(momentum) * 0.05)
            
            # Moderate signals
            elif momentum > 0.5 and volume_ratio > 1.0:
                action = 'buy'
                confidence = 0.58
            elif momentum < -0.5 and volume_ratio > 1.0:
                action = 'sell'
                confidence = 0.58
            
            return {
                'action': action,
                'confidence': confidence,
                'momentum': momentum,
                'price_trend': price_trend,
                'volume_ratio': volume_ratio,
                'current_price': current_price,
                'short_ma': short_ma,
                'long_ma': long_ma,
                'timestamp': datetime.now().isoformat(),
                'model': self.model_name
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'action': 'hold',
                'confidence': 0.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Initialize ML model
ml_model = SimpleTradingML()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'SmartMarketOOPS ML Service',
        'model': ml_model.model_name,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Generate trading prediction"""
    try:
        data = request.get_json()
        
        if not data or 'market_data' not in data:
            return jsonify({
                'error': 'market_data required',
                'status': 'error'
            }), 400
        
        market_data = data['market_data']
        prediction = ml_model.predict(market_data)
        
        return jsonify({
            'prediction': prediction,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/models/info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'models': [
            {
                'name': ml_model.model_name,
                'type': 'momentum_based',
                'confidence_threshold': ml_model.confidence_threshold,
                'features': [
                    'price_momentum',
                    'moving_averages',
                    'volume_analysis',
                    'trend_detection'
                ]
            }
        ],
        'status': 'success'
    })

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    return """# HELP ml_service_predictions_total Total number of predictions made
# TYPE ml_service_predictions_total counter
ml_service_predictions_total 0

# HELP ml_service_health Service health status
# TYPE ml_service_health gauge
ml_service_health 1
"""

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    logger.info(f"Starting ML service on port {port}")
    logger.info(f"Model: {ml_model.model_name}")
    logger.info(f"Confidence threshold: {ml_model.confidence_threshold}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)