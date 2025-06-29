/**
 * MASTER TRADING AI - REAL MACHINE LEARNING INTEGRATION
 * 
 * This integrates with the actual Python ML infrastructure:
 * - EnhancedNeuralNetwork (LSTM + Attention)
 * - MultiModelEnsemble (CNN-LSTM, Transformer, SMC)
 * - Real feature engineering and training pipelines
 * - Dynamic ensemble voting with confidence scoring
 * 
 * NO MORE FAKE IF/ELSE LOGIC - REAL MACHINE LEARNING!
 */

import axios from 'axios';
import { logger } from '../../utils/logger';

// Real ML Model Interfaces
interface MLFeatures {
  price_data: Array<{
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
    timestamp: Date;
  }>;
  technical_indicators: {
    rsi?: number;
    macd?: number;
    macd_signal?: number;
    macd_histogram?: number;
    ema_12?: number;
    ema_26?: number;
    bollinger_upper?: number;
    bollinger_lower?: number;
    bollinger_middle?: number;
    atr?: number;
    volume_sma?: number;
  };
  market_structure: {
    spread: number;
    price_velocity: number;
    volume_ratio: number;
    momentum: number;
    volatility: number;
  };
  metadata: {
    symbol: string;
    timeframe: string;
    timestamp: Date;
    data_points: number;
  };
}

interface MLModelPrediction {
  prediction: number;
  confidence: number;
  model_name: string;
  features_used: string[];
  reasoning?: string[];
  timestamp: Date;
}

interface EnsemblePrediction {
  ensemble_prediction: number;
  ensemble_confidence: number;
  quality_score: number;
  uncertainty: number;
  individual_predictions: Record<string, number>;
  individual_confidences: Record<string, number>;
  weights: number[];
  voting_method: string;
  market_regime?: string;
  recommendation: 'BUY' | 'SELL' | 'HOLD';
  model_status: Record<string, boolean>;
}

interface MarketData {
  timestamp: Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  indicators?: {
    rsi?: number;
    macd?: number;
    macd_signal?: number;
    macd_histogram?: number;
    ema_12?: number;
    ema_26?: number;
    bollinger_upper?: number;
    bollinger_lower?: number;
    bollinger_middle?: number;
    atr?: number;
    volume_sma?: number;
  };
}

interface TradingSignalOutput {
  id: string;
  symbol: string;
  timeframe: string;
  signalType: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  strength: 'VERY_STRONG' | 'STRONG' | 'MODERATE' | 'WEAK';
  entryPrice?: number;
  stopLoss?: number;
  takeProfit?: number;
  riskRewardRatio?: number;
  timestamp: Date;
  reasoning: string[];
  metadata: {
    ml_model: string;
    prediction: number;
    quality_score: number;
    ensemble_data: EnsemblePrediction;
    source: string;
    model_performances: Record<string, number>;
  };
}

export class MasterTradingAI {
  private readonly ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8001';
  
  // Professional thresholds
  private readonly CONFIDENCE_THRESHOLD = 0.75;  // High confidence required
  private readonly QUALITY_THRESHOLD = 0.80;     // High quality required
  private readonly MIN_MODELS_REQUIRED = 2;      // Ensemble requirement
  private readonly MAX_TRADES_PER_DAY = 3;       // Quality over quantity

  constructor() {
    logger.info('🤖 Initializing Master Trading AI with REAL ML integration');
  }

  /**
   * Generate ML-based trading signal using the real Python ensemble
   */
  async generateSignal(
    symbol: string,
    timeframe: string,
    data: MarketData[]
  ): Promise<TradingSignalOutput | null> {
    try {
      logger.info(`🔮 Generating ML signal for ${symbol} ${timeframe} with ${data.length} data points`);

      if (data.length < 50) {
        logger.warn('❌ Insufficient data for ML prediction (need 50+ points)');
        return null;
      }

      // 1. Prepare features for the real ML models
      const features = this.prepareMLFeatures(data, symbol, timeframe);
      
      // 2. Call the real ensemble prediction API
      const ensemblePrediction = await this.getEnsemblePrediction(features);
      
      if (!ensemblePrediction) {
        logger.warn('❌ ML ensemble prediction failed');
        return null;
      }

      // 3. Validate signal quality
      const signalQuality = this.validateSignalQuality(ensemblePrediction);
      
      if (!signalQuality.isValid) {
        logger.info(`❌ Signal quality insufficient: ${signalQuality.reason}`);
        return null;
      }

      // 4. Generate professional trading signal
      const signal = this.createProfessionalTradingSignal(
        ensemblePrediction,
        symbol,
        timeframe,
        data[data.length - 1],
        signalQuality
      );

      logger.info(`✅ HIGH QUALITY ML SIGNAL: ${signal.signalType} with ${signal.confidence.toFixed(1)}% confidence`);
      return signal;

    } catch (error) {
      logger.error(`🚨 ML signal generation failed: ${error.message}`);
      return null;
    }
  }

  /**
   * Prepare features for the real ML models using proper feature engineering
   */
  private prepareMLFeatures(
    data: MarketData[],
    symbol: string,
    timeframe: string
  ): MLFeatures {
    const latest = data[data.length - 1];
    const previous = data[data.length - 2];
    
    // Real feature engineering for ML models
    return {
      // Historical price data (last 100 points for sequence models)
      price_data: data.slice(-100).map(d => ({
        open: d.open,
        high: d.high,
        low: d.low,
        close: d.close,
        volume: d.volume,
        timestamp: d.timestamp
      })),
      
      // Technical indicators (computed by the backend)
      technical_indicators: {
        rsi: latest.indicators?.rsi,
        macd: latest.indicators?.macd,
        macd_signal: latest.indicators?.macd_signal,
        macd_histogram: latest.indicators?.macd_histogram,
        ema_12: latest.indicators?.ema_12,
        ema_26: latest.indicators?.ema_26,
        bollinger_upper: latest.indicators?.bollinger_upper,
        bollinger_lower: latest.indicators?.bollinger_lower,
        bollinger_middle: latest.indicators?.bollinger_middle,
        atr: latest.indicators?.atr,
        volume_sma: latest.indicators?.volume_sma
      },
      
      // Market microstructure features
      market_structure: {
        spread: (latest.high - latest.low) / latest.close,
        price_velocity: previous ? (latest.close - previous.close) / previous.close : 0,
        volume_ratio: latest.indicators?.volume_sma ? 
          latest.volume / latest.indicators.volume_sma : 1.0,
        momentum: this.calculateMomentum(data.slice(-20)),
        volatility: this.calculateVolatility(data.slice(-20))
      },
      
      // Metadata for the ML system
      metadata: {
        symbol,
        timeframe,
        timestamp: latest.timestamp,
        data_points: data.length
      }
    };
  }

  /**
   * Call the real ensemble prediction API
   */
  private async getEnsemblePrediction(features: MLFeatures): Promise<EnsemblePrediction | null> {
    try {
      logger.info('🧠 Calling MultiModelEnsemble for prediction...');
      
      const response = await axios.post(
        `${this.ML_SERVICE_URL}/api/v1/ensemble/predict`,
        {
          features,
          config: {
            voting_method: 'confidence_weighted',
            min_models_required: this.MIN_MODELS_REQUIRED,
            confidence_threshold: this.CONFIDENCE_THRESHOLD,
            return_details: true
          }
        },
        {
          timeout: 30000, // 30 seconds for ML computation
          headers: { 'Content-Type': 'application/json' }
        }
      );

      if (response.data && typeof response.data.ensemble_prediction === 'number') {
        logger.info(`🎯 Ensemble prediction: ${response.data.ensemble_prediction.toFixed(3)} (confidence: ${response.data.ensemble_confidence.toFixed(3)})`);
        
        // Log individual model contributions
        if (response.data.individual_predictions) {
          Object.entries(response.data.individual_predictions).forEach(([model, pred]) => {
            const conf = response.data.individual_confidences[model] || 0;
            logger.info(`   📊 ${model}: ${(pred as number).toFixed(3)} (conf: ${conf.toFixed(3)})`);
          });
        }
        
        return response.data;
      } else {
        logger.warn('❌ Invalid ensemble response format');
        return null;
      }

    } catch (error) {
      if (error.code === 'ECONNREFUSED') {
        logger.warn('⚠️ ML service not available - ensure Python ML service is running on port 8001');
      } else {
        logger.warn(`⚠️ Ensemble prediction failed: ${error.message}`);
      }
      return null;
    }
  }

  /**
   * Validate signal quality using professional standards
   */
  private validateSignalQuality(ensemble: EnsemblePrediction): { isValid: boolean; reason?: string } {
    // 1. Check minimum confidence threshold
    if (ensemble.ensemble_confidence < this.CONFIDENCE_THRESHOLD) {
      return {
        isValid: false,
        reason: `Low confidence: ${ensemble.ensemble_confidence.toFixed(3)} < ${this.CONFIDENCE_THRESHOLD}`
      };
    }

    // 2. Check quality score
    if (ensemble.quality_score < this.QUALITY_THRESHOLD) {
      return {
        isValid: false,
        reason: `Low quality: ${ensemble.quality_score.toFixed(3)} < ${this.QUALITY_THRESHOLD}`
      };
    }

    // 3. Check minimum models participated
    const activeModels = Object.values(ensemble.model_status).filter(status => status).length;
    if (activeModels < this.MIN_MODELS_REQUIRED) {
      return {
        isValid: false,
        reason: `Insufficient models: ${activeModels} < ${this.MIN_MODELS_REQUIRED}`
      };
    }

    // 4. Check uncertainty level
    if (ensemble.uncertainty > 0.3) {
      return {
        isValid: false,
        reason: `High uncertainty: ${ensemble.uncertainty.toFixed(3)} > 0.3`
      };
    }

    // 5. Check prediction strength (distance from neutral)
    const predictionStrength = Math.abs(ensemble.ensemble_prediction - 0.5);
    if (predictionStrength < 0.15) {
      return {
        isValid: false,
        reason: `Weak signal: prediction too close to neutral (${ensemble.ensemble_prediction.toFixed(3)})`
      };
    }

    return { isValid: true };
  }

  /**
   * Create professional trading signal from ML prediction
   */
  private createProfessionalTradingSignal(
    ensemble: EnsemblePrediction,
    symbol: string,
    timeframe: string,
    currentData: MarketData,
    quality: { isValid: boolean; reason?: string }
  ): TradingSignalOutput {
    
    const prediction = ensemble.ensemble_prediction;
    const confidence = ensemble.ensemble_confidence;
    const currentPrice = currentData.close;
    
    // Determine signal type based on ML prediction
    let signalType: 'BUY' | 'SELL' | 'HOLD';
    if (prediction > 0.65 && confidence > this.CONFIDENCE_THRESHOLD) {
      signalType = 'BUY';
    } else if (prediction < 0.35 && confidence > this.CONFIDENCE_THRESHOLD) {
      signalType = 'SELL';
    } else {
      signalType = 'HOLD';
    }

    // Professional risk management calculations
    const volatility = currentData.indicators?.atr || (currentPrice * 0.02);
    const confidenceMultiplier = Math.min(1.5, confidence / 0.7); // Scale risk with confidence
    
    let stopLoss: number | undefined;
    let takeProfit: number | undefined;
    let riskRewardRatio: number | undefined;

    if (signalType !== 'HOLD') {
      if (signalType === 'BUY') {
        stopLoss = currentPrice - (volatility * 1.5 * confidenceMultiplier);
        takeProfit = currentPrice + (volatility * 3.0 * confidenceMultiplier); // Min 2:1 R:R
      } else {
        stopLoss = currentPrice + (volatility * 1.5 * confidenceMultiplier);
        takeProfit = currentPrice - (volatility * 3.0 * confidenceMultiplier);
      }
      
      const risk = Math.abs(currentPrice - stopLoss);
      const reward = Math.abs(takeProfit - currentPrice);
      riskRewardRatio = reward / risk;
    }

    // Signal strength based on ML confidence and prediction strength
    const predictionStrength = Math.abs(prediction - 0.5) * 2;
    const overallStrength = (predictionStrength + confidence) / 2;
    
    const strength: 'VERY_STRONG' | 'STRONG' | 'MODERATE' | 'WEAK' = 
      overallStrength > 0.9 ? 'VERY_STRONG' :
      overallStrength > 0.8 ? 'STRONG' :
      overallStrength > 0.7 ? 'MODERATE' : 'WEAK';

    // Generate detailed reasoning from ML results
    const reasoning = [
      `🤖 ML Ensemble Prediction: ${prediction.toFixed(3)} (${ensemble.voting_method} voting)`,
      `🎯 Confidence: ${(confidence * 100).toFixed(1)}% (threshold: ${(this.CONFIDENCE_THRESHOLD * 100).toFixed(0)}%)`,
      `⭐ Quality Score: ${(ensemble.quality_score * 100).toFixed(1)}% (threshold: ${(this.QUALITY_THRESHOLD * 100).toFixed(0)}%)`,
      `📊 Model Consensus: ${Object.values(ensemble.model_status).filter(s => s).length}/${Object.keys(ensemble.model_status).length} models active`,
      `🎲 Uncertainty: ${(ensemble.uncertainty * 100).toFixed(1)}% (max: 30%)`,
      `💰 Risk Management: ${(confidenceMultiplier * 100).toFixed(0)}% position scaling, ${riskRewardRatio?.toFixed(1)}:1 R:R`,
      `🔬 Top Models: ${Object.entries(ensemble.individual_confidences)
        .sort(([,a], [,b]) => b - a)
        .slice(0, 3)
        .map(([model, conf]) => `${model}(${(conf * 100).toFixed(0)}%)`)
        .join(', ')}`
    ];

    return {
      id: `ml_ensemble_${Date.now()}`,
      symbol,
      timeframe,
      signalType,
      confidence: confidence * 100, // Convert to percentage
      strength,
      entryPrice: currentPrice,
      stopLoss,
      takeProfit,
      riskRewardRatio,
      timestamp: new Date(),
      reasoning,
      metadata: {
        ml_model: 'MultiModelEnsemble v2.0',
        prediction: prediction,
        quality_score: ensemble.quality_score,
        ensemble_data: ensemble,
        source: 'real_machine_learning',
        model_performances: ensemble.individual_confidences
      }
    };
  }

  /**
   * Calculate momentum from price data
   */
  private calculateMomentum(data: MarketData[]): number {
    if (data.length < 2) return 0;
    const current = data[data.length - 1].close;
    const previous = data[0].close;
    return (current - previous) / previous;
  }

  /**
   * Calculate volatility (standard deviation of returns)
   */
  private calculateVolatility(data: MarketData[]): number {
    if (data.length < 2) return 0.02;
    
    const returns = data.slice(1).map((d, i) => 
      (d.close - data[i].close) / data[i].close
    );
    
    const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
    
    return Math.sqrt(variance);
  }

  /**
   * Calculate professional position size based on ML confidence
   */
  async calculatePositionSize(
    signal: TradingSignalOutput,
    accountBalance: number,
    baseRiskPerTrade: number = 0.015 // 1.5% base risk
  ): Promise<number> {
    if (!signal.stopLoss || !signal.entryPrice) return 0;
    
    // Risk amount scaled by ML confidence
    const confidenceScaling = Math.min(1.2, signal.confidence / 75); // Max 20% scaling
    const adjustedRiskPercent = baseRiskPerTrade * confidenceScaling;
    const riskAmount = accountBalance * adjustedRiskPercent;
    
    // Position size calculation
    const riskPerShare = Math.abs(signal.entryPrice - signal.stopLoss);
    const positionSize = riskPerShare > 0 ? riskAmount / riskPerShare : 0;
    
    logger.info(`💰 Position sizing: ${adjustedRiskPercent.toFixed(3)}% risk, $${riskAmount.toFixed(2)} amount, ${positionSize.toFixed(6)} size`);
    
    return positionSize;
  }

  /**
   * Get current ML system status
   */
  async getSystemStatus(): Promise<{ available: boolean; models: Record<string, boolean>; version: string }> {
    try {
      const response = await axios.get(`${this.ML_SERVICE_URL}/api/v1/status`, { timeout: 5000 });
      return {
        available: true,
        models: response.data.models || {},
        version: response.data.version || 'unknown'
      };
    } catch (error) {
      return {
        available: false,
        models: {},
        version: 'unavailable'
      };
    }
  }

  getStrategyName(): string {
    return 'Master Trading AI v2.0 - Real ML Ensemble';
  }

  getStrategyDescription(): string {
    return 'Professional machine learning trading system using real neural networks, ensemble methods, and multi-model consensus for institutional-grade signal generation. Features LSTM+Attention, CNN-LSTM, Transformer models with dynamic ensemble voting.';
  }
} 