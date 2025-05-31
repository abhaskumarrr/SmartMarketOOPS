import { NextApiRequest, NextApiResponse } from 'next';

interface SignalQuality {
  symbol: string;
  prediction: number;
  confidence: number;
  quality_score: number;
  signal_valid: boolean;
  market_regime: string;
  recommendation: string;
  timestamp: string;
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<SignalQuality[] | { error: string }>
) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const mlServiceUrl = process.env.ML_SERVICE_URL || 'http://localhost:8000';
    
    const response = await fetch(`${mlServiceUrl}/api/monitoring/signals`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      // Return mock signal data if ML service is not available
      const mockSignals: SignalQuality[] = [
        {
          symbol: 'BTCUSDT',
          prediction: 0.785,
          confidence: 0.842,
          quality_score: 0.756,
          signal_valid: true,
          market_regime: 'trending_bullish',
          recommendation: 'BUY',
          timestamp: new Date(Date.now() - 30000).toISOString()
        },
        {
          symbol: 'ETHUSDT',
          prediction: 0.623,
          confidence: 0.712,
          quality_score: 0.689,
          signal_valid: true,
          market_regime: 'volatile',
          recommendation: 'HOLD',
          timestamp: new Date(Date.now() - 60000).toISOString()
        },
        {
          symbol: 'SOLUSDT',
          prediction: 0.456,
          confidence: 0.598,
          quality_score: 0.534,
          signal_valid: false,
          market_regime: 'ranging',
          recommendation: 'NEUTRAL',
          timestamp: new Date(Date.now() - 90000).toISOString()
        },
        {
          symbol: 'ADAUSDT',
          prediction: 0.734,
          confidence: 0.823,
          quality_score: 0.778,
          signal_valid: true,
          market_regime: 'breakout_bullish',
          recommendation: 'STRONG_BUY',
          timestamp: new Date(Date.now() - 120000).toISOString()
        },
        {
          symbol: 'BTCUSDT',
          prediction: 0.345,
          confidence: 0.756,
          quality_score: 0.612,
          signal_valid: true,
          market_regime: 'trending_bearish',
          recommendation: 'SELL',
          timestamp: new Date(Date.now() - 150000).toISOString()
        },
        {
          symbol: 'ETHUSDT',
          prediction: 0.567,
          confidence: 0.634,
          quality_score: 0.598,
          signal_valid: false,
          market_regime: 'consolidation',
          recommendation: 'HOLD',
          timestamp: new Date(Date.now() - 180000).toISOString()
        },
        {
          symbol: 'SOLUSDT',
          prediction: 0.812,
          confidence: 0.889,
          quality_score: 0.834,
          signal_valid: true,
          market_regime: 'breakout_bullish',
          recommendation: 'STRONG_BUY',
          timestamp: new Date(Date.now() - 210000).toISOString()
        },
        {
          symbol: 'ADAUSDT',
          prediction: 0.423,
          confidence: 0.567,
          quality_score: 0.489,
          signal_valid: false,
          market_regime: 'volatile',
          recommendation: 'NEUTRAL',
          timestamp: new Date(Date.now() - 240000).toISOString()
        },
        {
          symbol: 'BTCUSDT',
          prediction: 0.698,
          confidence: 0.745,
          quality_score: 0.723,
          signal_valid: true,
          market_regime: 'trending_bullish',
          recommendation: 'BUY',
          timestamp: new Date(Date.now() - 270000).toISOString()
        },
        {
          symbol: 'ETHUSDT',
          prediction: 0.289,
          confidence: 0.678,
          quality_score: 0.534,
          signal_valid: true,
          market_regime: 'trending_bearish',
          recommendation: 'SELL',
          timestamp: new Date(Date.now() - 300000).toISOString()
        }
      ];
      
      return res.status(200).json(mockSignals);
    }

    const data = await response.json();
    res.status(200).json(data);
  } catch (error) {
    console.error('Error fetching signal quality:', error);
    
    // Return mock data on error
    const mockSignals: SignalQuality[] = [
      {
        symbol: 'BTCUSDT',
        prediction: 0.785,
        confidence: 0.842,
        quality_score: 0.756,
        signal_valid: true,
        market_regime: 'trending_bullish',
        recommendation: 'BUY',
        timestamp: new Date().toISOString()
      },
      {
        symbol: 'ETHUSDT',
        prediction: 0.623,
        confidence: 0.712,
        quality_score: 0.689,
        signal_valid: true,
        market_regime: 'volatile',
        recommendation: 'HOLD',
        timestamp: new Date().toISOString()
      }
    ];
    
    res.status(200).json(mockSignals);
  }
}
