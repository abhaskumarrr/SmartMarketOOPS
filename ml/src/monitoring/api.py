"""
Monitoring API

This module provides API endpoints for the model monitoring system.
"""

import os
import logging
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
from pydantic import BaseModel

# Import project modules
from .performance_tracker import get_tracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


class MetricsLog(BaseModel):
    """Model for logging metrics"""
    metrics: Dict[str, float]
    model_version: str
    timestamp: Optional[str] = None
    custom_metrics: Optional[Dict[str, Any]] = None


class PredictionLog(BaseModel):
    """Model for logging predictions"""
    actual: float
    predicted: float
    model_version: str
    horizon: Optional[int] = 1
    features: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None


class DriftMetricsRequest(BaseModel):
    """Model for requesting drift metrics"""
    reference_version: str
    current_version: str
    window_size: Optional[int] = 100


class ReportRequest(BaseModel):
    """Model for requesting performance reports"""
    model_version: Optional[str] = None
    last_n_days: Optional[int] = None


@router.post("/metrics/{symbol}", tags=["Monitoring"])
async def log_metrics(symbol: str, log_data: MetricsLog) -> Dict[str, Any]:
    """
    Log performance metrics for a model.
    """
    try:
        tracker = get_tracker(symbol)
        
        tracker.log_metrics(
            metrics=log_data.metrics,
            model_version=log_data.model_version,
            timestamp=log_data.timestamp,
            custom_metrics=log_data.custom_metrics
        )
        
        return {
            "status": "success",
            "message": f"Metrics logged for {symbol} (version: {log_data.model_version})"
        }
    except Exception as e:
        logger.error(f"Error logging metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error logging metrics: {str(e)}")


@router.post("/predictions/{symbol}", tags=["Monitoring"])
async def log_prediction(symbol: str, log_data: PredictionLog) -> Dict[str, Any]:
    """
    Log a prediction and actual value pair.
    """
    try:
        tracker = get_tracker(symbol)
        
        tracker.log_prediction(
            actual=log_data.actual,
            predicted=log_data.predicted,
            model_version=log_data.model_version,
            horizon=log_data.horizon,
            features=log_data.features,
            timestamp=log_data.timestamp
        )
        
        return {
            "status": "success", 
            "message": f"Prediction logged for {symbol} (version: {log_data.model_version})"
        }
    except Exception as e:
        logger.error(f"Error logging prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error logging prediction: {str(e)}")


@router.get("/metrics/{symbol}", tags=["Monitoring"])
async def get_metrics_history(
    symbol: str,
    model_version: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: Optional[int] = 100
) -> Dict[str, Any]:
    """
    Get historical performance metrics.
    """
    try:
        tracker = get_tracker(symbol)
        
        metrics_df = tracker.get_metrics_history(
            model_version=model_version,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        
        # Convert to list of dictionaries
        metrics_list = metrics_df.to_dict(orient='records')
        
        # Parse custom_metrics from JSON string
        for metrics in metrics_list:
            if 'custom_metrics' in metrics and metrics['custom_metrics']:
                import json
                try:
                    metrics['custom_metrics'] = json.loads(metrics['custom_metrics'])
                except:
                    pass
        
        return {
            "symbol": symbol,
            "model_version": model_version or "all",
            "count": len(metrics_list),
            "metrics": metrics_list
        }
    except Exception as e:
        logger.error(f"Error getting metrics history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting metrics history: {str(e)}")


@router.get("/predictions/{symbol}", tags=["Monitoring"])
async def get_predictions_history(
    symbol: str,
    model_version: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: Optional[int] = 100
) -> Dict[str, Any]:
    """
    Get historical predictions.
    """
    try:
        tracker = get_tracker(symbol)
        
        predictions_df = tracker.get_predictions_history(
            model_version=model_version,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        
        # Convert to list of dictionaries
        predictions_list = predictions_df.to_dict(orient='records')
        
        # Parse features_json from JSON string
        for prediction in predictions_list:
            if 'features_json' in prediction and prediction['features_json']:
                import json
                try:
                    prediction['features'] = json.loads(prediction['features_json'])
                    del prediction['features_json']
                except:
                    prediction['features'] = {}
            else:
                prediction['features'] = {}
        
        return {
            "symbol": symbol,
            "model_version": model_version or "all",
            "count": len(predictions_list),
            "predictions": predictions_list
        }
    except Exception as e:
        logger.error(f"Error getting predictions history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting predictions history: {str(e)}")


@router.post("/drift/{symbol}", tags=["Monitoring"])
async def calculate_drift(symbol: str, request: DriftMetricsRequest) -> Dict[str, Any]:
    """
    Calculate drift metrics between model versions.
    """
    try:
        tracker = get_tracker(symbol)
        
        drift_metrics = tracker.calculate_drift_metrics(
            reference_version=request.reference_version,
            current_version=request.current_version,
            window_size=request.window_size
        )
        
        return {
            "symbol": symbol,
            "drift_metrics": drift_metrics
        }
    except Exception as e:
        logger.error(f"Error calculating drift metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating drift metrics: {str(e)}")


@router.post("/report/{symbol}", tags=["Monitoring"])
async def generate_report(symbol: str, request: ReportRequest) -> Dict[str, Any]:
    """
    Generate a performance report.
    """
    try:
        tracker = get_tracker(symbol)
        
        # Create output directory for reports
        output_dir = os.path.join("reports", symbol.replace("/", "_"))
        os.makedirs(output_dir, exist_ok=True)
        
        report = tracker.generate_performance_report(
            model_version=request.model_version,
            output_dir=output_dir,
            last_n_days=request.last_n_days
        )
        
        # Add report file paths
        report['report_files'] = {
            'performance_chart': f"/reports/{symbol.replace('/', '_')}/{symbol.replace('/', '_')}_performance.png",
            'predictions_chart': f"/reports/{symbol.replace('/', '_')}/{symbol.replace('/', '_')}_predictions.png"
        }
        
        return {
            "symbol": symbol,
            "report": report
        }
    except Exception as e:
        logger.error(f"Error generating performance report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating performance report: {str(e)}") 