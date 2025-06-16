"""
Flask API for Financial Analysis
Provides REST endpoints for CIP analysis and risk indicators.
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import traceback
import json
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import DataLoader, DataMerger
from src.data.preprocessor import DataPreprocessor
from src.analysis.cip_analysis import CIPAnalyzer
from src.analysis.risk_indicators import SystemicRiskAnalyzer
from src.visualization.charts import FinancialPlotter
from config.settings import CURRENCIES, ANALYSIS_CONFIG, PROCESSED_DATA_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global data storage
_cached_data = {}
_last_load_time = None
CACHE_DURATION = timedelta(hours=1)  # Cache data for 1 hour


class APIResponse:
    """Standardized API response helper."""
    
    @staticmethod
    def success(data, message="Success"):
        return jsonify({
            "status": "success",
            "message": message,
            "data": data,
            "timestamp": datetime.now().isoformat()
        })
    
    @staticmethod
    def error(message, error_code=400, details=None):
        response = {
            "status": "error",
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        if details:
            response["details"] = details
        return jsonify(response), error_code


def load_data():
    """Load and cache data if needed."""
    global _cached_data, _last_load_time
    
    current_time = datetime.now()
    
    # Check if we need to reload data
    if (_last_load_time is None or 
        current_time - _last_load_time > CACHE_DURATION or 
        not _cached_data):
        
        try:
            logger.info("Loading financial data...")
            
            # Load processed data if available
            master_data_path = PROCESSED_DATA_DIR / "master_dataset.csv"
            final_data_path = PROCESSED_DATA_DIR / "final_merged_data.csv"
            
            if final_data_path.exists():
                _cached_data['raw'] = pd.read_csv(final_data_path, index_col=0, parse_dates=True)
                logger.info(f"Loaded cached data: {len(_cached_data['raw'])} rows")
            elif master_data_path.exists():
                _cached_data['raw'] = pd.read_csv(master_data_path, index_col=0, parse_dates=True)
                logger.info(f"Loaded master dataset: {len(_cached_data['raw'])} rows")
            else:
                # Load fresh data using the data pipeline
                loader = DataLoader()
                merger = DataMerger(loader)
                preprocessor = DataPreprocessor()
                
                # Load and merge data
                merged_data = merger.merge_all_data()
                processed_data = preprocessor.preprocess_data(merged_data)
                _cached_data['raw'] = processed_data
                logger.info(f"Loaded fresh data: {len(_cached_data['raw'])} rows")
            
            # Initialize analyzers
            _cached_data['cip_analyzer'] = CIPAnalyzer()
            _cached_data['risk_analyzer'] = SystemicRiskAnalyzer()
            _cached_data['plotter'] = FinancialPlotter()
            
            _last_load_time = current_time
            logger.info("Data loading completed successfully")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    return _cached_data


@app.route('/')
def home():
    """API home page with documentation."""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Financial Analysis API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #3498db; }
            .method { color: #27ae60; font-weight: bold; }
            .path { color: #e74c3c; font-family: monospace; }
            .description { color: #7f8c8d; margin-top: 5px; }
            code { background: #ecf0f1; padding: 2px 4px; font-family: monospace; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🏦 Financial Analysis API</h1>
            <p>RESTful API for Covered Interest Parity analysis and systemic risk indicators</p>
        </div>
        
        <h2>📊 Available Endpoints</h2>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="path">/api/status</span>
            <div class="description">Check API status and data availability</div>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="path">/api/data/summary</span>
            <div class="description">Get summary statistics of the dataset</div>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="path">/api/data/currencies</span>
            <div class="description">List available currencies and their data ranges</div>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="path">/api/cip/deviations</span>
            <div class="description">Calculate CIP deviations for all currencies</div>
            <div>Parameters: <code>start_date</code>, <code>end_date</code>, <code>currency</code></div>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="path">/api/cip/analysis/{currency}</span>
            <div class="description">Detailed CIP analysis for specific currency</div>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="path">/api/risk/indicators</span>
            <div class="description">Calculate systemic risk indicators (CISS methodology)</div>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="path">/api/risk/ciss</span>
            <div class="description">Calculate Composite Indicator of Systemic Stress</div>
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> <span class="path">/api/analysis/custom</span>
            <div class="description">Run custom analysis with specified parameters</div>
        </div>
        
        <h2>📈 Example Usage</h2>
        <pre>
# Get CIP deviations for EUR/USD
curl "{{ request.url_root }}api/cip/deviations?currency=EUR&start_date=2020-01-01&end_date=2021-01-01"

# Get systemic risk indicators
curl "{{ request.url_root }}api/risk/indicators"

# Check API status
curl "{{ request.url_root }}api/status"
        </pre>
        
        <h2>📋 Data Coverage</h2>
        <p>This API provides access to financial data covering:</p>
        <ul>
            <li><strong>Currencies:</strong> EUR, USD, GBP, JPY, CHF, SEK</li>
            <li><strong>Time Period:</strong> 1999-2025 (26+ years)</li>
            <li><strong>Data Points:</strong> 6,876+ observations</li>
            <li><strong>Indicators:</strong> Forward rates, spot rates, interest rates, market stress indicators</li>
        </ul>
    </body>
    </html>
    """
    return render_template_string(html_template)


@app.route('/api/status')
def api_status():
    """Get API status and data information."""
    try:
        data = load_data()
        raw_data = data['raw']
        
        status_info = {
            "api_version": "1.0.0",
            "status": "operational",
            "data_loaded": True,
            "data_shape": raw_data.shape,
            "date_range": {
                "start": raw_data.index.min().isoformat(),
                "end": raw_data.index.max().isoformat()
            },
            "currencies": CURRENCIES,
            "last_update": _last_load_time.isoformat() if _last_load_time else None,
            "cache_status": "active" if _cached_data else "empty"
        }
        
        return APIResponse.success(status_info, "API is operational")
        
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        return APIResponse.error(f"API status check failed: {str(e)}", 500)


@app.route('/api/data/summary')
def data_summary():
    """Get summary statistics of the dataset."""
    try:
        data = load_data()
        raw_data = data['raw']
        
        # Basic summary
        summary = {
            "total_observations": len(raw_data),
            "date_range": {
                "start": raw_data.index.min().isoformat(),
                "end": raw_data.index.max().isoformat(),
                "days": (raw_data.index.max() - raw_data.index.min()).days
            },
            "columns": {
                "total": len(raw_data.columns),
                "names": raw_data.columns.tolist()
            },
            "missing_data": {
                "total_missing": int(raw_data.isnull().sum().sum()),
                "missing_by_column": raw_data.isnull().sum().to_dict()
            },
            "data_types": raw_data.dtypes.astype(str).to_dict()
        }
        
        # Add descriptive statistics for numeric columns
        numeric_cols = raw_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            desc_stats = raw_data[numeric_cols].describe()
            summary["descriptive_statistics"] = desc_stats.to_dict()
        
        return APIResponse.success(summary, "Data summary generated successfully")
        
    except Exception as e:
        logger.error(f"Data summary failed: {str(e)}")
        return APIResponse.error(f"Failed to generate data summary: {str(e)}", 500)


@app.route('/api/data/currencies')
def currencies_info():
    """Get information about available currencies."""
    try:
        data = load_data()
        raw_data = data['raw']
        
        currencies_info = {}
        
        for currency in CURRENCIES:
            # Find columns related to this currency
            currency_cols = [col for col in raw_data.columns 
                           if currency.lower() in col.lower()]
            
            if currency_cols:
                currency_data = raw_data[currency_cols].dropna()
                
                currencies_info[currency] = {
                    "available_columns": currency_cols,
                    "data_points": len(currency_data),
                    "date_range": {
                        "start": currency_data.index.min().isoformat() if len(currency_data) > 0 else None,
                        "end": currency_data.index.max().isoformat() if len(currency_data) > 0 else None
                    },
                    "completeness": round(len(currency_data) / len(raw_data) * 100, 2) if len(raw_data) > 0 else 0
                }
        
        return APIResponse.success(currencies_info, "Currency information retrieved successfully")
        
    except Exception as e:
        logger.error(f"Currency info failed: {str(e)}")
        return APIResponse.error(f"Failed to get currency information: {str(e)}", 500)


@app.route('/api/cip/deviations')
def cip_deviations():
    """Calculate CIP deviations for specified parameters."""
    try:
        data = load_data()
        raw_data = data['raw']
        cip_analyzer = data['cip_analyzer']
        
        # Get query parameters
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        currency = request.args.get('currency', '').upper()
        
        # Filter data by date range if specified
        filtered_data = raw_data.copy()
        if start_date:
            filtered_data = filtered_data[filtered_data.index >= start_date]
        if end_date:
            filtered_data = filtered_data[filtered_data.index <= end_date]
        
        # Calculate CIP deviations
        cip_data = cip_analyzer.calculate_cip_deviations(filtered_data)
        
        # Filter by currency if specified
        if currency and currency in CURRENCIES:
            deviation_cols = [col for col in cip_data.columns 
                            if f'cip_deviation_{currency.lower()}' in col.lower()]
            if deviation_cols:
                cip_data = cip_data[deviation_cols]
            else:
                return APIResponse.error(f"No CIP deviation data found for currency: {currency}")
        
        # Prepare response
        result = {
            "date_range": {
                "start": cip_data.index.min().isoformat(),
                "end": cip_data.index.max().isoformat()
            },
            "observations": len(cip_data),
            "currencies": currency if currency else "all",
            "deviations": {}
        }
        
        # Add deviation statistics
        for col in cip_data.columns:
            if 'cip_deviation' in col.lower():
                series = cip_data[col].dropna()
                if len(series) > 0:
                    result["deviations"][col] = {
                        "mean": float(series.mean()),
                        "std": float(series.std()),
                        "min": float(series.min()),
                        "max": float(series.max()),
                        "latest": float(series.iloc[-1]) if len(series) > 0 else None,
                        "data_points": len(series)
                    }
        
        # Add recent data if requested
        if request.args.get('include_data', '').lower() == 'true':
            # Return last 100 observations
            recent_data = cip_data.tail(100)
            result["recent_data"] = recent_data.to_dict('index')
        
        return APIResponse.success(result, "CIP deviations calculated successfully")
        
    except Exception as e:
        logger.error(f"CIP deviations calculation failed: {str(e)}")
        return APIResponse.error(f"Failed to calculate CIP deviations: {str(e)}", 500)


@app.route('/api/cip/analysis/<currency>')
def cip_analysis_currency(currency):
    """Detailed CIP analysis for specific currency."""
    try:
        currency = currency.upper()
        if currency not in CURRENCIES:
            return APIResponse.error(f"Currency {currency} not supported. Available: {CURRENCIES}")
        
        data = load_data()
        raw_data = data['raw']
        cip_analyzer = data['cip_analyzer']
        
        # Calculate comprehensive CIP analysis
        cip_data = cip_analyzer.calculate_cip_deviations(raw_data)
        
        # Find relevant columns for this currency
        currency_cols = [col for col in cip_data.columns 
                        if currency.lower() in col.lower()]
        
        if not currency_cols:
            return APIResponse.error(f"No data found for currency: {currency}")
        
        currency_data = cip_data[currency_cols].dropna()
        
        # Calculate analysis metrics
        analysis_result = {
            "currency": currency,
            "analysis_period": {
                "start": currency_data.index.min().isoformat(),
                "end": currency_data.index.max().isoformat(),
                "observations": len(currency_data)
            },
            "metrics": {}
        }
        
        # Calculate metrics for each relevant column
        for col in currency_cols:
            series = currency_data[col].dropna()
            if len(series) > 0:
                analysis_result["metrics"][col] = {
                    "descriptive_stats": {
                        "mean": float(series.mean()),
                        "median": float(series.median()),
                        "std": float(series.std()),
                        "min": float(series.min()),
                        "max": float(series.max()),
                        "skewness": float(series.skew()),
                        "kurtosis": float(series.kurtosis())
                    },
                    "recent_values": series.tail(10).to_dict(),
                    "volatility": {
                        "rolling_30d": series.rolling(30).std().tail(1).iloc[0] if len(series) >= 30 else None,
                        "rolling_90d": series.rolling(90).std().tail(1).iloc[0] if len(series) >= 90 else None
                    }
                }
        
        return APIResponse.success(analysis_result, f"CIP analysis for {currency} completed successfully")
        
    except Exception as e:
        logger.error(f"CIP analysis for {currency} failed: {str(e)}")
        return APIResponse.error(f"Failed to analyze CIP for {currency}: {str(e)}", 500)


@app.route('/api/risk/indicators')
def risk_indicators():
    """Calculate systemic risk indicators."""
    try:
        data = load_data()
        raw_data = data['raw']
        risk_analyzer = data['risk_analyzer']
        
        # Calculate systemic risk indicators
        risk_data = risk_analyzer.create_market_blocks(raw_data)
        
        # Calculate CISS if possible
        try:
            ciss_data = risk_analyzer.calculate_ciss(risk_data)
            has_ciss = True
        except Exception as e:
            logger.warning(f"CISS calculation failed: {str(e)}")
            ciss_data = None
            has_ciss = False
        
        # Prepare response
        result = {
            "date_range": {
                "start": risk_data.index.min().isoformat(),
                "end": risk_data.index.max().isoformat()
            },
            "observations": len(risk_data),
            "has_ciss": has_ciss,
            "market_blocks": {},
            "risk_indicators": {}
        }
        
        # Analyze market blocks
        block_cols = [col for col in risk_data.columns if 'block' in col.lower()]
        for col in block_cols:
            series = risk_data[col].dropna()
            if len(series) > 0:
                result["market_blocks"][col] = {
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "latest": float(series.iloc[-1]),
                    "data_points": len(series)
                }
        
        # Add CISS data if available
        if has_ciss and ciss_data is not None:
            ciss_series = ciss_data['ciss'] if isinstance(ciss_data, dict) else ciss_data
            if isinstance(ciss_series, pd.Series) and len(ciss_series) > 0:
                result["ciss"] = {
                    "latest_value": float(ciss_series.iloc[-1]),
                    "mean": float(ciss_series.mean()),
                    "std": float(ciss_series.std()),
                    "percentile_95": float(ciss_series.quantile(0.95)),
                    "recent_trend": ciss_series.tail(10).to_dict()
                }
        
        return APIResponse.success(result, "Risk indicators calculated successfully")
        
    except Exception as e:
        logger.error(f"Risk indicators calculation failed: {str(e)}")
        return APIResponse.error(f"Failed to calculate risk indicators: {str(e)}", 500)


@app.route('/api/risk/ciss')
def ciss_indicator():
    """Calculate Composite Indicator of Systemic Stress (CISS)."""
    try:
        data = load_data()
        raw_data = data['raw']
        risk_analyzer = data['risk_analyzer']
        
        # Calculate market blocks first
        risk_data = risk_analyzer.create_market_blocks(raw_data)
        
        # Calculate CISS
        ciss_result = risk_analyzer.calculate_ciss(risk_data)
        
        if isinstance(ciss_result, dict):
            ciss_series = ciss_result.get('ciss')
            blocks_data = ciss_result.get('blocks', {})
        else:
            ciss_series = ciss_result
            blocks_data = {}
        
        if ciss_series is None or len(ciss_series) == 0:
            return APIResponse.error("CISS calculation returned no data")
        
        # Prepare detailed CISS response
        result = {
            "ciss_overview": {
                "latest_value": float(ciss_series.iloc[-1]),
                "date": ciss_series.index[-1].isoformat(),
                "mean": float(ciss_series.mean()),
                "std": float(ciss_series.std()),
                "min": float(ciss_series.min()),
                "max": float(ciss_series.max())
            },
            "stress_levels": {
                "current": "High" if ciss_series.iloc[-1] > ciss_series.quantile(0.8) else 
                          "Medium" if ciss_series.iloc[-1] > ciss_series.quantile(0.6) else "Low",
                "percentiles": {
                    "p50": float(ciss_series.quantile(0.5)),
                    "p75": float(ciss_series.quantile(0.75)),
                    "p90": float(ciss_series.quantile(0.9)),
                    "p95": float(ciss_series.quantile(0.95))
                }
            },
            "time_series": {
                "observations": len(ciss_series),
                "date_range": {
                    "start": ciss_series.index.min().isoformat(),
                    "end": ciss_series.index.max().isoformat()
                }
            }
        }
        
        # Add recent data
        if request.args.get('include_data', '').lower() == 'true':
            result["recent_data"] = ciss_series.tail(50).to_dict()
        
        # Add block contributions if available
        if blocks_data:
            result["block_contributions"] = blocks_data
        
        return APIResponse.success(result, "CISS calculated successfully")
        
    except Exception as e:
        logger.error(f"CISS calculation failed: {str(e)}")
        return APIResponse.error(f"Failed to calculate CISS: {str(e)}", 500)


@app.route('/api/analysis/custom', methods=['POST'])
def custom_analysis():
    """Run custom analysis with specified parameters."""
    try:
        # Get request data
        request_data = request.get_json()
        
        if not request_data:
            return APIResponse.error("No JSON data provided")
        
        analysis_type = request_data.get('analysis_type', '').lower()
        parameters = request_data.get('parameters', {})
        
        data = load_data()
        raw_data = data['raw']
        
        result = {"analysis_type": analysis_type, "parameters": parameters}
        
        if analysis_type == 'correlation':
            # Calculate correlation matrix
            numeric_cols = raw_data.select_dtypes(include=[np.number]).columns
            selected_cols = parameters.get('columns', numeric_cols[:10])  # Limit to first 10 if not specified
            
            correlation_matrix = raw_data[selected_cols].corr()
            result["correlation_matrix"] = correlation_matrix.to_dict()
            
        elif analysis_type == 'volatility':
            # Calculate rolling volatility
            window = parameters.get('window', 30)
            columns = parameters.get('columns', [])
            
            if not columns:
                return APIResponse.error("Columns parameter required for volatility analysis")
            
            volatility_data = {}
            for col in columns:
                if col in raw_data.columns:
                    series = raw_data[col].dropna()
                    if len(series) > window:
                        vol = series.rolling(window).std()
                        volatility_data[col] = {
                            "latest": float(vol.iloc[-1]) if not vol.empty else None,
                            "mean": float(vol.mean()) if not vol.empty else None,
                            "recent_values": vol.tail(10).to_dict()
                        }
            
            result["volatility_analysis"] = volatility_data
            
        elif analysis_type == 'summary_stats':
            # Calculate summary statistics
            columns = parameters.get('columns', raw_data.select_dtypes(include=[np.number]).columns[:5])
            
            summary_stats = {}
            for col in columns:
                if col in raw_data.columns:
                    series = raw_data[col].dropna()
                    if len(series) > 0:
                        summary_stats[col] = {
                            "count": len(series),
                            "mean": float(series.mean()),
                            "std": float(series.std()),
                            "min": float(series.min()),
                            "max": float(series.max()),
                            "median": float(series.median()),
                            "q25": float(series.quantile(0.25)),
                            "q75": float(series.quantile(0.75))
                        }
            
            result["summary_statistics"] = summary_stats
            
        else:
            return APIResponse.error(f"Unsupported analysis type: {analysis_type}")
        
        return APIResponse.success(result, "Custom analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Custom analysis failed: {str(e)}")
        return APIResponse.error(f"Custom analysis failed: {str(e)}", 500)


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return APIResponse.error("Endpoint not found", 404)


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return APIResponse.error("Internal server error", 500)


if __name__ == '__main__':
    print("🏦 Starting Financial Analysis API...")
    print(f"📊 Data directory: {PROCESSED_DATA_DIR}")
    print(f"💱 Supported currencies: {', '.join(CURRENCIES)}")
    print("🌐 API will be available at: http://localhost:5000")
    print("📚 Documentation available at: http://localhost:5000")
    
    # Pre-load data
    try:
        load_data()
        print("✅ Data loaded successfully")
    except Exception as e:
        print(f"⚠️  Warning: Could not pre-load data: {e}")
        print("   Data will be loaded on first request")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
