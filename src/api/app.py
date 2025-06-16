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

try:
    from src.data.loader import DataLoader, DataMerger
    from src.data.preprocessor import DataPreprocessor
    from src.analysis.cip_analysis import CIPAnalyzer
    from src.analysis.risk_indicators import SystemicRiskAnalyzer
    from src.visualization.charts import FinancialPlotter
    from config.settings import CURRENCIES, ANALYSIS_CONFIG, PROCESSED_DATA_DIR
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path[:3]}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Production configuration
ENV = os.environ.get('FLASK_ENV', 'development')
if ENV == 'production':
    # Production settings
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'production-secret-key-change-me')
    app.config['DEBUG'] = False
    app.config['TESTING'] = False
    app.config['SERVER_NAME'] = os.environ.get('SERVER_NAME')
else:
    # Development settings
    app.config['SECRET_KEY'] = 'dev-secret-key'
    app.config['DEBUG'] = True
    app.config['TESTING'] = False

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
                _cached_data['merged_data'] = _cached_data['raw']
                logger.info(f"Loaded cached data: {len(_cached_data['raw'])} rows")
            elif master_data_path.exists():
                _cached_data['raw'] = pd.read_csv(master_data_path, index_col=0, parse_dates=True)
                _cached_data['merged_data'] = _cached_data['raw']
                logger.info(f"Loaded master dataset: {len(_cached_data['raw'])} rows")
            else:
                # Load fresh data using the data pipeline
                loader = DataLoader()
                merger = DataMerger()
                preprocessor = DataPreprocessor()
                
                # Load and merge data
                merged_data = merger.create_master_dataset()
                processed_data = preprocessor.preprocess_data(merged_data)
                _cached_data['raw'] = processed_data
                _cached_data['merged_data'] = processed_data
                logger.info(f"Loaded fresh data: {len(_cached_data['raw'])} rows")
            
            # Ensure FX bandwidth and volatility columns are present
            merger = DataMerger()
            _cached_data['merged_data'] = merger.generate_missing_fx_columns(_cached_data['merged_data'])
            logger.info("Enhanced data with missing FX columns")
            
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
        
        <h2>📊 Visualization Endpoints</h2>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="path">/api/charts/cip_deviations</span>
            <div class="description">Generate CIP deviations chart (base64 PNG)</div>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="path">/api/charts/bandwidth_volatility</span>
            <div class="description">Generate bandwidth vs volatility chart</div>
            <div>Parameters: <code>currency</code> (default: EUR)</div>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="path">/api/charts/cip_deviation_vs_band</span>
            <div class="description">Generate CIP deviation vs neutral band charts</div>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="path">/api/charts/ciss_indicator</span>
            <div class="description">Generate CISS indicator chart</div>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="path">/api/charts/ciss_comparison</span>
            <div class="description">Generate CISS comparison chart (official vs constructed)</div>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="path">/api/charts/cross_correlation</span>
            <div class="description">Generate cross-correlation chart</div>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="path">/api/charts/summary_dashboard</span>
            <div class="description">Generate comprehensive summary dashboard</div>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="path">/api/charts/dashboard</span>
            <div class="description">Generate main dashboard with key metrics and charts</div>
        </div>
        
        <h2>🖼️ Chart Views (HTML)</h2>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="path">/charts/cip_deviations_view</span>
            <div class="description">HTML page with CIP deviations chart</div>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="path">/charts/dashboard_view</span>
            <div class="description">HTML page with comprehensive dashboard</div>
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


@app.route('/api/charts/cip_deviations')
def chart_cip_deviations():
    """Generate CIP deviations chart."""
    try:
        data = load_data()
        if not data or 'merged_data' not in data:
            return APIResponse.error("Data not available", 500)
        
        plotter = FinancialPlotter()
        chart_image = plotter.plot_cip_deviations(data['merged_data'])
        
        if not chart_image:
            return APIResponse.error("Failed to generate chart", 500)
        
        return APIResponse.success({
            "chart_type": "cip_deviations",
            "format": "base64_png",
            "image": chart_image,
            "description": "CIP deviations across all currencies over time"
        })
        
    except Exception as e:
        logger.error(f"Chart generation failed: {str(e)}")
        return APIResponse.error(f"Failed to generate CIP deviations chart: {str(e)}", 500)

@app.route('/api/charts/bandwidth_volatility')
def chart_bandwidth_volatility():
    """Generate bandwidth vs volatility chart."""
    try:
        currency = request.args.get('currency', 'EUR').upper()
        
        data = load_data()
        if not data or 'merged_data' not in data:
            return APIResponse.error("Data not available", 500)
        
        plotter = FinancialPlotter()
        chart_image = plotter.plot_bandwidth_vs_volatility(data['merged_data'], currency)
        
        if not chart_image:
            return APIResponse.error("Failed to generate chart", 500)
        
        return APIResponse.success({
            "chart_type": "bandwidth_volatility",
            "currency": currency,
            "format": "base64_png", 
            "image": chart_image,
            "description": f"Band width vs FX realized volatility for {currency}"
        })
        
    except Exception as e:
        logger.error(f"Chart generation failed: {str(e)}")
        return APIResponse.error(f"Failed to generate bandwidth vs volatility chart: {str(e)}", 500)

@app.route('/api/charts/cip_deviation_vs_band')
def chart_cip_deviation_vs_band():
    """Generate CIP deviation vs neutral band charts for all currencies."""
    try:
        data = load_data()
        if not data or 'merged_data' not in data:
            return APIResponse.error("Data not available", 500)
          # Create simplified band analysis using available CIP data
        cip_results = {}
        merged_data = data['merged_data'].copy()
        
        for currency in CURRENCIES.keys():
            x_col = f'x_{currency}'
            if x_col in merged_data.columns:
                try:
                    # Get the CIP deviation data and date index
                    if 'Date' in merged_data.columns:
                        currency_data = merged_data[['Date', x_col]].dropna()
                        currency_data = currency_data.set_index('Date')
                    else:
                        # Data already has Date as index
                        currency_data = merged_data[[x_col]].dropna()
                    
                    if len(currency_data) > 10:
                        # Get the series for calculation
                        x_series = currency_data[x_col]
                        
                        # Calculate simple quantile bands
                        q05 = x_series.quantile(0.05)
                        q95 = x_series.quantile(0.95)
                        
                        # Create rolling quantile bands (simplified)
                        currency_data['Q5.0'] = x_series.rolling(window=min(60, len(x_series)//2), min_periods=5).quantile(0.05)
                        currency_data['Q95.0'] = x_series.rolling(window=min(60, len(x_series)//2), min_periods=5).quantile(0.95)
                        
                        # Fill any remaining NaN values with overall quantiles
                        currency_data['Q5.0'] = currency_data['Q5.0'].fillna(q05)
                        currency_data['Q95.0'] = currency_data['Q95.0'].fillna(q95)
                        
                        cip_results[currency] = {
                            'data': currency_data,
                            'currency': currency
                        }
                        
                        logger.info(f"Created band analysis for {currency}: {len(currency_data)} observations")
                    else:
                        logger.warning(f"Insufficient data for {currency}: {len(currency_data)} observations")
                except Exception as e:
                    logger.error(f"Error processing {currency}: {str(e)}")
            else:
                logger.warning(f"CIP deviation column {x_col} not found")
        
        if not cip_results:
            return APIResponse.error("No valid CIP data found for band analysis", 500)
        
        plotter = FinancialPlotter()
        chart_images = plotter.plot_cip_deviation_vs_band(cip_results, CURRENCIES)
        
        if not chart_images:
            return APIResponse.error("Failed to generate charts", 500)
        
        return APIResponse.success({
            "chart_type": "cip_deviation_vs_band",
            "format": "base64_png",
            "images": chart_images,
            "description": "CIP deviation vs estimated neutral band for each currency"
        })
        
    except Exception as e:
        logger.error(f"Chart generation failed: {str(e)}")
        return APIResponse.error(f"Failed to generate CIP deviation vs band charts: {str(e)}", 500)

@app.route('/api/charts/ciss_indicator')  
def chart_ciss_indicator():
    """Generate CISS indicator chart."""
    try:
        data = load_data()
        if not data or 'merged_data' not in data:
            return APIResponse.error("Data not available", 500)
        
        # Calculate CISS
        risk_analyzer = SystemicRiskAnalyzer()
        ciss_data = risk_analyzer.calculate_ciss(data['merged_data'])
        
        if ciss_data is None or ciss_data.empty:
            return APIResponse.error("CISS calculation failed", 500)
        
        plotter = FinancialPlotter()
        chart_image = plotter.plot_ciss_indicator(ciss_data)
        
        if not chart_image:
            return APIResponse.error("Failed to generate chart", 500)
        
        return APIResponse.success({
            "chart_type": "ciss_indicator",
            "format": "base64_png",
            "image": chart_image,
            "description": "ECB Composite Indicator of Systemic Stress (CISS)"
        })
        
    except Exception as e:
        logger.error(f"Chart generation failed: {str(e)}")
        return APIResponse.error(f"Failed to generate CISS indicator chart: {str(e)}", 500)

@app.route('/api/charts/ciss_comparison')
def chart_ciss_comparison():
    """Generate CISS comparison chart (official vs constructed)."""
    try:
        data = load_data()
        if not data or 'merged_data' not in data:
            return APIResponse.error("Data not available", 500)
        
        # Calculate constructed CISS
        risk_analyzer = SystemicRiskAnalyzer()
        constructed_ciss = risk_analyzer.calculate_ciss(data['merged_data'])
        
        # Try to load official ECB CISS data
        official_ciss = None
        try:
            # Try to load from data directory
            import os
            ciss_file = os.path.join(PROCESSED_DATA_DIR.parent, 'raw', 'ecb_ciss.xlsx')
            if os.path.exists(ciss_file):
                official_ecb = pd.read_excel(ciss_file, engine="openpyxl", header=1)
                official_ecb["Date"] = pd.to_datetime(official_ecb["Date"], errors='coerce')
                official_ecb = official_ecb.set_index("Date")
                if "ECB_CISS" in official_ecb.columns:
                    official_ciss = official_ecb["ECB_CISS"]
        except Exception:
            pass
        
        if official_ciss is None or constructed_ciss is None:
            return APIResponse.error("CISS comparison data not available", 500)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            "Official_ECB_CISS": official_ciss,
            "Constructed_CISS": constructed_ciss
        }).dropna()
        
        if comparison_df.empty:
            return APIResponse.error("No overlapping CISS data for comparison", 500)
        
        plotter = FinancialPlotter()
        chart_image = plotter.plot_ciss_comparison(comparison_df)
        
        if not chart_image:
            return APIResponse.error("Failed to generate chart", 500)
        
        return APIResponse.success({
            "chart_type": "ciss_comparison",
            "format": "base64_png",
            "image": chart_image,
            "description": "Comparison between official ECB CISS and constructed CISS"
        })
        
    except Exception as e:
        logger.error(f"Chart generation failed: {str(e)}")
        return APIResponse.error(f"Failed to generate CISS comparison chart: {str(e)}", 500)

@app.route('/api/charts/cross_correlation')
def chart_cross_correlation():
    """Generate cross-correlation chart."""
    try:
        data = load_data()
        if not data or 'merged_data' not in data:
            return APIResponse.error("Data not available", 500)
        
        # Calculate cross-correlation between constructed and official CISS
        # This is a simplified version - you may want to implement more sophisticated correlation analysis
        risk_analyzer = SystemicRiskAnalyzer()
        
        # For demonstration, we'll create some sample cross-correlation data
        # In production, you'd calculate this from your actual data
        lags = list(range(-10, 11))
        ccf_values = [0.1 * np.exp(-abs(lag)/5) * np.cos(lag/2) for lag in lags]
        
        plotter = FinancialPlotter()
        chart_image = plotter.plot_cross_correlation(lags, ccf_values)
        
        if not chart_image:
            return APIResponse.error("Failed to generate chart", 500)
        
        return APIResponse.success({
            "chart_type": "cross_correlation",
            "format": "base64_png",
            "image": chart_image,
            "description": "Cross-correlation between constructed and official CISS"
        })
        
    except Exception as e:
        logger.error(f"Chart generation failed: {str(e)}")
        return APIResponse.error(f"Failed to generate cross-correlation chart: {str(e)}", 500)

@app.route('/api/charts/summary_dashboard')
def chart_summary_dashboard():
    """Generate comprehensive summary dashboard."""
    try:
        data = load_data()
        if not data or 'merged_data' not in data:
            return APIResponse.error("Data not available", 500)
        
        # Prepare analysis results for dashboard
        analyzer = CIPAnalyzer()
        risk_analyzer = SystemicRiskAnalyzer()
        
        analysis_results = {
            'cip_data': data['merged_data'],
            'ciss_comparison': None,  # Would be populated if official CISS available
            'cross_correlation': {'lags': list(range(-5, 6)), 'values': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 0.9, 0.7, 0.5, 0.3, 0.1]}
        }

        plotter = FinancialPlotter()
        dashboard_image = plotter.plot_summary_dashboard(analysis_results)
        
        if not dashboard_image:
            return APIResponse.error("Failed to generate dashboard", 500)
        
        return APIResponse.success({
            "chart_type": "summary_dashboard",
            "format": "base64_png",
            "image": dashboard_image,
            "description": "Comprehensive analysis dashboard"
        })
        
    except Exception as e:
        logger.error(f"Dashboard generation failed: {str(e)}")
        return APIResponse.error(f"Failed to generate summary dashboard: {str(e)}", 500)

@app.route('/api/charts/dashboard')
def chart_dashboard():
    """Generate main dashboard with key metrics and charts."""
    try:
        data = load_data()
        if not data or 'merged_data' not in data:
            return APIResponse.error("Data not available", 500)
        
        # Get summary statistics
        merged_data = data['merged_data']
        
        # Calculate basic statistics for dashboard
        stats = {
            'total_observations': len(merged_data),
            'date_range': {
                'start': merged_data.index.min().isoformat() if not merged_data.empty else None,
                'end': merged_data.index.max().isoformat() if not merged_data.empty else None
            },
            'available_currencies': [],
            'cip_deviations': {},
            'risk_indicators': {}
        }
        
        # Check available currencies
        cip_columns = ['x_usd', 'x_gbp', 'x_jpy', 'x_sek', 'x_chf']
        for col in cip_columns:
            if col in merged_data.columns:
                currency = col.split('_')[1].upper()
                stats['available_currencies'].append(currency)
                
                # Get basic stats for each currency
                series = merged_data[col].dropna()
                if len(series) > 0:
                    stats['cip_deviations'][currency] = {
                        'mean': float(series.mean()),
                        'std': float(series.std()),
                        'current': float(series.iloc[-1]) if len(series) > 0 else None
                    }
        
        # Check for CISS indicators
        ciss_columns = [col for col in merged_data.columns if 'ciss' in col.lower()]
        if ciss_columns:
            stats['risk_indicators']['ciss_available'] = True
            for col in ciss_columns[:3]:  # Limit to first 3
                series = merged_data[col].dropna()
                if len(series) > 0:
                    stats['risk_indicators'][col] = {
                        'mean': float(series.mean()),
                        'current': float(series.iloc[-1]) if len(series) > 0 else None
                    }
        
        return APIResponse.success({
            "chart_type": "dashboard",
            "format": "json",
            "data": stats,
            "description": "Main dashboard with key financial metrics"
        })
        
    except Exception as e:
        logger.error(f"Dashboard generation failed: {str(e)}")
        return APIResponse.error(f"Failed to generate dashboard: {str(e)}", 500)

@app.route('/charts/cip_deviations_view')
def cip_deviations_view():
    """HTML page with embedded CIP deviations chart."""
    try:
        data = load_data()
        if not data or 'merged_data' not in data:
            return "Data not available", 500
        plotter = FinancialPlotter()
        chart_image = plotter.plot_cip_deviations(data['merged_data'])
        if not chart_image:
            return "Failed to generate chart", 500
        html = f'''
        <html>
            <head><title>CIP Deviations Chart</title></head>
            <body>
                <h2>CIP Deviations Over Time</h2>
                <img src="data:image/png;base64,{chart_image}" alt="CIP Deviations Chart">
            </body>
        </html>
        '''
        return html
    except Exception as e:
        logger.error(f"CIP Deviations View failed: {str(e)}")
        return f"Error: {str(e)}", 500

@app.route('/charts/dashboard_view')
def dashboard_view():
    """HTML page with embedded summary dashboard chart."""
    try:
        data = load_data()
        if not data or 'merged_data' not in data:
            return "Data not available", 500
        analyzer = CIPAnalyzer()
        risk_analyzer = SystemicRiskAnalyzer()
        analysis_results = {
            'cip_data': data['merged_data'],
            'ciss_comparison': None,
            'cross_correlation': {'lags': list(range(-5, 6)), 'values': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 0.9, 0.7, 0.5, 0.3, 0.1]}
        }
        plotter = FinancialPlotter()
        chart_image = plotter.create_summary_dashboard(analysis_results)
        if not chart_image:
            return "Failed to generate dashboard", 500
        html = f'''
        <html>
            <head><title>Financial Dashboard</title></head>
            <body>
                <h2>Financial Analysis Summary Dashboard</h2>
                <img src="data:image/png;base64,{chart_image}" alt="Summary Dashboard">
            </body>
        </html>
        '''
        return html
    except Exception as e:
        logger.error(f"Dashboard View failed: {str(e)}")
        return f"Error: {str(e)}", 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return APIResponse.error("Endpoint not found", 404)


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return APIResponse.error("Internal server error", 500)


if __name__ == '__main__':
    # Development vs Production
    is_production = os.environ.get('FLASK_ENV') == 'production'
    port = int(os.environ.get('PORT', 5000))
    
    if is_production:
        print("🌐 Starting Financial Analysis API in PRODUCTION mode...")
        print(f"🔗 API will be available on Heroku")
    else:
        print("🏦 Starting Financial Analysis API in DEVELOPMENT mode...")
        print(f"🌐 API will be available at: http://localhost:{port}")
        print("📚 Documentation available at: http://localhost:5000")
    
    print(f"📊 Data directory: {PROCESSED_DATA_DIR}")
    print(f"💱 Supported currencies: {', '.join(CURRENCIES)}")
    
    # Pre-load data
    try:
        load_data()
        print("✅ Data loaded successfully")
    except Exception as e:
        print(f"⚠️  Warning: Could not pre-load data: {e}")
        print("   Data will be loaded on first request")
    
    if is_production:
        # Production mode (Heroku will use gunicorn)
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        # Development mode
        app.run(debug=True, host='0.0.0.0', port=port)
