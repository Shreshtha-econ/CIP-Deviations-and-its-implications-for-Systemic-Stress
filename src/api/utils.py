"""
API Utilities
Helper functions and classes for the Flask API.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import json
from flask import request
import logging

logger = logging.getLogger(__name__)


class DataSerializer:
    """Handles serialization of complex data types for JSON responses."""
    
    @staticmethod
    def serialize_pandas(obj):
        """Convert pandas objects to JSON-serializable formats."""
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, (pd.Int64Dtype, pd.Float64Dtype)):
            return str(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    @staticmethod
    def serialize_datetime(obj):
        """Convert datetime objects to ISO format."""
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        return obj
    
    @classmethod
    def serialize_response(cls, data: Any) -> Any:
        """Recursively serialize response data."""
        if isinstance(data, dict):
            return {key: cls.serialize_response(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [cls.serialize_response(item) for item in data]
        elif isinstance(data, (pd.DataFrame, pd.Series, pd.Timestamp)):
            return cls.serialize_pandas(data)
        elif isinstance(data, (datetime, pd.Timestamp)):
            return cls.serialize_datetime(data)
        elif isinstance(data, (np.ndarray, np.integer, np.floating, np.bool_)):
            return cls.serialize_pandas(data)
        else:
            return data


class ParameterValidator:
    """Validates API request parameters."""
    
    @staticmethod
    def validate_date(date_str: str, param_name: str = "date") -> datetime:
        """Validate and parse date parameter."""
        if not date_str:
            raise ValueError(f"{param_name} parameter is required")
        
        # Try different date formats
        date_formats = ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%d/%m/%Y', '%d-%m-%Y']
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        raise ValueError(f"Invalid {param_name} format. Expected: YYYY-MM-DD")
    
    @staticmethod
    def validate_date_range(start_date: str, end_date: str) -> tuple:
        """Validate date range parameters."""
        start = ParameterValidator.validate_date(start_date, "start_date")
        end = ParameterValidator.validate_date(end_date, "end_date")
        
        if start >= end:
            raise ValueError("start_date must be before end_date")
        
        # Check if range is reasonable (not more than 10 years)
        if (end - start).days > 365 * 10:
            raise ValueError("Date range cannot exceed 10 years")
        
        return start, end
    
    @staticmethod
    def validate_currency(currency: str, supported_currencies: List[str]) -> str:
        """Validate currency parameter."""
        if not currency:
            raise ValueError("Currency parameter is required")
        
        currency = currency.upper()
        if currency not in supported_currencies:
            raise ValueError(f"Unsupported currency: {currency}. Supported: {', '.join(supported_currencies)}")
        
        return currency
    
    @staticmethod
    def validate_numeric_range(value: Union[str, int, float], param_name: str, 
                             min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
        """Validate numeric parameter within specified range."""
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            raise ValueError(f"{param_name} must be a number")
        
        if min_val is not None and numeric_value < min_val:
            raise ValueError(f"{param_name} must be >= {min_val}")
        
        if max_val is not None and numeric_value > max_val:
            raise ValueError(f"{param_name} must be <= {max_val}")
        
        return numeric_value
    
    @staticmethod
    def validate_columns(columns: Union[str, List[str]], available_columns: List[str]) -> List[str]:
        """Validate column names parameter."""
        if isinstance(columns, str):
            columns = [col.strip() for col in columns.split(',')]
        
        if not isinstance(columns, list):
            raise ValueError("Columns must be a list or comma-separated string")
        
        invalid_columns = [col for col in columns if col not in available_columns]
        if invalid_columns:
            raise ValueError(f"Invalid columns: {', '.join(invalid_columns)}")
        
        return columns


class DataFilter:
    """Filters and transforms data based on API parameters."""
    
    @staticmethod
    def filter_by_date_range(df: pd.DataFrame, start_date: Optional[datetime] = None, 
                           end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Filter DataFrame by date range."""
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        return df
    
    @staticmethod
    def filter_by_currency(df: pd.DataFrame, currency: str) -> pd.DataFrame:
        """Filter DataFrame columns by currency."""
        currency_cols = [col for col in df.columns if currency.lower() in col.lower()]
        if not currency_cols:
            raise ValueError(f"No columns found for currency: {currency}")
        return df[currency_cols]
    
    @staticmethod
    def limit_response_size(df: pd.DataFrame, max_rows: int = 1000) -> pd.DataFrame:
        """Limit DataFrame size for API response."""
        if len(df) > max_rows:
            logger.warning(f"Response size limited to {max_rows} rows (original: {len(df)})")
            return df.tail(max_rows)  # Return most recent data
        return df
    
    @staticmethod
    def paginate_data(df: pd.DataFrame, page: int = 1, per_page: int = 100) -> Dict:
        """Paginate DataFrame for API response."""
        total_items = len(df)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        paginated_df = df.iloc[start_idx:end_idx]
        
        return {
            'data': paginated_df,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total_items': total_items,
                'total_pages': (total_items + per_page - 1) // per_page,
                'has_next': end_idx < total_items,
                'has_prev': page > 1
            }
        }


class StatisticsCalculator:
    """Calculates various statistical measures for API responses."""
    
    @staticmethod
    def basic_stats(series: pd.Series) -> Dict:
        """Calculate basic statistical measures."""
        if len(series) == 0:
            return {}
        
        series_clean = series.dropna()
        if len(series_clean) == 0:
            return {"count": 0, "missing": len(series)}
        
        return {
            "count": len(series_clean),
            "missing": len(series) - len(series_clean),
            "mean": float(series_clean.mean()),
            "median": float(series_clean.median()),
            "std": float(series_clean.std()),
            "min": float(series_clean.min()),
            "max": float(series_clean.max()),
            "q25": float(series_clean.quantile(0.25)),
            "q75": float(series_clean.quantile(0.75))
        }
    
    @staticmethod
    def rolling_stats(series: pd.Series, window: int = 30) -> Dict:
        """Calculate rolling statistics."""
        if len(series) < window:
            return {}
        
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        
        return {
            "window": window,
            "latest_mean": float(rolling_mean.iloc[-1]) if not rolling_mean.empty else None,
            "latest_std": float(rolling_std.iloc[-1]) if not rolling_std.empty else None,
            "mean_trend": "increasing" if rolling_mean.iloc[-1] > rolling_mean.iloc[-window] else "decreasing"
        }
    
    @staticmethod
    def correlation_analysis(df: pd.DataFrame, target_column: str) -> Dict:
        """Calculate correlations with target column."""
        if target_column not in df.columns:
            return {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = df[numeric_cols].corrwith(df[target_column]).dropna()
        
        return {
            "target_column": target_column,
            "correlations": correlations.to_dict(),
            "highest_positive": correlations.nlargest(5).to_dict(),
            "highest_negative": correlations.nsmallest(5).to_dict()
        }


class RequestParser:
    """Parses and validates API request parameters."""
    
    @staticmethod
    def get_pagination_params() -> Dict:
        """Get pagination parameters from request."""
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 100, type=int), 1000)  # Max 1000 per page
        
        if page < 1:
            page = 1
        if per_page < 1:
            per_page = 100
            
        return {'page': page, 'per_page': per_page}
    
    @staticmethod
    def get_date_range_params() -> Dict:
        """Get date range parameters from request."""
        params = {}
        
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        if start_date:
            params['start_date'] = ParameterValidator.validate_date(start_date, 'start_date')
        
        if end_date:
            params['end_date'] = ParameterValidator.validate_date(end_date, 'end_date')
        
        # Validate range if both provided
        if 'start_date' in params and 'end_date' in params:
            if params['start_date'] >= params['end_date']:
                raise ValueError("start_date must be before end_date")
        
        return params
    
    @staticmethod
    def get_filter_params() -> Dict:
        """Get common filter parameters from request."""
        return {
            'currency': request.args.get('currency', '').upper(),
            'include_data': request.args.get('include_data', 'false').lower() == 'true',
            'format': request.args.get('format', 'json').lower(),
            'columns': request.args.get('columns', '').split(',') if request.args.get('columns') else []
        }


class ResponseBuilder:
    """Builds standardized API responses."""
    
    @staticmethod
    def build_success_response(data: Any, message: str = "Success", 
                             metadata: Optional[Dict] = None) -> Dict:
        """Build successful response."""
        response = {
            "status": "success",
            "message": message,
            "data": DataSerializer.serialize_response(data),
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata:
            response["metadata"] = metadata
        
        return response
    
    @staticmethod
    def build_error_response(message: str, error_code: str = "GENERAL_ERROR", 
                           details: Optional[Dict] = None) -> Dict:
        """Build error response."""
        response = {
            "status": "error",
            "error_code": error_code,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        if details:
            response["details"] = details
        
        return response
    
    @staticmethod
    def build_paginated_response(data: pd.DataFrame, page: int, per_page: int, 
                               message: str = "Success") -> Dict:
        """Build paginated response."""
        paginated = DataFilter.paginate_data(data, page, per_page)
        
        return ResponseBuilder.build_success_response(
            data=paginated['data'],
            message=message,
            metadata={
                "pagination": paginated['pagination']
            }
        )
