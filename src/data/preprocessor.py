"""
Data Preprocessing Module
Handles data cleaning, transformation, and validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data preprocessing and transformation."""
    
    def __init__(self):
        self.processed_columns = []
        self.transformation_log = []
    
    def convert_numeric_columns(self, df: pd.DataFrame, 
                              exclude_columns: List[str] = None) -> pd.DataFrame:
        """Convert columns to numeric, excluding specified columns."""
        if exclude_columns is None:
            exclude_columns = ['Date', 'Year', 'Month']
        
        df_processed = df.copy()
        cols_to_convert = df_processed.columns.difference(exclude_columns)
        
        conversion_results = {}
        for col in cols_to_convert:
            try:
                original_dtype = df_processed[col].dtype
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                
                # Log conversion results
                null_count = df_processed[col].isnull().sum()
                conversion_results[col] = {
                    'original_dtype': str(original_dtype),
                    'new_dtype': str(df_processed[col].dtype),
                    'null_values_created': null_count
                }
                
                if null_count > 0:
                    logger.warning(f"Column {col}: {null_count} values converted to NaN")
                
            except Exception as e:
                logger.error(f"Failed to convert column {col}: {str(e)}")
                conversion_results[col] = {'error': str(e)}
        
        self.transformation_log.append({
            'operation': 'convert_numeric_columns',
            'results': conversion_results
        })
        
        logger.info(f"Converted {len(cols_to_convert)} columns to numeric")
        return df_processed
    
    def handle_missing_values(self, df: pd.DataFrame, 
                            strategy: str = 'forward_fill',
                            columns: List[str] = None) -> pd.DataFrame:
        """Handle missing values using specified strategy."""
        df_processed = df.copy()
        
        if columns is None:
            columns = df_processed.columns.tolist()
        
        missing_before = df_processed[columns].isnull().sum().sum()
        
        if strategy == 'forward_fill':
            df_processed[columns] = df_processed[columns].fillna(method='ffill')
        elif strategy == 'backward_fill':
            df_processed[columns] = df_processed[columns].fillna(method='bfill')
        elif strategy == 'interpolate':
            df_processed[columns] = df_processed[columns].interpolate()
        elif strategy == 'drop':
            df_processed = df_processed.dropna(subset=columns)
        elif strategy == 'zero':
            df_processed[columns] = df_processed[columns].fillna(0)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        missing_after = df_processed[columns].isnull().sum().sum()
        
        self.transformation_log.append({
            'operation': 'handle_missing_values',
            'strategy': strategy,
            'missing_before': missing_before,
            'missing_after': missing_after,
            'columns_processed': len(columns)
        })
        
        logger.info(f"Missing values: {missing_before} -> {missing_after} using {strategy}")
        return df_processed
    
    def create_time_features(self, df: pd.DataFrame, 
                           date_column: str = 'Date') -> pd.DataFrame:
        """Create additional time-based features."""
        df_processed = df.copy()
        
        if date_column not in df_processed.columns:
            logger.warning(f"Date column {date_column} not found")
            return df_processed
        
        # Ensure date column is datetime
        df_processed[date_column] = pd.to_datetime(df_processed[date_column], errors='coerce')
        
        # Create time features
        time_features = {}
        if 'Year' not in df_processed.columns:
            df_processed['Year'] = df_processed[date_column].dt.year
            time_features['Year'] = 'Extracted year'
        
        if 'Month' not in df_processed.columns:
            df_processed['Month'] = df_processed[date_column].dt.month
            time_features['Month'] = 'Extracted month'
        
        if 'Quarter' not in df_processed.columns:
            df_processed['Quarter'] = df_processed[date_column].dt.quarter
            time_features['Quarter'] = 'Extracted quarter'
        
        if 'DayOfWeek' not in df_processed.columns:
            df_processed['DayOfWeek'] = df_processed[date_column].dt.dayofweek
            time_features['DayOfWeek'] = 'Extracted day of week (0=Monday)'
        
        if 'IsWeekend' not in df_processed.columns:
            df_processed['IsWeekend'] = df_processed['DayOfWeek'].isin([5, 6]).astype(int)
            time_features['IsWeekend'] = 'Weekend indicator (1=Weekend)'
        
        self.transformation_log.append({
            'operation': 'create_time_features',
            'features_created': time_features
        })
        
        logger.info(f"Created {len(time_features)} time features")
        return df_processed
    
    def remove_outliers(self, df: pd.DataFrame, 
                       columns: List[str] = None,
                       method: str = 'iqr',
                       threshold: float = 1.5) -> pd.DataFrame:
        """Remove outliers using specified method."""
        df_processed = df.copy()
        
        if columns is None:
            # Select only numeric columns
            columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        
        rows_before = len(df_processed)
        outliers_removed = {}
        
        for col in columns:
            if col not in df_processed.columns:
                continue
                
            if method == 'iqr':
                Q1 = df_processed[col].quantile(0.25)
                Q3 = df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outlier_mask = (df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)
                outliers_count = outlier_mask.sum()
                
                df_processed = df_processed[~outlier_mask]
                outliers_removed[col] = outliers_count
                
            elif method == 'zscore':
                z_scores = np.abs((df_processed[col] - df_processed[col].mean()) / df_processed[col].std())
                outlier_mask = z_scores > threshold
                outliers_count = outlier_mask.sum()
                
                df_processed = df_processed[~outlier_mask]
                outliers_removed[col] = outliers_count
        
        rows_after = len(df_processed)
        
        self.transformation_log.append({
            'operation': 'remove_outliers',
            'method': method,
            'threshold': threshold,
            'rows_before': rows_before,
            'rows_after': rows_after,
            'outliers_by_column': outliers_removed
        })
        
        logger.info(f"Outlier removal: {rows_before} -> {rows_after} rows")
        return df_processed
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """Perform comprehensive data quality validation."""
        validation_results = {
            'shape': df.shape,
            'columns': len(df.columns),
            'total_cells': df.shape[0] * df.shape[1],
            'missing_values': {},
            'data_types': {},
            'duplicates': 0,
            'date_range': {},
            'quality_score': 0
        }
        
        # Missing values analysis
        missing_counts = df.isnull().sum()
        validation_results['missing_values'] = {
            'total_missing': missing_counts.sum(),
            'missing_percentage': (missing_counts.sum() / validation_results['total_cells']) * 100,
            'columns_with_missing': (missing_counts > 0).sum(),
            'by_column': missing_counts[missing_counts > 0].to_dict()
        }
        
        # Data types
        validation_results['data_types'] = df.dtypes.value_counts().to_dict()
        
        # Duplicates
        validation_results['duplicates'] = df.duplicated().sum()
        
        # Date range analysis
        date_columns = df.select_dtypes(include=['datetime64']).columns
        if len(date_columns) > 0:
            for date_col in date_columns:
                validation_results['date_range'][date_col] = {
                    'start': str(df[date_col].min()),
                    'end': str(df[date_col].max()),
                    'unique_dates': df[date_col].nunique()
                }
        
        # Calculate quality score (0-100)
        quality_factors = []
        
        # Factor 1: Missing data (lower missing = higher score)
        missing_penalty = min(validation_results['missing_values']['missing_percentage'] / 10, 10)
        quality_factors.append(max(0, 10 - missing_penalty))
        
        # Factor 2: Duplicate data (lower duplicates = higher score)
        duplicate_percentage = (validation_results['duplicates'] / df.shape[0]) * 100
        duplicate_penalty = min(duplicate_percentage / 5, 10)
        quality_factors.append(max(0, 10 - duplicate_penalty))
        
        # Factor 3: Data type consistency (more numeric = higher score)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_ratio = len(numeric_columns) / len(df.columns)
        quality_factors.append(numeric_ratio * 10)
        
        validation_results['quality_score'] = sum(quality_factors) / len(quality_factors) * 10
        
        # Log validation results
        logger.info(f"Data Quality Report:")
        logger.info(f"  Shape: {validation_results['shape']}")
        logger.info(f"  Missing: {validation_results['missing_values']['missing_percentage']:.2f}%")
        logger.info(f"  Duplicates: {validation_results['duplicates']}")
        logger.info(f"  Quality Score: {validation_results['quality_score']:.1f}/100")
        
        return validation_results
    
    def get_transformation_summary(self) -> Dict:
        """Get summary of all transformations performed."""
        return {
            'transformations_performed': len(self.transformation_log),
            'transformation_details': self.transformation_log
        }
