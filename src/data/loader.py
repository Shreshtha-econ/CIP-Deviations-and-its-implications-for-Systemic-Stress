"""
Data Loading and Management Module
Handles loading and initial processing of financial data from Excel files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
from functools import reduce

from config.settings import DATA_FILES, RAW_DATA_DIR, PROCESSED_DATA_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and basic processing of financial data."""
    
    def __init__(self, data_dir: Union[str, Path] = None):
        self.data_dir = Path(data_dir) if data_dir else RAW_DATA_DIR
        self.processed_dir = PROCESSED_DATA_DIR
        
    def load_excel_file(self, file_path: Union[str, Path], 
                       header: int = 1, 
                       sheet_name: str = None) -> pd.DataFrame:
        """Load a single Excel file with error handling."""
        try:
            full_path = self.data_dir / file_path
            
            # Handle multiple sheets by reading all and selecting the best one
            if sheet_name is None:
                # Read all sheets to handle multi-sheet files
                all_sheets = pd.read_excel(full_path, engine='openpyxl', 
                                         header=header, sheet_name=None)
                
                if isinstance(all_sheets, dict):
                    # Multiple sheets - find the best one
                    if len(all_sheets) == 1:
                        # Only one sheet, use it
                        df = list(all_sheets.values())[0]
                    else:
                        # Multiple sheets - prefer common names or largest
                        preferred_names = ['Data', 'Sheet1', 'data', 'DATA']
                        sheet_to_use = None
                        
                        for preferred in preferred_names:
                            if preferred in all_sheets:
                                sheet_to_use = preferred
                                break
                        
                        if sheet_to_use is None:
                            # Use the largest sheet
                            sheet_to_use = max(all_sheets.keys(), 
                                             key=lambda x: len(all_sheets[x]))
                        
                        df = all_sheets[sheet_to_use]
                        logger.info(f"Successfully loaded {file_path} (sheet: {sheet_to_use})")
                        return df
                else:
                    # Single sheet returned directly
                    df = all_sheets
            else:
                # Specific sheet requested
                df = pd.read_excel(full_path, engine='openpyxl', 
                                 header=header, sheet_name=sheet_name)
            
            logger.info(f"Successfully loaded {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {str(e)}")
            raise
    
    def load_forward_rates(self) -> Dict[str, pd.DataFrame]:
        """Load all forward rate data."""
        forward_rates = {}
        for currency, file_path in DATA_FILES['forward_rates'].items():
            try:
                df = self.load_excel_file(file_path)
                forward_rates[currency] = df
            except Exception as e:
                logger.warning(f"Could not load forward rate for {currency}: {e}")
        return forward_rates
    
    def load_spot_rates(self) -> Dict[str, pd.DataFrame]:
        """Load all spot rate data."""
        spot_rates = {}
        for currency, file_path in DATA_FILES['spot_rates'].items():
            try:
                df = self.load_excel_file(file_path)
                spot_rates[currency] = df
            except Exception as e:
                logger.warning(f"Could not load spot rate for {currency}: {e}")
        return spot_rates
    
    def load_interest_rates(self) -> Dict[str, pd.DataFrame]:
        """Load all interest rate data."""
        interest_rates = {}
        for rate_type, file_path in DATA_FILES['interest_rates'].items():
            try:
                df = self.load_excel_file(file_path)
                interest_rates[rate_type] = df
            except Exception as e:
                logger.warning(f"Could not load interest rate for {rate_type}: {e}")
        return interest_rates
    
    def load_market_indicators(self) -> Dict[str, pd.DataFrame]:
        """Load all market indicator data."""
        market_data = {}
        for indicator, file_path in DATA_FILES['market_indicators'].items():
            try:
                df = self.load_excel_file(file_path)
                market_data[indicator] = df
            except Exception as e:
                logger.warning(f"Could not load market indicator {indicator}: {e}")
        return market_data
    
    def load_all_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Load all data categories."""
        return {
            'forward_rates': self.load_forward_rates(),
            'spot_rates': self.load_spot_rates(),
            'interest_rates': self.load_interest_rates(),
            'market_indicators': self.load_market_indicators()
        }
    
    @staticmethod
    def expand_monthly_to_daily(df: pd.DataFrame, 
                              date_col: str = 'Date', 
                              rate_col: str = 'Rate') -> pd.DataFrame:
        """Expand monthly data to daily frequency."""
        expanded_data = []
        for _, row in df.iterrows():
            start_date = pd.to_datetime(row[date_col]).replace(day=1)
            end_date = start_date + pd.offsets.MonthEnd(0)
            date_range = pd.date_range(start_date, end_date)
            
            expanded_df = pd.DataFrame({
                date_col: date_range,
                rate_col: row[rate_col]
            })
            expanded_data.append(expanded_df)
        
        return pd.concat(expanded_data, ignore_index=True)


class DataMerger:
    """Handles merging of multiple DataFrames."""
    
    def __init__(self):
        self.loader = DataLoader()
    
    def prepare_dataframes_for_merge(self, dataframes: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """Prepare dataframes for merging by standardizing date columns."""
        prepared_dfs = []
        
        for i, df in enumerate(dataframes):
            try:
                if not isinstance(df, pd.DataFrame):
                    logger.warning(f"Item {i} is not a DataFrame: {type(df)}")
                    continue
                    
                if df.empty:
                    logger.warning(f"DataFrame {i} is empty, skipping")
                    continue
                
                df_copy = df.copy()
                
                # Ensure Date column exists and is properly formatted
                if 'Date' not in df_copy.columns:
                    if df_copy.index.name == 'Date':
                        df_copy = df_copy.reset_index()
                    elif 'date' in df_copy.columns:
                        df_copy = df_copy.rename(columns={'date': 'Date'})
                    else:
                        logger.warning(f"DataFrame {i} is missing 'Date' column: {df_copy.columns.tolist()}")
                        continue
                
                # Convert Date to datetime and set as index
                df_copy['Date'] = pd.to_datetime(df_copy['Date'], errors='coerce')
                
                # Remove rows with invalid dates
                df_copy = df_copy.dropna(subset=['Date'])
                
                if df_copy.empty:
                    logger.warning(f"DataFrame {i} has no valid dates after cleaning")
                    continue
                
                df_copy = df_copy.set_index('Date')
                prepared_dfs.append(df_copy)
                
            except Exception as e:
                logger.error(f"Error preparing DataFrame {i}: {e}")
                continue
        
        return prepared_dfs
    
    def merge_dataframes(self, dataframes: List[pd.DataFrame], 
                        how: str = 'outer') -> pd.DataFrame:
        """Merge multiple dataframes on Date index."""
        if not dataframes:
            raise ValueError("No dataframes provided for merging")
        
        prepared_dfs = self.prepare_dataframes_for_merge(dataframes)
        
        # Merge all dataframes
        merged = reduce(
            lambda left, right: pd.merge(
                left, right, 
                how=how, 
                left_index=True, 
                right_index=True,
                suffixes=('', '_dup')
            ), 
            prepared_dfs
        )
        
        # Clean up duplicate columns
        merged = merged.loc[:, ~merged.columns.str.endswith('_dup')]
        
        # Clean up any remaining unwanted columns
        merged = merged.drop(columns=['level_0', 'index'], errors='ignore')
          # Sort by date and reset index
        merged = merged.sort_index()
        merged = merged.reset_index()
        
        return merged
    
    def create_master_dataset(self, start_date: str = '1999-01-01') -> pd.DataFrame:
        """Create the master merged dataset."""
        logger.info("Loading all data sources...")
        
        # Load all data
        all_data = self.loader.load_all_data()
        
        # Collect all dataframes for merging
        dfs_to_merge = []
        
        # Add forward rates
        for currency, df in all_data['forward_rates'].items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                dfs_to_merge.append(df)
            else:
                logger.warning(f"Skipping forward rate {currency}: not a valid DataFrame")
        
        # Add spot rates
        for currency, df in all_data['spot_rates'].items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                dfs_to_merge.append(df)
            else:
                logger.warning(f"Skipping spot rate {currency}: not a valid DataFrame")
        
        # Add interest rates
        for rate_type, df in all_data['interest_rates'].items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                dfs_to_merge.append(df)
            else:
                logger.warning(f"Skipping interest rate {rate_type}: not a valid DataFrame")
        
        # Add market indicators
        if 'market_indicators' in all_data:
            for indicator, df in all_data['market_indicators'].items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    dfs_to_merge.append(df)
                else:
                    logger.warning(f"Skipping market indicator {indicator}: not a valid DataFrame")
        
        if not dfs_to_merge:
            raise ValueError("No valid dataframes found for merging")
        
        logger.info(f"Merging {len(dfs_to_merge)} dataframes...")
        
        # Merge all dataframes
        merged_df = self.merge_dataframes(dfs_to_merge)
        
        # Filter by start date
        if start_date:
            merged_df = merged_df[merged_df['Date'] >= start_date]
        
        logger.info(f"Master dataset created with shape: {merged_df.shape}")
        
        return merged_df


def save_processed_data(df: pd.DataFrame, filename: str) -> None:
    """Save processed data to CSV."""
    file_path = PROCESSED_DATA_DIR / filename
    df.to_csv(file_path, index=False)
    logger.info(f"Saved processed data to {file_path}")


def load_processed_data(filename: str) -> pd.DataFrame:
    """Load processed data from CSV."""
    file_path = PROCESSED_DATA_DIR / filename
    if file_path.exists():
        df = pd.read_csv(file_path, parse_dates=['Date'])
        logger.info(f"Loaded processed data from {file_path}")
        return df
    else:
        raise FileNotFoundError(f"Processed data file not found: {file_path}")
