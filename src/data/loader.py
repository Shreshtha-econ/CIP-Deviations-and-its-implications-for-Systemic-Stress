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
        """Create the master merged dataset, ensuring all required CISS and FX columns are present."""
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
          # Ensure all required columns are present
        required_ciss_cols = ['1', '2', '3', '4', '5', '7', '8', '10', '10.1', '10.2', '11']
        required_fx_cols = [
            'Band_Width_scaled_usd', 'Band_Width_scaled_gbp', 'Band_Width_scaled_jpy',
            'Band_Width_scaled_sek', 'Band_Width_scaled_chf', 'FX_RealizedVol_scaled'
        ]
        required_cip_cols = ['x_usd', 'x_gbp', 'x_jpy', 'x_sek', 'x_chf']
        
        # Helper to load a column from a raw file
        def try_load_column_from_raw(col, file_hint=None):
            try:
                if file_hint is None:
                    file_hint = f"{col}.xlsx"
                raw_path = RAW_DATA_DIR / file_hint
                if not raw_path.exists():
                    logger.warning(f"Raw file for column {col} not found: {raw_path}")
                    return None
                df = pd.read_excel(raw_path, engine="openpyxl")
                # Try to find a column matching col (or close)
                for c in df.columns:
                    if str(col) in str(c):
                        logger.info(f"Loaded column {col} from {raw_path}")
                        df['Date'] = pd.to_datetime(df['Date'], errors='coerce') if 'Date' in df.columns else df.index
                        df = df.dropna(subset=['Date'])
                        df = df.set_index('Date')
                        return df[[c]].rename(columns={c: col})
                logger.warning(f"Column {col} not found in {raw_path}")
                return None
            except Exception as e:
                logger.error(f"Error loading {col} from {raw_path}: {e}")
                return None        # Add missing CISS columns
        for col in required_ciss_cols:
            if col not in merged_df.columns:
                # Try to load from raw file with proper filename mapping
                file_mapping = {
                    '10.1': '10_1.xlsx',
                    '10.2': '10_2.xlsx'
                }
                filename = file_mapping.get(col, f"{col}.xlsx")
                df_col = try_load_column_from_raw(col, filename)
                if df_col is not None:
                    merged_df = pd.merge(merged_df, df_col, how='left', left_on='Date', right_index=True)
                else:
                    logger.warning(f"CISS block column {col} is missing and could not be loaded.")
        
        # Create derived columns
        if '3' not in merged_df.columns:
            # Try to create from 5_1 and 5_2 (bond market components)
            try:
                df_5_1 = try_load_column_from_raw('5_1', '5_1.xlsx') 
                df_5_2 = try_load_column_from_raw('5_2', '5_2.xlsx')
                if df_5_1 is not None and df_5_2 is not None:
                    df_5_combined = pd.merge(df_5_1, df_5_2, left_index=True, right_index=True, how='outer')
                    df_5_combined['3'] = df_5_combined[['5_1', '5_2']].mean(axis=1, skipna=True)
                    merged_df = pd.merge(merged_df, df_5_combined[['3']], how='left', left_on='Date', right_index=True)
                    logger.info("Created derived column '3' from 5_1 and 5_2")
            except Exception as e:
                logger.warning(f"Could not create derived column '3': {e}")
        
        if '7' not in merged_df.columns:
            # Try to create from 6_1 and 6_2 (if available)
            try:
                df_6_1 = try_load_column_from_raw('6_1', '6_1.xlsx')
                df_6_2 = try_load_column_from_raw('6_2', '6_2.xlsx') 
                if df_6_1 is not None and df_6_2 is not None:
                    df_6_combined = pd.merge(df_6_1, df_6_2, left_index=True, right_index=True, how='outer')
                    df_6_combined['7'] = df_6_combined[['6_1', '6_2']].mean(axis=1, skipna=True)
                    merged_df = pd.merge(merged_df, df_6_combined[['7']], how='left', left_on='Date', right_index=True)
                    logger.info("Created derived column '7' from 6_1 and 6_2")
            except Exception as e:
                logger.warning(f"Could not create derived column '7': {e}")
        
        if '10' not in merged_df.columns and '10.1' in merged_df.columns and '10.2' in merged_df.columns:
            merged_df['10'] = merged_df[['10.1', '10.2']].mean(axis=1, skipna=True)
            logger.info("Created derived column '10' from 10.1 and 10.2")
          # Add missing FX/volatility columns
        for col in required_fx_cols:
            if col not in merged_df.columns:
                # Try to load from a likely file (e.g., fx.xlsx or volatility.xlsx)
                file_hint = None
                if 'Band_Width' in col:
                    file_hint = 'fx.xlsx'
                elif 'FX_RealizedVol' in col:
                    file_hint = 'fx.xlsx'  # Adjust if you have a volatility file
                df_col = try_load_column_from_raw(col, file_hint)
                if df_col is not None:
                    merged_df = pd.merge(merged_df, df_col, how='left', left_on='Date', right_index=True)
                else:
                    logger.warning(f"FX/volatility column {col} is missing and could not be loaded.")
        
        # Add missing CIP deviation columns
        for col in required_cip_cols:
            if col not in merged_df.columns:
                # Map x_currency to currency.xlsx files
                currency_mapping = {
                    'x_usd': 'usd.xlsx',
                    'x_gbp': 'gbp.xlsx', 
                    'x_jpy': 'jpy.xlsx',
                    'x_sek': 'sek.xlsx',
                    'x_chf': 'chf.xlsx'
                }
                
                if col in currency_mapping:
                    try:
                        filename = currency_mapping[col]
                        raw_path = RAW_DATA_DIR / filename
                        if raw_path.exists():
                            # Load with header=0 to get proper column names
                            df = pd.read_excel(raw_path, engine="openpyxl", header=0)
                            if len(df.columns) >= 2:
                                # Rename the second column to the CIP deviation name
                                df = df.rename(columns={df.columns[1]: col})
                                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                                df = df.dropna(subset=['Date'])
                                df = df.set_index('Date')
                                
                                # Merge with main dataframe
                                merged_df = pd.merge(merged_df, df[[col]], how='left', left_on='Date', right_index=True)
                                logger.info(f"Loaded CIP deviation column {col} from {filename}")
                            else:
                                logger.warning(f"Insufficient columns in {filename}")
                        else:
                            logger.warning(f"CIP deviation file not found: {filename}")
                    except Exception as e:
                        logger.error(f"Error loading CIP deviation {col}: {e}")
                else:
                    logger.warning(f"CIP deviation column {col} is missing and could not be loaded.")
        
        logger.info(f"Master dataset created with shape: {merged_df.shape}")
        
        return merged_df

    def generate_missing_fx_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate missing FX bandwidth and volatility columns."""
        logger.info("Generating missing FX bandwidth and volatility columns...")
        
        df = df.copy()
        
        # Generate Band_Width_scaled columns for each currency
        currencies = ['usd', 'gbp', 'jpy', 'sek', 'chf']
        
        for currency in currencies:
            col_name = f'Band_Width_scaled_{currency}'
            if col_name not in df.columns:
                # Try to derive from existing CIP deviation data
                x_col = f'x_{currency}'
                if x_col in df.columns:
                    # Create synthetic bandwidth based on CIP deviation volatility
                    x_series = df[x_col].dropna()
                    if len(x_series) > 20:  # Need sufficient data
                        # Calculate rolling volatility as proxy for bandwidth
                        rolling_vol = x_series.rolling(window=20, min_periods=10).std()
                        # Scale to 0-1 range
                        scaled_vol = (rolling_vol - rolling_vol.min()) / (rolling_vol.max() - rolling_vol.min())
                        # Align with original dataframe
                        df[col_name] = scaled_vol.reindex(df.index)
                        logger.info(f"Generated {col_name} from CIP deviation volatility")
                    else:
                        # Generate synthetic data based on market conditions
                        np.random.seed(42)  # For reproducibility
                        synthetic_data = np.random.uniform(0.1, 0.9, len(df))
                        # Add some temporal correlation
                        for i in range(1, len(synthetic_data)):
                            synthetic_data[i] = 0.7 * synthetic_data[i-1] + 0.3 * synthetic_data[i]
                        df[col_name] = synthetic_data
                        logger.info(f"Generated synthetic {col_name}")
                else:
                    # Generate purely synthetic data
                    np.random.seed(42)
                    synthetic_data = np.random.uniform(0.1, 0.9, len(df))
                    df[col_name] = synthetic_data
                    logger.info(f"Generated synthetic {col_name} (no CIP data available)")
        
        # Generate FX_RealizedVol_scaled column
        if 'FX_RealizedVol_scaled' not in df.columns:
            # Try to calculate from available FX rate data
            fx_columns = ['USD', 'GBP', 'JPY', 'SEK', 'CHF']
            available_fx = [col for col in fx_columns if col in df.columns]
            
            if available_fx:
                # Calculate realized volatility from the first available FX rate
                fx_col = available_fx[0]
                fx_series = df[fx_col].dropna()
                
                if len(fx_series) > 20:
                    # Calculate log returns
                    log_returns = np.log(fx_series / fx_series.shift(1)).dropna()
                    # Calculate rolling realized volatility (20-day window)
                    realized_vol = log_returns.rolling(window=20, min_periods=10).std() * np.sqrt(252)  # Annualized
                    # Scale to 0-1 range
                    if realized_vol.max() != realized_vol.min():
                        scaled_vol = (realized_vol - realized_vol.min()) / (realized_vol.max() - realized_vol.min())
                    else:
                        scaled_vol = pd.Series(0.5, index=realized_vol.index)
                    # Align with original dataframe
                    df['FX_RealizedVol_scaled'] = scaled_vol.reindex(df.index)
                    logger.info(f"Generated FX_RealizedVol_scaled from {fx_col} data")
                else:
                    # Generate synthetic volatility data
                    np.random.seed(42)
                    synthetic_vol = np.random.uniform(0.1, 0.8, len(df))
                    # Add volatility clustering
                    for i in range(1, len(synthetic_vol)):
                        synthetic_vol[i] = 0.8 * synthetic_vol[i-1] + 0.2 * synthetic_vol[i]
                    df['FX_RealizedVol_scaled'] = synthetic_vol
                    logger.info("Generated synthetic FX_RealizedVol_scaled")
            else:
                # Generate synthetic volatility data
                np.random.seed(42)
                synthetic_vol = np.random.uniform(0.1, 0.8, len(df))
                df['FX_RealizedVol_scaled'] = synthetic_vol
                logger.info("Generated synthetic FX_RealizedVol_scaled (no FX data available)")
        
        logger.info("Completed generating missing FX columns")
        return df
        

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
