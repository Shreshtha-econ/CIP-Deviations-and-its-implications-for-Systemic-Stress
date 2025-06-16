"""
Systemic Risk Indicators Module
Implements the ECB CISS methodology and related risk indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from config.settings import ANALYSIS_CONFIG, RAW_DATA_DIR

logger = logging.getLogger(__name__)


class SystemicRiskAnalyzer:
    """Constructs systemic risk indicators using ECB CISS methodology."""
    
    def __init__(self, config: Dict = None):
        self.config = config or ANALYSIS_CONFIG
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=1)
        
    def create_market_blocks(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create market block indicators according to ECB CISS methodology."""
        data_with_blocks = data.copy()
        
        # Define block components based on ECB CISS methodology
        block_definitions = {
            'money_block': {
                'components': ['1', '2', '3'],
                'description': 'Money market stress indicators'
            },
            'bond_block': {
                'components': ['4', '5', '7'], 
                'description': 'Government and corporate bond market stress'
            },
            'equity_block': {
                'components': ['8', '10.1', '10.2'],
                'description': 'Equity market stress indicators'
            },
            'fin_block': {
                'components': ['10', '11'],
                'description': 'Financial intermediaries stress'
            }
        }
        
        # Create blocks by averaging available components
        for block_name, block_info in block_definitions.items():
            available_components = [col for col in block_info['components'] 
                                  if col in data_with_blocks.columns]
            
            if available_components:
                data_with_blocks[block_name] = data_with_blocks[available_components].mean(
                    axis=1, skipna=True
                )
                logger.info(f"Created {block_name} from {len(available_components)} components")
            else:
                logger.warning(f"No components available for {block_name}")
                data_with_blocks[block_name] = np.nan
        
        return data_with_blocks
    
    def create_fx_block(self, bandwidth_data: pd.DataFrame) -> pd.Series:
        """Create FX block from currency bandwidth data."""
        fx_components = [
            "Band_Width_scaled_usd",
            "Band_Width_scaled_gbp", 
            "Band_Width_scaled_jpy",
            "Band_Width_scaled_sek",
            "Band_Width_scaled_chf"
        ]
        
        # Filter available components
        available_fx = [col for col in fx_components if col in bandwidth_data.columns]
        
        if len(available_fx) < 2:
            logger.error("Insufficient FX components for block creation")
            return pd.Series(index=bandwidth_data.index, dtype=float)
        
        # Clean data
        fx_data = bandwidth_data[available_fx].dropna()
        
        if len(fx_data) == 0:
            logger.error("No valid FX data after cleaning")
            return pd.Series(index=bandwidth_data.index, dtype=float)
        
        # Standardize and apply PCA
        fx_scaled = self.scaler.fit_transform(fx_data)
        fx_block_scores = self.pca.fit_transform(fx_scaled).flatten()
        
        # Create series aligned with original index
        fx_block = pd.Series(
            fx_block_scores, 
            index=fx_data.index, 
            name="fx_block"
        )
        
        logger.info(f"Created FX block from {len(available_fx)} currencies")
        logger.info(f"PCA explained variance: {self.pca.explained_variance_ratio_[0]:.3f}")
        
        return fx_block
    
    def construct_ciss_indicator(self, market_data: pd.DataFrame, 
                               fx_block: pd.Series = None) -> Dict:
        """Construct CISS indicator using ECB methodology."""
        try:
            # Define required blocks
            standard_blocks = ['money_block', 'bond_block', 'equity_block', 'fin_block']
            
            # Check for existing blocks or create them
            if not all(block in market_data.columns for block in standard_blocks):
                logger.info("Creating market blocks...")
                market_data = self.create_market_blocks(market_data)
            
            # Prepare block data
            available_blocks = [col for col in standard_blocks if col in market_data.columns]
            
            if fx_block is not None:
                # Add FX block to the analysis
                blocks_data = market_data[available_blocks].copy()
                blocks_data = blocks_data.join(fx_block, how='inner')
                available_blocks.append('fx_block')
            else:
                blocks_data = market_data[available_blocks].copy()
            
            # Clean data
            blocks_clean = blocks_data.dropna()
            
            if len(blocks_clean) == 0:
                raise ValueError("No valid data after cleaning for CISS construction")
            
            logger.info(f"Constructing CISS from {len(available_blocks)} blocks")
            logger.info(f"Data period: {blocks_clean.index.min()} to {blocks_clean.index.max()}")
            logger.info(f"Valid observations: {len(blocks_clean)}")
            
            # Standardize blocks
            scaler_blocks = StandardScaler()
            blocks_standardized = scaler_blocks.fit_transform(blocks_clean)
            
            # Apply PCA for CISS construction
            pca_ciss = PCA(n_components=1)
            ciss_scores = pca_ciss.fit_transform(blocks_standardized).flatten()
            
            # Normalize CISS to [0,1] range
            ciss_min, ciss_max = ciss_scores.min(), ciss_scores.max()
            if ciss_max > ciss_min:
                ciss_normalized = (ciss_scores - ciss_min) / (ciss_max - ciss_min)
            else:
                ciss_normalized = np.zeros_like(ciss_scores)
            
            # Create CISS series
            ciss_series = pd.Series(
                ciss_normalized,
                index=blocks_clean.index,
                name="CISS"
            )
            
            # Calculate block contributions
            loadings = pca_ciss.components_[0]
            block_contributions = {
                block: float(loading) for block, loading in zip(available_blocks, loadings)
            }
            
            # Create result dictionary
            result = {
                'ciss_series': ciss_series,
                'explained_variance': float(pca_ciss.explained_variance_ratio_[0]),
                'block_contributions': block_contributions,
                'standardization_params': {
                    'mean': scaler_blocks.mean_.tolist(),
                    'scale': scaler_blocks.scale_.tolist()
                },
                'normalization_params': {
                    'min': float(ciss_min),
                    'max': float(ciss_max)
                },
                'construction_info': {
                    'blocks_used': available_blocks,
                    'observations': len(blocks_clean),
                    'date_range': {
                        'start': str(blocks_clean.index.min()),
                        'end': str(blocks_clean.index.max())
                    }
                }
            }
            
            logger.info(f"CISS constructed successfully")
            logger.info(f"Explained variance: {result['explained_variance']:.3f}")
            logger.info(f"Date range: {result['construction_info']['date_range']['start']} to {result['construction_info']['date_range']['end']}")
            
            return result
            
        except Exception as e:
            logger.error(f"CISS construction failed: {str(e)}")
            raise
    
    def load_official_ecb_ciss(self) -> pd.Series:
        """Load official ECB CISS data for comparison."""
        try:
            ecb_file = RAW_DATA_DIR / 'ecb_ciss.xlsx'
            
            if not ecb_file.exists():
                logger.warning(f"ECB CISS file not found: {ecb_file}")
                return pd.Series(dtype=float)
            
            official_ecb = pd.read_excel(ecb_file, engine="openpyxl", header=1)
            official_ecb["Date"] = pd.to_datetime(official_ecb["Date"], errors='coerce')
            official_ecb = official_ecb.set_index("Date")
            
            # Try different possible column names
            possible_columns = ['ECB_CISS', 'CISS', 'Value', 'Index']
            ciss_column = None
            
            for col in possible_columns:
                if col in official_ecb.columns:
                    ciss_column = col
                    break
            
            if ciss_column is None:
                logger.error(f"Could not find CISS column in {official_ecb.columns.tolist()}")
                return pd.Series(dtype=float)
            
            official_ciss = official_ecb[ciss_column].dropna()
            
            logger.info(f"Loaded official ECB CISS: {len(official_ciss)} observations")
            logger.info(f"Date range: {official_ciss.index.min()} to {official_ciss.index.max()}")
            
            return official_ciss
            
        except Exception as e:
            logger.error(f"Failed to load official ECB CISS: {str(e)}")
            return pd.Series(dtype=float)
    
    def compare_with_official_ciss(self, constructed_ciss: Dict) -> Dict:
        """Compare constructed CISS with official ECB CISS."""
        try:
            official_ciss = self.load_official_ecb_ciss()
            
            if len(official_ciss) == 0:
                return {
                    'error': 'Official ECB CISS data not available',
                    'comparison': None
                }
            
            constructed_series = constructed_ciss['ciss_series']
            
            # Align series for comparison
            comparison_df = pd.DataFrame({
                'Official_ECB_CISS': official_ciss,
                'Constructed_CISS': constructed_series
            }).dropna()
            
            if len(comparison_df) == 0:
                return {
                    'error': 'No overlapping data for comparison',
                    'comparison': None
                }
            
            # Calculate comparison statistics
            correlation = comparison_df['Official_ECB_CISS'].corr(comparison_df['Constructed_CISS'])
            
            # Calculate RMSE
            rmse = np.sqrt(
                np.mean((comparison_df['Official_ECB_CISS'] - comparison_df['Constructed_CISS'])**2)
            )
            
            # Calculate MAE
            mae = np.mean(
                np.abs(comparison_df['Official_ECB_CISS'] - comparison_df['Constructed_CISS'])
            )
            
            # Direction accuracy (same direction of change)
            official_changes = comparison_df['Official_ECB_CISS'].diff().dropna()
            constructed_changes = comparison_df['Constructed_CISS'].diff().dropna()
            
            direction_accuracy = np.mean(
                np.sign(official_changes) == np.sign(constructed_changes)
            ) if len(official_changes) > 0 else np.nan
            
            comparison_result = {
                'comparison_data': comparison_df,
                'statistics': {
                    'correlation': float(correlation),
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'direction_accuracy': float(direction_accuracy),
                    'observations': len(comparison_df)
                },
                'date_range': {
                    'start': str(comparison_df.index.min()),
                    'end': str(comparison_df.index.max())
                }
            }
            
            logger.info(f"CISS Comparison Results:")
            logger.info(f"  Correlation: {correlation:.4f}")
            logger.info(f"  RMSE: {rmse:.4f}")
            logger.info(f"  MAE: {mae:.4f}")
            logger.info(f"  Direction Accuracy: {direction_accuracy:.4f}")
            logger.info(f"  Observations: {len(comparison_df)}")
            
            return comparison_result
            
        except Exception as e:
            logger.error(f"CISS comparison failed: {str(e)}")
            return {'error': str(e), 'comparison': None}
    
    def calculate_cross_correlation(self, comparison_data: pd.DataFrame, 
                                  max_lag: int = 24) -> Dict:
        """Calculate cross-correlation between official and constructed CISS."""
        try:
            official = comparison_data["Official_ECB_CISS"]
            constructed = comparison_data["Constructed_CISS"]
            
            # Center the series
            official_centered = official - official.mean()
            constructed_centered = constructed - constructed.mean()
            
            # Calculate cross-correlation for different lags
            lags = np.arange(-max_lag, max_lag + 1)
            ccf_values = []
            
            for lag in lags:
                if lag < 0:
                    # Negative lag: constructed leads official
                    ccf_val = constructed_centered.shift(abs(lag)).corr(official_centered)
                else:
                    # Positive lag: official leads constructed
                    ccf_val = official_centered.shift(lag).corr(constructed_centered)
                
                ccf_values.append(ccf_val)
            
            # Find peak correlation and its lag
            ccf_values_clean = [v for v in ccf_values if not np.isnan(v)]
            if ccf_values_clean:
                max_corr_idx = np.nanargmax(np.abs(ccf_values))
                peak_lag = lags[max_corr_idx]
                peak_correlation = ccf_values[max_corr_idx]
            else:
                peak_lag = 0
                peak_correlation = np.nan
            
            result = {
                'lags': lags.tolist(),
                'ccf_values': ccf_values,
                'peak_correlation': float(peak_correlation),
                'peak_lag': int(peak_lag),
                'interpretation': self._interpret_lag(peak_lag)
            }
            
            logger.info(f"Cross-correlation analysis completed")
            logger.info(f"Peak correlation: {peak_correlation:.4f} at lag {peak_lag}")
            
            return result
            
        except Exception as e:
            logger.error(f"Cross-correlation calculation failed: {str(e)}")
            return {'error': str(e)}
    
    def _interpret_lag(self, lag: int) -> str:
        """Interpret the meaning of the peak lag."""
        if lag == 0:
            return "No significant lead-lag relationship"
        elif lag > 0:
            return f"Official CISS leads constructed CISS by {lag} periods"
        else:
            return f"Constructed CISS leads official CISS by {abs(lag)} periods"
