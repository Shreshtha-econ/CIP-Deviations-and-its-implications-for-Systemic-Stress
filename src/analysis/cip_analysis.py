"""
CIP Analysis Module
Contains functions for calculating Covered Interest Parity deviations and related metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

from config.settings import CURRENCIES, ANALYSIS_CONFIG

logger = logging.getLogger(__name__)


class CIPAnalyzer:
    """Handles Covered Interest Parity deviation analysis."""
    
    def __init__(self, config: Dict = None):
        self.config = config or ANALYSIS_CONFIG
        self.currencies = CURRENCIES
    
    def calculate_rate_conversions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert USD/EUR rates to EUR/USD rates."""
        df = df.copy()
        
        # Invert spot rate and forward rate
        if 'SpotRateUSDtoEUR' in df.columns:
            df['SpotRateEURtoUSD'] = 1 / df['SpotRateUSDtoEUR']
        
        if 'ForwardRateUSDtoEUR' in df.columns:
            df['ForwardRateEURtoUSD'] = 1 / df['ForwardRateUSDtoEUR']
        
        # For high/low forward rates, invert and swap
        if all(col in df.columns for col in ['ForwardRateUSDtoEUR_high', 'ForwardRateUSDtoEUR_low']):
            df['ForwardRateEURtoUSD_low'] = 1 / df['ForwardRateUSDtoEUR_high']
            df['ForwardRateEURtoUSD_high'] = 1 / df['ForwardRateUSDtoEUR_low']
        
        return df
    
    def calculate_cip_deviations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate CIP deviations for all currencies."""
        df = df.copy()
        
        # Calculate rho (forward premium) and CIP deviations for each currency
        currency_mappings = {
            'usd': {
                'forward': 'ForwardRateEURtoUSD',
                'spot': 'SpotRateEURtoUSD',
                'domestic_rate': 'EUROBIR',
                'foreign_rate': 'USDTreasuryRate'
            },
            'gbp': {
                'forward': 'ForwardRateEURtoGBP',
                'spot': 'GBP',
                'domestic_rate': 'EUROBIR',
                'foreign_rate': 'GBPOvernightRate'
            },
            'jpy': {
                'forward': 'ForwardRateEURtoJPY',
                'spot': 'JPY',
                'domestic_rate': 'EUROBIR',
                'foreign_rate': 'JPYOvernightRate'
            },
            'sek': {
                'forward': 'ForwardRateEURtoSEK',
                'spot': 'SEK',
                'domestic_rate': 'EUROBIR',
                'foreign_rate': 'SEKTreasuryRate'
            },
            'chf': {
                'forward': 'ForwardRateEURtoCHF',
                'spot': 'CHF',
                'domestic_rate': 'EUROBIR',
                'foreign_rate': 'CHFTreasuryRate'
            }
        }
        
        for currency, mapping in currency_mappings.items():
            if all(col in df.columns for col in mapping.values()):
                # Calculate forward premium (rho)
                df[f"rho_{currency}"] = (
                    np.log(df[mapping['forward']]) - 
                    np.log(df[mapping['spot']])
                )
                
                # Calculate CIP deviation (x)
                df[f"x_{currency}"] = (
                    df[mapping['domestic_rate']] - 
                    df[mapping['foreign_rate']] - 
                    df[f"rho_{currency}"]
                )
            else:
                logger.warning(f"Missing data for {currency} CIP calculation")
        
        return df
    
    def calculate_trading_costs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trading costs from bid-ask spreads."""
        df = df.copy()
        
        cost_mappings = {
            'usd': {
                'high': 'ForwardRateEURtoUSD_high',
                'low': 'ForwardRateEURtoUSD_low',
                'spot': 'SpotRateEURtoUSD'
            },
            'gbp': {
                'high': 'ForwardRateEURtoGBP_high',
                'low': 'ForwardRateEURtoGBP_low',
                'spot': 'GBP'
            },
            'jpy': {
                'high': 'ForwardRateEURtoJPY_high',
                'low': 'ForwardRateEURtoJPY_low',
                'spot': 'JPY'
            },
            'sek': {
                'high': 'ForwardRateEURtoSEK_high',
                'low': 'ForwardRateEURtoSEK_low',
                'spot': 'SEK'
            },
            'chf': {
                'high': 'ForwardRateEURtoCHF_high',
                'low': 'ForwardRateEURtoCHF_low',
                'spot': 'CHF'
            }
        }
        
        for currency, mapping in cost_mappings.items():
            if all(col in df.columns for col in mapping.values()):
                # Calculate forward spread
                df[f"ForwardSpread_{currency}"] = (
                    df[mapping['high']] - df[mapping['low']]
                )
                
                # Calculate trading cost (normalized by spot rate)
                df[f"TradingCost_{currency}"] = (
                    df[f"ForwardSpread_{currency}"] / df[mapping['spot']]
                )
        
        return df
    
    def clean_trading_costs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean trading cost data by handling zeros and missing values."""
        df = df.copy()
        
        trading_cost_cols = [
            "TradingCost_usd", "TradingCost_gbp", "TradingCost_jpy",
            "TradingCost_sek", "TradingCost_chf"
        ]
        
        for col in trading_cost_cols:
            if col in df.columns:
                # Replace 0 with NaN and forward-fill
                df[col] = df[col].replace(0, np.nan)
                df[col] = df[col].fillna(method='ffill')
        
        return df


class QuantileEstimator:
    """Handles quantile estimation using kernel methods."""
    
    @staticmethod
    def gaussian_kernel(dist: np.ndarray, h: float) -> np.ndarray:
        """Gaussian kernel function."""
        return np.exp(-0.5 * (dist / h)**2)
    
    def kernel_quantile_estimate(self, X_train: np.ndarray, y_train: np.ndarray, 
                               X_pred: np.ndarray, tau: float, h: float) -> np.ndarray:
        """Estimate quantiles using kernel methods."""
        dists = cdist(X_pred, X_train)
        weights = self.gaussian_kernel(dists, h)
        quantiles = []
        
        for i in range(weights.shape[0]):
            w = weights[i]
            if np.all(w == 0) or np.isnan(w).any():
                quantiles.append(np.nan)
                continue
            
            sorted_idx = np.argsort(y_train)
            y_sorted = y_train[sorted_idx]
            w_sorted = w[sorted_idx]
            w_cumsum = np.cumsum(w_sorted)
            
            if w_cumsum[-1] == 0:
                quantiles.append(np.nan)
                continue
            
            w_cumsum /= w_cumsum[-1]
            idx = np.searchsorted(w_cumsum, tau)
            q_value = y_sorted[min(idx, len(y_sorted) - 1)]
            quantiles.append(float(q_value))
        
        return np.array(quantiles)


class CurrencyAnalyzer:
    """Analyzes individual currency CIP deviations."""
    
    def __init__(self, config: Dict = None):
        self.config = config or ANALYSIS_CONFIG
        self.quantile_estimator = QuantileEstimator()
    
    def analyze_currency(self, data: pd.DataFrame, currency: str) -> Dict:
        """Perform complete analysis for a single currency."""
        logger.info(f"Analyzing {currency.upper()}")
        
        currency_config = CURRENCIES[currency]
        x_col = currency_config["x"]
        cost_col = currency_config["trading_cost"]
        spot_col = currency_config["spot"]
        lag_col = f"{x_col}_lag_1"
        
        # Create lag column
        data[lag_col] = data[x_col].shift(1)
        
        # Clean data
        clean_data = data.dropna(subset=[x_col, cost_col, lag_col]).copy()
        
        # Winsorize trading cost
        clean_data[cost_col] = winsorize(clean_data[cost_col].values, limits=[0.01, 0.01])
        
        # Prepare features for quantile estimation
        block_cols = ["bond_block", "equity_block", "fin_block", "money_block"]
        macro_features = [lag_col] + [col for col in block_cols if col in clean_data.columns]
        
        # Standardize and apply PCA
        X_macro = StandardScaler().fit_transform(clean_data[macro_features])
        X_pca = PCA(n_components=self.config['pca_params']['n_components']).fit_transform(X_macro)
        
        # Prepare final feature matrix
        y = clean_data[x_col].values
        tc_scaled = StandardScaler().fit_transform(clean_data[cost_col].values.reshape(-1, 1))
        
        # Remove rows with NaN values
        mask = ~(np.isnan(X_macro).any(axis=1) | np.isnan(tc_scaled).any(axis=1) | np.isnan(y))
        X_final = np.hstack([X_pca, tc_scaled])[mask]
        y_clean = y[mask]
        data_clean = clean_data.loc[mask].reset_index(drop=True)
        
        # Standardize final features
        X_final_scaled = StandardScaler().fit_transform(X_final)
        
        # Estimate quantiles
        tau_values = self.config['quantile_params']['tau_values']
        bandwidth = self.config['quantile_params']['bandwidth']
        
        for tau in tau_values:
            quantile_col = f"Q{tau*100:.1f}"
            data_clean[quantile_col] = self.quantile_estimator.kernel_quantile_estimate(
                X_final_scaled, y_clean, X_final_scaled, tau, bandwidth
            )
        
        # Calculate band width and stress indicators
        data_clean = data_clean.dropna(subset=["Q95.0", "Q5.0"])
        data_clean["Band_Width"] = (data_clean["Q95.0"] - data_clean["Q5.0"]).clip(lower=1e-4)
        
        # Calculate CIP stress
        eps = 1e-6
        data_clean["CIP_Stress"] = np.where(
            (y_clean < (data_clean["Q5.0"] - eps)) | (y_clean > (data_clean["Q95.0"] + eps)),
            np.maximum(
                np.abs(y_clean - data_clean["Q5.0"]), 
                np.abs(y_clean - data_clean["Q95.0"])
            ),
            0
        )
        data_clean["CIP_Stress_Log"] = np.log1p(data_clean["CIP_Stress"])
        
        # Calculate volatility measures
        data_clean["Date"] = pd.to_datetime(data_clean["Date"], errors='coerce')
        data_clean = data_clean.set_index("Date")
        
        if spot_col in data_clean.columns:
            spot_rate = data_clean[spot_col].dropna()
            data_clean["Log_Returns"] = np.log(spot_rate / spot_rate.shift(1))
            data_clean["Rolling_Volatility"] = (
                data_clean["Log_Returns"].rolling(20).std() * 
                np.sqrt(self.config['volatility_params']['annualization_factor'])
            )
        
        # Scale measures
        data_clean["Band_Width_scaled"] = self._min_max_scale(data_clean["Band_Width"])
        if "Rolling_Volatility" in data_clean.columns:
            data_clean["FX_RealizedVol_scaled"] = self._min_max_scale(data_clean["Rolling_Volatility"])
        
        # Perform cointegration analysis
        cointegration_result = self._perform_cointegration_analysis(data_clean)
        
        return {
            'data': data_clean,
            'cointegration': cointegration_result,
            'currency': currency
        }
    
    def _min_max_scale(self, series: pd.Series) -> pd.Series:
        """Min-max scaling for a series."""
        return (series - series.min()) / (series.max() - series.min())
    
    def _perform_cointegration_analysis(self, data: pd.DataFrame) -> Dict:
        """Perform cointegration analysis between bandwidth and volatility."""
        if not all(col in data.columns for col in ["Band_Width_scaled", "FX_RealizedVol_scaled"]):
            return {"error": "Missing required columns for cointegration analysis"}
        
        comp_data = data.dropna(subset=["Band_Width_scaled", "FX_RealizedVol_scaled"])
        
        # Cointegration test
        try:
            coint_stat, p_value, _ = coint(
                comp_data["FX_RealizedVol_scaled"], 
                comp_data["Band_Width_scaled"]
            )
            
            # Decide on model specification
            if p_value > self.config['cointegration_threshold']:
                logger.info("No cointegration. Using first differences.")
                comp_data["BW"] = comp_data["Band_Width_scaled"].diff()
                comp_data["VOL"] = comp_data["FX_RealizedVol_scaled"].diff()
            else:
                logger.info("Cointegrated. Using levels.")
                comp_data["BW"] = comp_data["Band_Width_scaled"]
                comp_data["VOL"] = comp_data["FX_RealizedVol_scaled"]
            
            # OLS regression
            comp_data_clean = comp_data.dropna(subset=["BW", "VOL"])
            X = sm.add_constant(comp_data_clean["BW"])
            y = comp_data_clean["VOL"]
            model = sm.OLS(y, X).fit()
            
            return {
                'cointegration_stat': coint_stat,
                'p_value': p_value,
                'model': model,
                'data': comp_data_clean
            }
            
        except Exception as e:
            logger.error(f"Cointegration analysis failed: {e}")
            return {"error": str(e)}
