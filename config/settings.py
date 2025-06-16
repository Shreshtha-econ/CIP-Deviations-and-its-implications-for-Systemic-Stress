# Configuration Settings
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

# Ensure directories exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data file configurations
DATA_FILES = {
    'forward_rates': {
        'usd_eur': 'ForwardRateUSDtoEUR.xlsx',
        'eur_chf': 'Forward Rates all currencies/Cleaned/ForwardRateEURtoCHF.xlsx',
        'eur_gbp': 'Forward Rates all currencies/Cleaned/ForwardRateEURtoGBP.xlsx',
        'eur_jpy': 'Forward Rates all currencies/Cleaned/ForwardRateEURtoJPY.xlsx',
        'eur_sek': 'Forward Rates all currencies/Cleaned/ForwardRateEURtoSEK.xlsx'
    },
    'spot_rates': {
        'usd_eur': 'SpotRateUSDtoEUR.xlsx',
        'usd': 'usd.xlsx',
        'gbp': 'gbp.xlsx',
        'jpy': 'jpy.xlsx',
        'sek': 'sek.xlsx',
        'chf': 'chf.xlsx'
    },
    'interest_rates': {
        'usd_treasury': 'USDTreasuryRate.xlsx',
        'euribor': 'EUROBIR.xlsx',
        'chf_treasury': 'Interest Rates all currencies/CHFTreasuryRate.xlsx',
        'gbp_overnight': 'Interest Rates all currencies/GBPOvernightRate.xlsx',
        'jpy_overnight': 'Interest Rates all currencies/JPYOvernightRate.xlsx',
        'sek_treasury': 'Interest Rates all currencies/SEKTreasuryRate.xlsx'
    },
    'market_indicators': {
        'eonia_estr': '1.xlsx',
        'euribor_alt': '2.xlsx',
        'ig_oas': '4.xlsx',
        'hy_bonds': '5_1.xlsx',
        'aaa_bonds': '5_2.xlsx',
        'italian_bonds': '6_1.xlsx',
        'german_bonds': '6_2.xlsx',
        'vstoxx': '8.xlsx',
        'dax': '10_1.xlsx',
        'cac': '10_2.xlsx',
        'sx7e': '11.xlsx',
        'ecb_ciss': 'ecb_ciss.xlsx'
    }
}

# Currency configurations
CURRENCIES = {
    "usd": {"x": "x_usd", "trading_cost": "TradingCost_usd", "spot": "USD"},
    "gbp": {"x": "x_gbp", "trading_cost": "TradingCost_gbp", "spot": "GBP"},
    "jpy": {"x": "x_jpy", "trading_cost": "TradingCost_jpy", "spot": "JPY"},
    "sek": {"x": "x_sek", "trading_cost": "TradingCost_sek", "spot": "SEK"},
    "chf": {"x": "x_chf", "trading_cost": "TradingCost_chf", "spot": "CHF"},
}

# Analysis parameters
ANALYSIS_CONFIG = {
    'date_range': {
        'start': '1999-01-01',
        'end': '2024-12-31'
    },
    'quantile_params': {
        'tau_values': [0.05, 0.95],
        'bandwidth': 3,
        'kernel_type': 'gaussian'
    },
    'pca_params': {
        'n_components': 0.95,
        'standardize': True
    },
    'volatility_params': {
        'window': 21,
        'annualization_factor': 252
    },
    'cointegration_threshold': 0.15
}

# Visualization settings
VIZ_CONFIG = {
    'figure_size': (14, 7),
    'dpi': 300,
    'color_palette': ["royalblue", "darkorange", "green", "crimson", "purple"],
    'style': 'seaborn-v0_8',
    'font_size': 12
}

# Flask API settings
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True
}
