##Installation Commands (Run first 4 once to set up packages)
# Install GARCH and volatility modeling tools from the arch package

# Install Quandl API wrapper for accessing macroeconomic and financial datasets

# Install Yahoo Finance data downloader (yfinance), plus pandas for data handling and matplotlib for plotting

# Install openpyxl to handle Excel files (.xlsx), and lxml to parse XML/HTML



##Data Loading & Handling
# pandas is the core library for data analysis and manipulation using DataFrames
import pandas as pd
# numpy provides support for high-performance numerical computations and array operations
import numpy as np
# xml.etree.ElementTree is used to parse XML files — often used to load structured data like macroeconomic feeds
import xml.etree.ElementTree as ET


##Data Acquisition
# quandl provides access to economic, financial, and alternative datasets via the Quandl API
import quandl
# yfinance allows downloading of historical market data from Yahoo Finance (e.g., stock prices, index levels)
import yfinance as yf


##Visualization
# matplotlib.pyplot is the base library for data visualization in Python — supports line plots, histograms, etc.
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# seaborn builds on matplotlib, providing cleaner syntax and advanced statistical charting (e.g., heatmaps, violin plots)
import seaborn as sns


##Time Series Modeling
# SARIMAX supports seasonal ARIMA models with exogenous regressors — ideal for time series with seasonality and macro drivers
from statsmodels.tsa.statespace.sarimax import SARIMAX
# ARIMA is a widely used model that combines AR (autoregression), I (differencing), and MA (moving average)
from statsmodels.tsa.arima.model import ARIMA
# AutoReg is a simple autoregressive model, useful for univariate time series with autocorrelation
from statsmodels.tsa.ar_model import AutoReg
# VAR is a model for multivariate time series where each variable can depend on past values of all variables
from statsmodels.tsa.api import VAR
# adfuller is the Augmented Dickey-Fuller test to check for stationarity in time series data
from statsmodels.tsa.stattools import adfuller
# plot_acf and plot_pacf visualize autocorrelation and partial autocorrelation, helping to select ARIMA lags
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


##Volatility Modeling
# arch_model builds ARCH and GARCH models to forecast volatility — critical in finance for modeling returns risk
from arch import arch_model


##Regression & Econometrics
# statsmodels.api offers tools for linear regression, hypothesis testing, and general econometric modeling
import statsmodels.api as sm
# variance_inflation_factor helps detect multicollinearity among regressors by calculating the VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
# add_constant adds an intercept to your design matrix — required for most regression models
from statsmodels.tools.tools import add_constant
# KernelReg provides non-parametric regression estimation, useful for modeling nonlinear relationships
from statsmodels.nonparametric.kernel_regression import KernelReg
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import ccf

##Machine Learning & Feature Engineering
# PCA reduces dimensionality of datasets while retaining most variance — useful for macroeconomic indicators or large features
from sklearn.decomposition import PCA
# StandardScaler standardizes data to zero mean and unit variance — essential before PCA or ML models
from sklearn.preprocessing import StandardScaler


##Distance Metrics & Similarity
# cdist calculates pairwise distances (e.g., Euclidean, cosine) between sets of observations — used in clustering or similarity search
from scipy.spatial.distance import cdist
from scipy.stats.mstats import winsorize


##Functional Programming Tools
# reduce is used to apply a function cumulatively across a list — often used to merge multiple DataFrames or apply chained operations
from functools import reduce

import io
import base64

#Forward Exchange Rate (USD to EUR)
df_ForwardRateUSDtoEUR = pd.read_excel('data/ForwardRateUSDtoEUR.xlsx', engine='openpyxl', header=1)
df_ForwardRateUSDtoEUR.head()

#Spot Exchange Rate (USD to EUR)

df_SpotRateUSDtoEUR = pd.read_excel('data/SpotRateUSDtoEUR.xlsx', engine='openpyxl', header=1)
df_SpotRateUSDtoEUR.head()

#U.S. Risk-Free Rate (3-Month Treasury Yield)
 
df_USDTreasuryRate = pd.read_excel('data//USDTreasuryRate.xlsx', engine='openpyxl', header=1)
df_USDTreasuryRate.head()

#Euro Area Risk-Free Rate (3-Month EURIBOR)

df_EUROBIR = pd.read_excel('data//EUROBIR.xlsx', engine='openpyxl', header=1)
df_EUROBIR.head()

#GDP_growth
 
df_GDP_growth = pd.read_excel('data//GDP_growth.xlsx', engine='openpyxl', header=1)
df_GDP_growth.head()

#Euro Overnight Rate (EONIA/ESTR)-1
 
df_EONIA_ESTR = pd.read_excel('data/1.xlsx', engine='openpyxl')
df_EONIA_ESTR.head()

##Euro Area Risk-Free Rate (3-Month EURIBOR)-2

df_EURIBOR = pd.read_excel('data/2.xlsx', engine='openpyxl')
df_EURIBOR.head()

#Investment Grade (IG) Corporate Bond Option-Adjusted Spread (OAS)-4

df_IG_OAS= pd.read_excel('data/4.xlsx', engine='openpyxl')
df_IG_OAS.head()

#yield on a high-yield (HY) corporate bond index -5_1
 
df_YieldCorporateBondIndex = pd.read_excel('data/5_1.xlsx', engine='openpyxl')
df_YieldCorporateBondIndex.head()

#yield on AAA-rated bond index-5_2
 
df_YieldonAAAratedbondindex = pd.read_excel('data/5_2.xlsx', engine='openpyxl')
df_YieldonAAAratedbondindex.head()

# yield on Italian long-term government bonds-6_1

df_Italiangovbonds = pd.read_excel('data/6_1.xlsx', engine='openpyxl')
df_Italiangovbonds.head()

# yield on German long-term government bonds-6_2

df_Germangovbonds = pd.read_excel('data/6_2.xlsx', engine='openpyxl')
df_Germangovbonds.head()

#The VSTOXX index measures the expected 30-day volatility of the Euro STOXX 50 index, based on options price-8

df_VSTOXX = pd.read_excel('data/8.xlsx', engine='openpyxl')
df_VSTOXX.head()

#German stock idex(DAX)-10_1
 
df_DAX = pd.read_excel('data/10_1.xlsx', engine='openpyxl')
df_DAX.head()

#French stock idex(CAC)-10_2

df_CAC = pd.read_excel('data/10_2.xlsx', engine='openpyxl')
df_CAC.head()

#The SX7E is the EURO STOXX Banks index, tracking the performance of leading European banking stocks-11

df_SX7E = pd.read_excel('data/11.xlsx', engine='openpyxl')
df_SX7E.head()

#Spot Exchange Rate (EUR to USD)

df_usd = pd.read_excel('data/usd.xlsx', engine='openpyxl')
df_usd.head()

#Spot Exchange Rate (EUR to GBP)

df_gbp = pd.read_excel('data/gbp.xlsx', engine='openpyxl')
df_gbp.head()

#Spot Exchange Rate (EUR to JPY)

df_jpy = pd.read_excel('data/jpy.xlsx', engine='openpyxl')
df_jpy.head()

#Spot Exchange Rate (EUR to SEK)

df_sek = pd.read_excel('data/sek.xlsx', engine='openpyxl')
df_sek.head()

#Spot Exchange Rate (EUR to CHF)

df_chf = pd.read_excel('data/chf.xlsx', engine='openpyxl')
df_chf.head()



# EUR to CHF Forward Rate

df_eur_chf = pd.read_excel('data/Forward Rates all currencies/Cleaned/ForwardRateEURtoCHF.xlsx', engine='openpyxl')
df_eur_chf.head()

# EUR to GBP Forward Rate

df_eur_gbp = pd.read_excel('data/Forward Rates all currencies/Cleaned/ForwardRateEURtoGBP.xlsx', engine='openpyxl')
df_eur_gbp.head()

# EUR to JPY Forward Rate

df_eur_jpy = pd.read_excel('data/Forward Rates all currencies/Cleaned/ForwardRateEURtoJPY.xlsx', engine='openpyxl')
df_eur_jpy.head()

# EUR to SEK Forward Rate

df_eur_sek = pd.read_excel('data/Forward Rates all currencies/Cleaned/ForwardRateEURtoSEK.xlsx', engine='openpyxl')
df_eur_sek.head()



# CHF Interest Rate

df_CHFTreasuryRate = pd.read_excel('data/Interest Rates all currencies/CHFTreasuryRate.xlsx', engine='openpyxl')
df_CHFTreasuryRate.head()

# GBP Interest Rate

df_GBPOvernightRate = pd.read_excel('data/Interest Rates all currencies/GBPOvernightRate.xlsx', engine='openpyxl')
df_GBPOvernightRate.head()

# JPY Interest Rate

df_JPYOvernightRate = pd.read_excel('data/Interest Rates all currencies/JPYOvernightRate.xlsx', engine='openpyxl')
df_JPYOvernightRate.head()

# SEK Interest Rate

df_SEKTreasuryRate = pd.read_excel('data/Interest Rates all currencies/SEKTreasuryRate.xlsx', engine='openpyxl')
df_SEKTreasuryRate.head()

def expand_monthly_to_daily(df, date_col='Date', rate_col='JPYOvernightRate'):
    return pd.concat([pd.DataFrame({date_col: pd.date_range(pd.to_datetime(row[date_col]).replace(day=1), pd.to_datetime(row[date_col]).replace(day=1) + pd.offsets.MonthEnd(0)), rate_col: row[rate_col]}) for _, row in df.iterrows()]).reset_index(drop=True)

# 1. Expand monthly JPY Overnight Rate to daily
df_JPYOvernightRate = expand_monthly_to_daily(df_JPYOvernightRate, date_col='Date', rate_col='JPYOvernightRate')
df_JPYOvernightRate.head()

#MERGING FIRST DATASET
# Step 1: Create the list of DataFrames
dfs = [
    df_ForwardRateUSDtoEUR, df_SpotRateUSDtoEUR, df_USDTreasuryRate, df_EUROBIR,
    df_usd, df_jpy, df_sek, df_chf, df_gbp,
    df_eur_chf, df_eur_gbp, df_eur_jpy, df_eur_sek,
    df_CHFTreasuryRate, df_GBPOvernightRate, df_JPYOvernightRate, df_SEKTreasuryRate
]

# Step 2: Ensure 'Date' is datetime and set as index
for i, df in enumerate(dfs):
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])  # parse
        df.set_index('Date', inplace=True)       # set as index
    elif df.index.name != 'Date':
        raise ValueError(f"DataFrame {i} is missing 'Date' index or column")

# Step 3: Merge them on the index
merged = reduce(lambda left, right: pd.merge(
    left, right, how='outer', left_index=True, right_index=True), dfs)

# Step 4: Optional - sort by Date
merged.sort_index(inplace=True)

merged = merged.drop(columns=['level_0', 'index'], errors='ignore')
merged = merged[merged.index >= '1999-01-01']
merged = merged.reset_index()
# Done!
merged.head(1000)

######3

#VARIABLE TRANSFORMATIONS BELOW
#Ensuring all variables are numeric
cols_to_convert = merged.columns.difference(['Date', 'Year'])
for col in cols_to_convert:
    merged[col] = pd.to_numeric(merged[col], errors='coerce')
merged.head(1000)

#convert variables from usd/eur to eur/usd

# Invert spot rate and forward rate
merged['SpotRateEURtoUSD'] = 1 / merged['SpotRateUSDtoEUR']
merged['ForwardRateEURtoUSD'] = 1 / merged['ForwardRateUSDtoEUR']

# For high/low forward rates, invert and swap low and high
merged['ForwardRateEURtoUSD_low'] = 1 / merged['ForwardRateUSDtoEUR_high']   
merged['ForwardRateEURtoUSD_high'] = 1 / merged['ForwardRateUSDtoEUR_low']   
merged.head(1000)

#Calculating CIP deviation for all currencies according to Hernandez Paper



merged["rho_usd"] = np.log(merged["ForwardRateEURtoUSD"]) - np.log(merged["SpotRateEURtoUSD"])
merged["x_usd"] = merged["EUROBIR"] - merged["USDTreasuryRate"] - merged["rho_usd"]

merged["rho_gbp"] = np.log(merged["ForwardRateEURtoGBP"]) - np.log(merged["GBP"])
merged["x_gbp"] = merged["EUROBIR"] - merged["GBPOvernightRate"] - merged["rho_gbp"]

merged["rho_jpy"] = np.log(merged["ForwardRateEURtoJPY"]) - np.log(merged["JPY"])
merged["x_jpy"] = merged["EUROBIR"] - merged["JPYOvernightRate"] - merged["rho_jpy"]

merged["rho_sek"] = np.log(merged["ForwardRateEURtoSEK"]) - np.log(merged["SEK"])
merged["x_sek"] = merged["EUROBIR"] - merged["SEKTreasuryRate"] - merged["rho_sek"]

merged["rho_chf"] = np.log(merged["ForwardRateEURtoCHF"]) - np.log(merged["CHF"])
merged["x_chf"] = merged["EUROBIR"] - merged["CHFTreasuryRate"] - merged["rho_chf"]

merged.head()

#Compute trading costs from bid-ask spread

merged["ForwardSpread_usd"] = merged["ForwardRateEURtoUSD_high"] - merged["ForwardRateEURtoUSD_low"]
merged["TradingCost_usd"] = merged["ForwardSpread_usd"] / merged["SpotRateEURtoUSD"]  # Approx basis in returns

merged["ForwardSpread_gbp"] = merged["ForwardRateEURtoGBP_high"] - merged["ForwardRateEURtoGBP_low"]
merged["TradingCost_gbp"] = merged["ForwardSpread_gbp"] / merged["GBP"]  # Approx basis in returns

merged["ForwardSpread_jpy"] = merged["ForwardRateEURtoJPY_high"] - merged["ForwardRateEURtoJPY_low"]
merged["TradingCost_jpy"] = merged["ForwardSpread_jpy"] / merged["JPY"]  # Approx basis in returns

merged["ForwardSpread_sek"] = merged["ForwardRateEURtoSEK_high"] - merged["ForwardRateEURtoSEK_low"]
merged["TradingCost_sek"] = merged["ForwardSpread_sek"] / merged["SEK"]  # Approx basis in returns

merged["ForwardSpread_chf"] = merged["ForwardRateEURtoCHF_high"] - merged["ForwardRateEURtoCHF_low"]
merged["TradingCost_chf"] = merged["ForwardSpread_chf"] / merged["CHF"]  # Approx basis in returns


merged.head(1000)

#Forward filling NAN values in trading cost

# List all trading cost columns
trading_cost_cols = [
    "TradingCost_usd",
    "TradingCost_gbp",
    "TradingCost_jpy",
    "TradingCost_sek",
    "TradingCost_chf"
]

# Replace 0 with NaN and forward-fill for each
for col in trading_cost_cols:
    merged[col].replace(0, np.nan, inplace=True)
    merged[col].fillna(method="ffill", inplace=True)  # Or use merged[col].ffill(inplace=True)


#saving data file with cip deviation, forward rate, spote rate, interest rates
merged.to_csv('merged.csv', index=False)

# Preview result

merged.head(1000)

#PLOTTING CIP DEVIATIONS FOR ALL CURRENCIES


# Ensure 'Date' is datetime and sorted
merged["Date"] = pd.to_datetime(merged["Date"], errors='coerce')
merged = merged.sort_values("Date")
merged.to_csv('merged.csv', index=False)
# List of CIP deviation columns and labels
def plot_cip_deviations(merged):
    cip_columns = {
        "x_usd": "USD",
        "x_gbp": "GBP",
        "x_jpy": "JPY",
        "x_sek": "SEK",
        "x_chf": "CHF"
    }
    colors = ["royalblue", "darkorange", "green", "crimson", "purple"]

    plt.figure(figsize=(14, 7))
    for (col, label), color in zip(cip_columns.items(), colors):
        temp = merged[["Date", col]].dropna()
        if temp.empty:
            continue
        plt.plot(temp["Date"], temp[col], label=label, color=color, linewidth=2)

    plt.title("Covered Interest Parity (CIP) Deviations Over Time", fontsize=16, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("CIP Deviation", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    plt.axhline(0, color='black', linestyle=':', linewidth=1)
    plt.legend(loc="upper right", fontsize=11)
    plt.tight_layout()

    # Save figure to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    plt.close()
    buf.seek(0)

    # Encode buffer to base64 string
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

# DOWNLOAD AND PREPARE 'STOXX50E_realized_vol_21d'
# -----------------------------
stoxx = yf.download('^STOXX50E', start='2000-01-01', end='2024-12-31')
stoxx['log_return'] = np.log(stoxx['Close'] / stoxx['Close'].shift(1))
stoxx['realized_vol_21d'] = stoxx['log_return'].rolling(window=21).std() * np.sqrt(252)
STOXX50E_realized_vol_21d = stoxx.reset_index()[['Date', 'realized_vol_21d']]
STOXX50E_realized_vol_21d.rename(columns={'realized_vol_21d': 'STOXX50E_realized_vol_21d'}, inplace=True)

# -----------------------------
# DOWNLOAD iShares MSCI Europe Financials
# -----------------------------
data = yf.download('EUFN', start='2000-01-01').reset_index()
data.head()

#PREPARING DATASET

dfs = [df_EONIA_ESTR, df_EURIBOR, df_IG_OAS, df_YieldCorporateBondIndex, df_YieldonAAAratedbondindex, df_Italiangovbonds, df_Germangovbonds, STOXX50E_realized_vol_21d, df_DAX, df_CAC, df_SX7E]
dfs

# -----------------------------
# CLEAN EACH DATAFRAME
# -----------------------------
for i, df in enumerate(dfs):
    # Clean column names
    df.columns = df.columns.map(str).str.strip()

    if 'Date' not in df.columns:
        df.reset_index(inplace=True)

        # Find any column that looks like a date column after reset_index
        possible_date_cols = [col for col in df.columns if 'date' in col.lower()]

        if possible_date_cols:
            # Rename the first found date-like column to 'Date'
            df.rename(columns={possible_date_cols[0]: 'Date'}, inplace=True)
        else:
            # Fallback: rename the first column to 'Date'
            df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

    # Convert 'Date' column to datetime (coerce errors to NaT)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    dfs[i] = df

# -----------------------------
# MERGE ALL ON 'Date'
# -----------------------------
merged_df = dfs[0]
for df in dfs[1:]:
    merged_df = pd.merge(merged_df, df, on='Date', how='outer', suffixes=('', '_dup'))
    dup_cols = [col for col in merged_df.columns if col.endswith('_dup')]
    merged_df.drop(columns=dup_cols, inplace=True)

merged_df = merged_df.sort_values('Date').reset_index(drop=True)
merged_df.head()

# -----------------------------
#DERIVE VARIABLES NEEDED
# -----------------------------
merged_df['Year'] = merged_df['Date'].dt.year
merged_df['Month'] = merged_df['Date'].dt.month

# Merge monthly data for variables '2' and '3' if exist
if set(['Year', 'Month', '2', '3']).issubset(merged_df.columns):
    monthly_data = merged_df[['Year', 'Month', '2', '3']].drop_duplicates()
    merged_df = merged_df.drop(columns=['2', '3'], errors='ignore')
    merged_df = pd.merge(merged_df, monthly_data, on=['Year', 'Month'], how='left')

# Merge yearly data for variables '6.1' and '6.2' if exist
if set(['Year', '6.1', '6.2']).issubset(merged_df.columns):
    yearly_data = merged_df[['Year', '6.1', '6.2']].drop_duplicates()
    merged_df = merged_df.drop(columns=['6.1', '6.2'], errors='ignore')
    merged_df = pd.merge(merged_df, yearly_data, on='Year', how='left')

# Convert selected columns to numeric
for col in ['1', '2', '6.1', '6.2', '10.1', '10.2']:
    if col in merged_df.columns:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

# Calculate derived columns
merged_df['3'] = merged_df['2'] - merged_df['1'] if all(c in merged_df.columns for c in ['2', '1']) else np.nan
merged_df['5'] = merged_df['5_1'] - merged_df['5_2'] if all(c in merged_df.columns for c in ['5_1', '5_2']) else np.nan
merged_df['7'] = merged_df['6.1'] - merged_df['6.2'] if all(c in merged_df.columns for c in ['6.1', '6.2']) else np.nan

# Rolling correlation for variable '10'
if all(c in merged_df.columns for c in ['10.1', '10.2']):
    merged_df['10'] = merged_df['10.1'].rolling(window=30, min_periods=10).corr(merged_df['10.2'])

merged_df.head()

#FURTHER CLEANING

# Flatten any multi-level columns
merged_df.columns = [col if isinstance(col, str) else col[0] for col in merged_df.columns]
# Rename any unwanted 'Unnamed' columns (if present)
rename_map = {
    'Unnamed: 1': '1',
    'Unnamed: 2': '2',
    'Unnamed: 3': '3',
    'Unnamed: 4': '4',
    'Unnamed: 5': '5',
    'Unnamed: 6': '6_1',
    'Unnamed: 7': '6_2',
    'Unnamed: 8': '8',
    'Unnamed: 9': '10_1',
    'Unnamed: 10': '10_2',
}
merged_df.rename(columns=rename_map, inplace=True)

merged_df.head(10)

# Create MARKET Block Indicators ---
# Ensure numeric columns for averaging
merged_df['money_block'] = merged_df[['1', '2', '3']].mean(axis=1, skipna=True)
merged_df['bond_block'] = merged_df[['4', '5', '7']].mean(axis=1, skipna=True)
merged_df['equity_block'] = merged_df[["('STOXX50E_realized_vol_21d', '')"]].mean(axis=1, skipna=True)  # Updated: Include '11'
merged_df['fin_block'] = merged_df[['10']].mean(axis=1, skipna=True)

merged_df.head()

#Chaecking missing values

# Check for missing values in block indicators
print(merged_df[['money_block', 'bond_block', 'equity_block', 'fin_block']].isna().sum())

# --- Step 3: Clean and Prepare for Systemic Correlation ---
block_cols = ['money_block', 'bond_block', 'equity_block', 'fin_block']
existing_blocks = [col for col in block_cols if col in merged_df.columns]

# Drop rows with NaNs in these block columns
merged_df_clean = merged_df.dropna(subset=existing_blocks).reset_index(drop=True)

merged_df.head(5)

#Defining functions for analysis
def gaussian_kernel(dist, h):
    return np.exp(-0.5 * (dist / h)**2)

def kernel_quantile_estimate(X_train, y_train, X_pred, tau, h, kernel_func):
    dists = cdist(X_pred, X_train)
    weights = kernel_func(dists, h)
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

def min_max_scale(series):
    return (series - series.min()) / (series.max() - series.min())

# Define Currencies for final analysis

currencies = {
    "usd": {"x": "x_usd", "trading_cost": "TradingCost_usd", "spot": "USD"},
    "gbp": {"x": "x_gbp", "trading_cost": "TradingCost_gbp", "spot": "GBP"},
    "jpy": {"x": "x_jpy", "trading_cost": "TradingCost_jpy", "spot": "JPY"},
    "sek": {"x": "x_sek", "trading_cost": "TradingCost_sek", "spot": "SEK"},
    "chf": {"x": "x_chf", "trading_cost": "TradingCost_chf", "spot": "CHF"},
}

#Sort and reset merged (your base DF)
merged = merged.sort_values("Date").reset_index(drop=True)
merged.head(10000)

#Loop 1

# Create lag columns for all currencies

for curr, params in currencies.items():
    merged[f"{params['x']}_lag_1"] = merged[params["x"]].shift(1)

results = {}
data2_clean_all = None  # This will hold merged data with all currencies’ bandwidth

#Loop 2

for curr, params in currencies.items():
    print(f"\n\n====== Processing: {curr.upper()} ======\n")

    x_col = params["x"]
    cost_col = params["trading_cost"]
    spot_col = params["spot"]
    lag_col = f"{x_col}_lag_1"

    data = merged.copy().dropna(subset=[x_col, cost_col, lag_col])

    # Winsorize trading cost
    data[cost_col] = winsorize(data[cost_col].values, limits=[0.01, 0.01])

    # Macro features + PCA
    block_cols = ["bond_block", "equity_block", "fin_block", "money_block"]
    macro_features = [lag_col] + [col for col in block_cols if col in data.columns]
    X_macro = StandardScaler().fit_transform(data[macro_features])
    X_pca = PCA(n_components=0.95).fit_transform(X_macro)

    y = data[x_col].values
    tc_scaled = StandardScaler().fit_transform(data[cost_col].values.reshape(-1, 1))
    mask = ~(np.isnan(X_macro).any(axis=1) | np.isnan(tc_scaled).any(axis=1) | np.isnan(y))
    X_final = np.hstack([X_pca, tc_scaled])[mask]
    y_clean = y[mask]
    data2_clean = data.loc[mask].reset_index(drop=True)

    X_final_scaled = StandardScaler().fit_transform(X_final)

    # Quantile estimates
    for tau in [0.05, 0.95]:
        print(f"Estimating τ = {tau}")
        data2_clean[f"Q{tau*100:.1f}"] = kernel_quantile_estimate(
            X_final_scaled, y_clean, X_final_scaled, tau, 3, gaussian_kernel
        )

    data2_clean = data2_clean.dropna(subset=["Q95.0", "Q5.0"])
    data2_clean["Band_Width"] = (data2_clean["Q95.0"] - data2_clean["Q5.0"]).clip(lower=1e-4)

    eps = 1e-6
    data2_clean["CIP_Stress"] = np.where(
        (y_clean < (data2_clean["Q5.0"] - eps)) | (y_clean > (data2_clean["Q95.0"] + eps)),
        np.maximum(np.abs(y_clean - data2_clean["Q5.0"]), np.abs(y_clean - data2_clean["Q95.0"])),
        0
    )
    data2_clean["CIP_Stress_Log"] = np.log1p(data2_clean["CIP_Stress"])

    data2_clean["Date"] = pd.to_datetime(data2_clean["Date"], errors='coerce')
    data2_clean = data2_clean.set_index("Date")

    spot_rate = data2_clean[spot_col].dropna()
    data2_clean["Log_Returns"] = np.log(spot_rate / spot_rate.shift(1))
    data2_clean["Rolling_Volatility"] = data2_clean["Log_Returns"].rolling(20).std() * np.sqrt(252)

    data2_clean["Band_Width_scaled"] = min_max_scale(data2_clean["Band_Width"])
    data2_clean["FX_RealizedVol_scaled"] = min_max_scale(data2_clean["Rolling_Volatility"])

    comp_data = data2_clean.dropna(subset=["Band_Width_scaled", "FX_RealizedVol_scaled"])

    # Cointegration test
    coint_stat, p_value, _ = coint(comp_data["FX_RealizedVol_scaled"], comp_data["Band_Width_scaled"])
    print(f"Cointegration test p-value: {p_value:.4f}")

    if p_value > 0.15:
        print("=> No cointegration. Using first differences.")
        comp_data["BW"] = comp_data["Band_Width_scaled"].diff()
        comp_data["VOL"] = comp_data["FX_RealizedVol_scaled"].diff()
    else:
        print("=> Cointegrated. Using levels.")
        comp_data["BW"] = comp_data["Band_Width_scaled"]
        comp_data["VOL"] = comp_data["FX_RealizedVol_scaled"]

    comp_data = comp_data.dropna(subset=["BW", "VOL"])

    X = sm.add_constant(comp_data["BW"])
    y = comp_data["VOL"]
    model = sm.OLS(y, X).fit()
    print(model.summary())

    results[curr] = {"data": data2_clean, "model": model}

    # Prepare columns with currency suffix
    currency_suffix = f"_{curr}"
    cols_to_merge = ["Band_Width", "Band_Width_scaled", "CIP_Stress", "CIP_Stress_Log", "FX_RealizedVol_scaled"]

    data_to_merge = data2_clean[cols_to_merge].copy()
    data_to_merge.columns = [col + currency_suffix for col in cols_to_merge]
    comp_data.to_csv('comp_data.csv', index=False)
    # Merge into global data2_clean_all on index=Date
    if data2_clean_all is None:
        data2_clean_all = data_to_merge
    else:
        data2_clean_all = data2_clean_all.join(data_to_merge, how='outer')

    def plot_bandwidth_vs_volatility(comp_data, curr):
        plt.figure(figsize=(14,6))
        plt.plot(comp_data.index, comp_data["Band_Width_scaled"], label="Band Width", color="blue")
        plt.plot(comp_data.index, comp_data["FX_RealizedVol_scaled"], label="FX Realized Volatility", color="orange")
        plt.title(f"{curr.upper()} - Band Width vs FX Realized Volatility")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300)
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return img_base64

data2_clean_all.head(10000)

# Plot CIP deviation and neutral band for each currency
# results.to_csv('results.csv', index=False)  # 'results' is a dict, not a DataFrame. Save individual items if needed.
def plot_cip_deviation_vs_band(results, currencies):
    images = {}
    for curr, result in results.items():
        df = result["data"].copy()
        x_col = currencies[curr]["x"]

        plt.figure(figsize=(14, 6))
        plt.plot(df.index, df[x_col], label="CIP Deviation", color="black", linewidth=2)
        plt.plot(df.index, df["Q5.0"], label="5th Percentile", color="blue", linestyle="--")
        plt.plot(df.index, df["Q95.0"], label="95th Percentile", color="red", linestyle="--")
        plt.fill_between(df.index, df["Q5.0"], df["Q95.0"], color="lightgray", alpha=0.5, label="Neutral Band")

        plt.title(f"{curr.upper()} - CIP Deviation vs Estimated Neutral Band")
        plt.xlabel("Date")
        plt.ylabel("CIP Deviation")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300)
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')

        images[curr] = img_base64

    return images

#Starting the process to construct a new systemic risk indicator

# Prepare the FX block by combining scaled bandwidths from all currencies ---

# Assume data2_clean is your combined DataFrame with these columns:
# 'Band_Width_scaled_usd', 'Band_Width_scaled_gbp', 'Band_Width_scaled_jpy', 'Band_Width_scaled_sek', 'Band_Width_scaled_chf'

fx_cols = [
    "Band_Width_scaled_usd",
    "Band_Width_scaled_gbp",
    "Band_Width_scaled_jpy",
    "Band_Width_scaled_sek",
    "Band_Width_scaled_chf",
]

# Check columns exist
fx_cols = [col for col in fx_cols if col in data2_clean_all.columns]

# Drop rows with missing FX bandwidth data to ensure clean input for PCA
fx_data = data2_clean_all[fx_cols].dropna()

# Standardize FX block components (mean 0, std 1)
scaler_fx = StandardScaler()
fx_scaled = scaler_fx.fit_transform(fx_data)

# Apply PCA on FX scaled data and take first principal component as FX block score
pca_fx = PCA(n_components=1)
fx_block_scores = pca_fx.fit_transform(fx_scaled).flatten()

# Insert FX block score back into data2_clean aligned by index
fx_block_series = pd.Series(fx_block_scores, index=fx_data.index, name="fx_block")

# Add the fx_block to your main DataFrame (aligning on index)
data2_clean = data2_clean_all.join(fx_block_series, how='left')

data2_clean.head(10000)

# Prepare other blocks ---

# List all block columns expected in data2_clean (except fx_block just created)
# Make sure these blocks exist in your DataFrame
other_blocks = ["money_block", "bond_block", "equity_block", "fin_block"]

# Filter to those columns existing in data2_clean
other_blocks = [col for col in other_blocks if col in data2_clean.columns]

# Subset dataframe for blocks + fx_block and drop rows with missing data
blocks_df = data2_clean[other_blocks + ["fx_block"]].dropna()

# Standardize all blocks (mean 0, std 1)
scaler_blocks = StandardScaler()
blocks_scaled = scaler_blocks.fit_transform(blocks_df)

print (blocks_scaled)

# Compute CISS according to official ECB methodology ---

# Covariance matrix of standardized blocks
cov_matrix = np.cov(blocks_scaled.T)

# PCA on covariance matrix
pca = PCA(n_components=1)
# Fit PCA on covariance matrix by trick: eigen decomposition of covariance matrix
# We can just use pca.fit(blocks_scaled), first PC explains systemic risk
ciss_scores = pca.fit_transform(blocks_scaled).flatten()

# Normalize CISS to [0,1]
ciss_min, ciss_max = ciss_scores.min(), ciss_scores.max()
ciss_norm = (ciss_scores - ciss_min) / (ciss_max - ciss_min)

# Create final CISS Series aligned with blocks_df index
ciss_series = pd.Series(ciss_norm, index=blocks_df.index, name="CISS")

# Join CISS back to data2_clean
data2_clean = data2_clean.join(ciss_series, how="left")

# Plot the CISS index
data.to_csv('data.csv', index=False)
def plot_ecb_ciss(data):
    plt.figure(figsize=(14,6))
    plt.plot(data.index, data["CISS"], label="ECB CISS Index", color="red")
    plt.title("ECB Composite Indicator of Systemic Stress (CISS)")
    plt.xlabel("Date")
    plt.ylabel("Normalized CISS")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return img_base64

# Load official ECB CISS
official_ecb = pd.read_excel('data/ecb_ciss.xlsx', engine="openpyxl", header=1)
official_ecb["Date"] = pd.to_datetime(official_ecb["Date"], errors='coerce')
official_ecb = official_ecb.set_index("Date")

# Extract official CISS column (adjust column name if different)
official_ciss = official_ecb["ECB_CISS"]

# Align with your constructed CISS (assuming 'CISS' column exists in data2_clean)
comparison_df = pd.DataFrame({
    "Official ECB CISS": official_ciss,
    "Constructed CISS": data2_clean["CISS"]
}).dropna()

comparison_df.head(10000)

# Plot both on the same graph
comparison_df.to_csv('comparison_df.csv', index=False)
def plot_ciss_comparison(comparison_df):
    plt.figure(figsize=(14,6))
    plt.plot(comparison_df.index, comparison_df["Official ECB CISS"], label="Official ECB CISS", color="blue")
    plt.plot(comparison_df.index, comparison_df["Constructed CISS"], label="Constructed CISS", color="red", alpha=0.8)
    plt.title("ECB Composite Indicator of Systemic Stress (CISS) - Official vs Constructed")
    plt.xlabel("Date")
    plt.ylabel("Normalized CISS")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
# Save plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()  # Close the figure to free memory
    return img_base64


#Lead-Lag Correlation between newly constructed index snd official CISS

# Get the two series as NumPy arrays
official = comparison_df["Official ECB CISS"] - comparison_df["Official ECB CISS"].mean()
constructed = comparison_df["Constructed CISS"] - comparison_df["Constructed CISS"].mean()

# Define a max lag (e.g., 24 months)
max_lag = 24
lags = np.arange(-max_lag, max_lag + 1)

# Calculate cross-correlation manually for negative lags
ccf_values = [constructed.shift(-lag).corr(official) for lag in lags]
# Convert ccf_values list to DataFrame before saving
ccf_df = pd.DataFrame({'lag': lags, 'ccf_value': ccf_values})
ccf_df.to_csv('ccf_values.csv', index=False)
# Plot cross-correlation
def plot_cross_correlation(lags, ccf_values):
    plt.figure(figsize=(12, 5))
    plt.stem(lags, ccf_values)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='grey', linestyle='--')
    plt.title("Cross-Correlation: Constructed CISS vs Official ECB CISS")
    plt.xlabel("Lag (months)")
    plt.ylabel("Correlation")
    plt.grid(True)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return img_base64


from flask import Flask, jsonify
import pandas as pd

# Import your plotting functions here or define them above
# from your_module import plot_cip_deviations, plot_bandwidth_vs_volatility, ...

app = Flask(__name__)

# Declare global variables for data; initialize as None
merged = None
comp_data = None
curr = "usd"
results = None
currencies = None
data = None
comparison_df = None
lags = None
ccf_values = None


@app.route('/api/cip_deviations')
def api_cip_deviations():
    if merged is None:
        return jsonify({"error": "Data not loaded"}), 500
    img = plot_cip_deviations(merged)
    return jsonify({"image": img})

@app.route('/api/bandwidth_volatility')
def api_bandwidth_volatility():
    if comp_data is None or curr is None:
        return jsonify({"error": "Data not loaded"}), 500
    img = plot_bandwidth_vs_volatility(comp_data, curr)
    return jsonify({"image": img})

@app.route('/api/cip_deviation_vs_band')
def api_cip_deviation_vs_band():
    if results is None or currencies is None:
        return jsonify({"error": "Data not loaded"}), 500
    imgs = plot_cip_deviation_vs_band(results, currencies)
    return jsonify({"images": imgs})

@app.route('/api/ecb_ciss')
def api_ecb_ciss():
    if data is None:
        return jsonify({"error": "Data not loaded"}), 500
    img = plot_ecb_ciss(data)
    return jsonify({"image": img})

@app.route('/api/ciss_comparison')
def api_ciss_comparison():
    if comparison_df is None:
        return jsonify({"error": "Data not loaded"}), 500
    img = plot_ciss_comparison(comparison_df)
    return jsonify({"image": img})

@app.route('/api/cross_correlation')
def api_cross_correlation():
    if lags is None or ccf_values is None:
        return jsonify({"error": "Data not loaded"}), 500
    img = plot_cross_correlation(lags, ccf_values)
    return jsonify({"image": img})


def load_all_data():
    global merged, comp_data, curr, results, currencies, data, comparison_df, lags, ccf_values
    
    merged = pd.read_csv("merged.csv", parse_dates=["Date"])
    comp_data = pd.read_csv("comp_data.csv")
    curr = "usd"
    
    # Expert: Load or initialize results, currencies, lags, ccf_values as needed
    # For now, set to None or minimal defaults to avoid NameError
    results = None
    currencies = None
    lags = None
    ccf_values = None
    
    data = pd.read_csv("data.csv", parse_dates=["Date"])
    comparison_df = pd.read_csv("comparison_df.csv")

@app.route('/plot_cip_deviations_view')
def plot_cip_deviations_view():
    if merged is None:
        return "Data not loaded", 500
    img_base64 = plot_cip_deviations(merged)
    return f'''
        <html>
            <head><title>CIP Deviation Plot</title></head>
            <body>
                <h2>CIP Deviation Plot</h2>
                <img src="data:image/png;base64,{img_base64}" alt="CIP Deviation Plot">
            </body>
        </html>
    '''
if __name__ == '__main__':
    load_all_data()
    app.run(debug=True, host='0.0.0.0', port=5000)