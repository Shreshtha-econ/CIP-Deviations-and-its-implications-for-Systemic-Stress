# Master Thesis Project: Covered Interest Parity Analysis & Systemic Risk Indicators

## 📋 Project Overview

This project implements a comprehensive financial analysis system for studying Covered Interest Parity (CIP) deviations across multiple currencies and constructing systemic risk indicators. The analysis covers five major currencies (USD, GBP, JPY, SEK, CHF) and implements the ECB's Composite Indicator of Systemic Stress (CISS) methodology.

## 🏗️ Project Structure

```
master_thesis_project/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config/                      # Configuration files
├── src/                         # Source code
│   ├── data/                   # Data loading and preprocessing
│   ├── analysis/               # Analysis modules
│   ├── visualization/          # Plotting and visualization
│   ├── api/                    # Flask REST API (NEW!)
│   └── utils/                  # Utility functions
├── tests/                      # Test suite
├── data/                       # Data files
│   ├── raw/                   # Original Excel files
│   ├── processed/             # Cleaned CSV files
│   └── results/               # Analysis outputs
├── notebooks/                  # Jupyter notebooks
├── docs/                       # Documentation
└── scripts/                    # Execution scripts
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to the project directory
cd "C:\Users\saurabh.dubey\Documents\Masters\Shre\Master Thesis Project"

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Complete Analysis

```bash
# Run the complete analysis pipeline
python scripts/run_analysis.py

# Force reload of data (if needed)
python scripts/run_analysis.py --force-reload
```

### 3. Start Flask API (NEW!)

```bash
# Start the REST API server
python src/api/app.py

# Or use the startup script
python scripts/start_api.py

# Access API documentation at: http://localhost:5000
```

### 3. Start Web Interface

```bash
# Start Flask API server
python src/api/app.py
```

Then open your browser to `http://localhost:5000` to view the visualizations.

## 📊 Key Features

### 1. **Data Management**
- **Automated data loading** from Excel files
- **Data validation** and quality checks
- **Missing value handling** strategies
- **Data preprocessing** pipeline

### 2. **CIP Analysis**
- **Covered Interest Parity deviation** calculations
- **Trading cost estimation** from bid-ask spreads
- **Quantile estimation** using kernel methods
- **Neutral band construction** for each currency

### 3. **Statistical Analysis**
- **Cointegration testing** between variables
- **OLS regression** analysis
- **Time series modeling** capabilities
- **Cross-correlation** analysis

### 4. **Risk Indicators**
- **CISS construction** using PCA methodology
- **Block indicator** creation (Money, Bond, Equity, FX)
- **Comparison** with official ECB CISS
- **Systemic stress** measurement

### 5. **Web API & Visualization**
- **RESTful Flask API** with 9+ endpoints
- **Interactive documentation** at root URL
- **JSON responses** with comprehensive data
- **CORS support** for web applications
- **Rate limiting** and caching
- **Multi-format export** capabilities
- **Professional charts** and plots
- **Export capabilities** for analysis results

## 💡 Usage Examples

### Basic Analysis

```python
from scripts.run_analysis import MasterThesisAnalyzer

# Initialize analyzer
analyzer = MasterThesisAnalyzer()

# Run complete analysis
results = analyzer.run_complete_analysis()

# Access results
currency_analysis = results['currency_analysis']
risk_analysis = results['risk_analysis']
```

### Individual Currency Analysis

```python
from src.analysis.cip_analysis import CurrencyAnalyzer

analyzer = CurrencyAnalyzer()
usd_results = analyzer.analyze_currency(data, 'usd')
```

### Data Loading

```python
from src.data.loader import DataLoader, DataMerger

# Load specific data types
loader = DataLoader()
forward_rates = loader.load_forward_rates()
spot_rates = loader.load_spot_rates()

# Create merged dataset
merger = DataMerger()
master_data = merger.create_master_dataset()
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test categories
pytest tests/test_analysis/
pytest tests/test_data/
```

## 📈 API Endpoints

The Flask API provides the following endpoints:

- `GET /api/cip_deviations` - CIP deviation plots
- `GET /api/bandwidth_volatility` - Bandwidth vs volatility analysis
- `GET /api/cip_deviation_vs_band` - CIP deviation vs neutral band
- `GET /api/ecb_ciss` - ECB CISS index plot
- `GET /api/ciss_comparison` - Official vs constructed CISS comparison
- `GET /api/cross_correlation` - Cross-correlation analysis

## 🔧 Configuration

Key settings can be modified in `config/settings.py`:

```python
# Analysis parameters
ANALYSIS_CONFIG = {
    'quantile_params': {
        'tau_values': [0.05, 0.95],
        'bandwidth': 3
    },
    'cointegration_threshold': 0.15,
    'volatility_params': {
        'window': 21,
        'annualization_factor': 252
    }
}
```

## 📝 Key Improvements Made

### 1. **Code Organization**
- ✅ Modular structure with separate concerns
- ✅ Object-oriented design patterns
- ✅ Clear separation of data, analysis, and visualization
- ✅ Configuration management system

### 2. **Data Management**
- ✅ Centralized data loading with error handling
- ✅ Data validation and quality checks
- ✅ Processed data caching system
- ✅ Flexible data source configuration

### 3. **Testing Infrastructure**
- ✅ Comprehensive test suite with pytest
- ✅ Unit tests for all major components
- ✅ Integration tests for workflows
- ✅ Performance testing for large datasets

### 4. **Error Handling & Logging**
- ✅ Structured logging throughout the application
- ✅ Graceful error handling and recovery
- ✅ Detailed error messages and debugging info
- ✅ Analysis pipeline monitoring

### 5. **Documentation & Usability**
- ✅ Clear README with examples
- ✅ Inline code documentation
- ✅ Configuration documentation
- ✅ API documentation

## 📊 Analysis Outputs

The analysis generates several key outputs:

1. **CIP Deviation Analysis**
   - Deviation calculations for 5 currencies
   - Neutral band estimates
   - Trading cost analysis

2. **Systemic Risk Indicators**
   - Constructed CISS index
   - Comparison with official ECB CISS
   - Block indicator decomposition

3. **Statistical Results**
   - Cointegration test results
   - OLS regression summaries
   - Cross-correlation analysis

4. **Visualizations**
   - Time series plots
   - Comparison charts
   - Risk indicator dashboards

## 🚨 Common Issues & Solutions

### Data Loading Issues
```bash
# If Excel files cannot be loaded
pip install --upgrade openpyxl xlrd

# If specific files are missing
python scripts/run_analysis.py --force-reload
```

### Memory Issues
```bash
# For large datasets, consider using chunking
# Modify ANALYSIS_CONFIG in config/settings.py
```

### API Issues
```bash
# If Flask server doesn't start
pip install --upgrade Flask Flask-CORS
```

## 🔄 Development Workflow

1. **Make changes** to source code
2. **Run tests** to ensure functionality
3. **Update documentation** if needed
4. **Test the complete pipeline**
5. **Commit changes** with clear messages

## 📞 Support & Maintenance

- **Logging**: Check `analysis.log` for detailed execution logs
- **Testing**: Run `pytest` before any major changes
- **Performance**: Monitor memory usage with large datasets
- **Configuration**: Adjust parameters in `config/settings.py`

## 🎯 Future Enhancements

- [ ] Add more sophisticated volatility models (GARCH)
- [ ] Implement real-time data feeds
- [ ] Add more currency pairs
- [ ] Enhance web interface with interactive dashboards
- [ ] Add automated report generation
- [ ] Implement model backtesting framework

---

**Author**: Shreshtha  
**Institution**: University of Amsterdam  
**Project**: Master Thesis - Financial Risk Analysis  
**Last Updated**: December 2024
