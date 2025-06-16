# Flask API Documentation

## Overview

The Financial Analysis API provides RESTful endpoints for accessing Covered Interest Parity (CIP) analysis and systemic risk indicators. This API exposes the financial analysis capabilities of the Master Thesis Project through a modern web interface.

## Base URL

- **Development**: `http://localhost:5000`
- **Local Network**: `http://0.0.0.0:5000`

## Features

- 🏦 **CIP Analysis**: Calculate and analyze Covered Interest Parity deviations
- 📊 **Risk Indicators**: Compute systemic risk indicators using ECB CISS methodology
- 💱 **Multi-Currency Support**: EUR, USD, GBP, JPY, CHF, SEK
- 📈 **Historical Data**: 26+ years of financial data (1999-2025)
- 🔍 **Custom Analysis**: Flexible analysis endpoints with parameters
- 📋 **Data Export**: JSON responses with comprehensive metadata
- ⚡ **Caching**: Intelligent data caching for optimal performance

## Authentication

Currently, the API does not require authentication. This is suitable for development and research purposes.

## Response Format

All API responses follow a consistent format:

### Success Response
```json
{
  "status": "success",
  "message": "Success message",
  "data": {
    // Response data
  },
  "timestamp": "2025-01-27T10:30:00.000Z"
}
```

### Error Response
```json
{
  "status": "error",
  "message": "Error description",
  "timestamp": "2025-01-27T10:30:00.000Z",
  "details": {
    // Optional error details
  }
}
```

## Endpoints

### 1. Home Page
- **GET** `/`
- **Description**: API documentation homepage
- **Response**: HTML documentation page

### 2. API Status
- **GET** `/api/status`
- **Description**: Check API status and data availability
- **Response**: System status information

**Example Response:**
```json
{
  "status": "success",
  "data": {
    "api_version": "1.0.0",
    "status": "operational",
    "data_loaded": true,
    "data_shape": [6876, 45],
    "date_range": {
      "start": "1999-01-01T00:00:00",
      "end": "2025-01-01T00:00:00"
    },
    "currencies": ["EUR", "USD", "GBP", "JPY", "CHF", "SEK"],
    "cache_status": "active"
  }
}
```

### 3. Data Summary
- **GET** `/api/data/summary`
- **Description**: Get comprehensive dataset summary statistics
- **Response**: Dataset overview and statistics

**Example Response:**
```json
{
  "status": "success",
  "data": {
    "total_observations": 6876,
    "date_range": {
      "start": "1999-01-01T00:00:00",
      "end": "2025-01-01T00:00:00",
      "days": 9497
    },
    "columns": {
      "total": 45,
      "names": ["SpotRateUSDtoEUR", "ForwardRateUSDtoEUR", ...]
    },
    "missing_data": {
      "total_missing": 12543,
      "missing_by_column": {...}
    },
    "descriptive_statistics": {...}
  }
}
```

### 4. Currency Information
- **GET** `/api/data/currencies`
- **Description**: Get information about available currencies and their data coverage
- **Response**: Currency-specific data availability

**Example Response:**
```json
{
  "status": "success",
  "data": {
    "EUR": {
      "available_columns": ["ForwardRateUSDtoEUR", "SpotRateUSDtoEUR"],
      "data_points": 5234,
      "date_range": {
        "start": "1999-01-01T00:00:00",
        "end": "2024-12-31T00:00:00"
      },
      "completeness": 76.15
    },
    "USD": {...}
  }
}
```

### 5. CIP Deviations
- **GET** `/api/cip/deviations`
- **Description**: Calculate CIP deviations for all or specific currencies
- **Parameters**:
  - `start_date` (optional): Start date (YYYY-MM-DD)
  - `end_date` (optional): End date (YYYY-MM-DD)
  - `currency` (optional): Specific currency code
  - `include_data` (optional): Include raw data in response (true/false)

**Example Request:**
```
GET /api/cip/deviations?currency=EUR&start_date=2020-01-01&end_date=2021-01-01
```

**Example Response:**
```json
{
  "status": "success",
  "data": {
    "date_range": {
      "start": "2020-01-01T00:00:00",
      "end": "2021-01-01T00:00:00"
    },
    "observations": 366,
    "currencies": "EUR",
    "deviations": {
      "cip_deviation_eur": {
        "mean": 0.0023,
        "std": 0.0156,
        "min": -0.0234,
        "max": 0.0456,
        "latest": 0.0012,
        "data_points": 366
      }
    }
  }
}
```

### 6. CIP Analysis by Currency
- **GET** `/api/cip/analysis/{currency}`
- **Description**: Detailed CIP analysis for specific currency
- **Parameters**:
  - `currency` (path): Currency code (EUR, USD, GBP, JPY, CHF, SEK)

**Example Request:**
```
GET /api/cip/analysis/EUR
```

**Example Response:**
```json
{
  "status": "success",
  "data": {
    "currency": "EUR",
    "analysis_period": {
      "start": "1999-01-01T00:00:00",
      "end": "2024-12-31T00:00:00",
      "observations": 6234
    },
    "metrics": {
      "cip_deviation_eur": {
        "descriptive_stats": {
          "mean": 0.0015,
          "median": 0.0008,
          "std": 0.0234,
          "min": -0.0876,
          "max": 0.1234,
          "skewness": 0.456,
          "kurtosis": 3.789
        },
        "recent_values": {...},
        "volatility": {
          "rolling_30d": 0.0156,
          "rolling_90d": 0.0198
        }
      }
    }
  }
}
```

### 7. Risk Indicators
- **GET** `/api/risk/indicators`
- **Description**: Calculate systemic risk indicators using ECB CISS methodology
- **Response**: Market block indicators and risk metrics

**Example Response:**
```json
{
  "status": "success",
  "data": {
    "date_range": {
      "start": "1999-01-01T00:00:00",
      "end": "2024-12-31T00:00:00"
    },
    "observations": 6234,
    "has_ciss": true,
    "market_blocks": {
      "money_block": {
        "mean": 0.234,
        "std": 0.456,
        "latest": 0.123,
        "data_points": 5678
      },
      "bond_block": {...},
      "equity_block": {...},
      "fin_block": {...}
    },
    "ciss": {
      "latest_value": 0.156,
      "mean": 0.234,
      "std": 0.123,
      "percentile_95": 0.567
    }
  }
}
```

### 8. CISS Indicator
- **GET** `/api/risk/ciss`
- **Description**: Calculate Composite Indicator of Systemic Stress
- **Parameters**:
  - `include_data` (optional): Include time series data (true/false)

**Example Response:**
```json
{
  "status": "success",
  "data": {
    "ciss_overview": {
      "latest_value": 0.156,
      "date": "2024-12-31T00:00:00",
      "mean": 0.234,
      "std": 0.123,
      "min": 0.001,
      "max": 0.876
    },
    "stress_levels": {
      "current": "Medium",
      "percentiles": {
        "p50": 0.234,
        "p75": 0.345,
        "p90": 0.456,
        "p95": 0.567
      }
    },
    "time_series": {
      "observations": 6234,
      "date_range": {
        "start": "1999-01-01T00:00:00",
        "end": "2024-12-31T00:00:00"
      }
    }
  }
}
```

### 9. Custom Analysis
- **POST** `/api/analysis/custom`
- **Description**: Run custom analysis with specified parameters
- **Content-Type**: `application/json`
- **Body**: Analysis configuration

**Supported Analysis Types:**
- `correlation`: Correlation analysis between variables
- `volatility`: Rolling volatility analysis
- `summary_stats`: Summary statistics for selected columns

**Example Request:**
```json
{
  "analysis_type": "correlation",
  "parameters": {
    "columns": ["SpotRateUSDtoEUR", "ForwardRateUSDtoEUR"]
  }
}
```

**Example Response:**
```json
{
  "status": "success",
  "data": {
    "analysis_type": "correlation",
    "parameters": {...},
    "correlation_matrix": {
      "SpotRateUSDtoEUR": {
        "SpotRateUSDtoEUR": 1.0,
        "ForwardRateUSDtoEUR": 0.987
      },
      "ForwardRateUSDtoEUR": {
        "SpotRateUSDtoEUR": 0.987,
        "ForwardRateUSDtoEUR": 1.0
      }
    }
  }
}
```

## Visualization Endpoints

### 10. CIP Deviations Chart
- **GET** `/api/charts/cip_deviations`
- **Description**: Generate CIP deviations chart for all currencies
- **Response**: Base64-encoded PNG image

### 11. Bandwidth vs Volatility Chart
- **GET** `/api/charts/bandwidth_volatility`
- **Description**: Generate bandwidth vs volatility comparison chart
- **Parameters**: `currency` (default: EUR)
- **Response**: Base64-encoded PNG image

### 12. CIP Deviation vs Band Charts
- **GET** `/api/charts/cip_deviation_vs_band`
- **Description**: Generate CIP deviation vs neutral band charts for all currencies
- **Response**: Multiple base64-encoded PNG images

### 13. CISS Indicator Chart
- **GET** `/api/charts/ciss_indicator`
- **Description**: Generate CISS indicator chart
- **Response**: Base64-encoded PNG image

### 14. CISS Comparison Chart
- **GET** `/api/charts/ciss_comparison`
- **Description**: Generate comparison chart between official and constructed CISS
- **Response**: Base64-encoded PNG image

### 15. Cross-Correlation Chart
- **GET** `/api/charts/cross_correlation`
- **Description**: Generate cross-correlation chart
- **Response**: Base64-encoded PNG image

### 16. Summary Dashboard
- **GET** `/api/charts/summary_dashboard`
- **Description**: Generate comprehensive summary dashboard with multiple plots
- **Response**: Base64-encoded PNG image

## HTML Chart Views

### 17. CIP Deviations View
- **GET** `/charts/cip_deviations_view`
- **Description**: HTML page with embedded CIP deviations chart
- **Response**: Complete HTML page

### 18. Dashboard View
- **GET** `/charts/dashboard_view`
- **Description**: HTML page with comprehensive dashboard
- **Response**: Complete HTML page

## Error Codes

| HTTP Status | Error Code | Description |
|-------------|------------|-------------|
| 400 | BAD_REQUEST | Invalid request parameters |
| 404 | NOT_FOUND | Endpoint not found |
| 500 | INTERNAL_ERROR | Server error |

## Rate Limiting

- **Development**: 60 requests per minute, 1000 per hour
- **Production**: 30 requests per minute, 500 per hour

## Data Caching

The API implements intelligent caching:
- **Cache Duration**: 1 hour
- **Cache Size**: 100 MB maximum
- **Cache Refresh**: Automatic refresh when data is updated

## Example Usage

### Python
```python
import requests

# Get API status
response = requests.get('http://localhost:5000/api/status')
print(response.json())

# Get CIP deviations for EUR
params = {
    'currency': 'EUR',
    'start_date': '2020-01-01',
    'end_date': '2021-01-01'
}
response = requests.get('http://localhost:5000/api/cip/deviations', params=params)
print(response.json())
```

### JavaScript
```javascript
// Get API status
fetch('http://localhost:5000/api/status')
  .then(response => response.json())
  .then(data => console.log(data));

// Get risk indicators
fetch('http://localhost:5000/api/risk/indicators')
  .then(response => response.json())
  .then(data => console.log(data));
```

### cURL
```bash
# Get API status
curl http://localhost:5000/api/status

# Get CIP deviations with parameters
curl "http://localhost:5000/api/cip/deviations?currency=EUR&start_date=2020-01-01"

# Custom analysis
curl -X POST http://localhost:5000/api/analysis/custom \
  -H "Content-Type: application/json" \
  -d '{"analysis_type": "correlation", "parameters": {"columns": ["SpotRateUSDtoEUR"]}}'
```

## Development

### Starting the API
```bash
# Method 1: Using startup script
python scripts/start_api.py

# Method 2: Direct execution
python src/api/app.py

# Method 3: Flask command
export FLASK_APP=src/api/app.py
flask run
```

### Running Tests
```bash
# Run API tests
python -m pytest tests/test_api.py -v

# Run specific test
python -m pytest tests/test_api.py::TestFlaskAPI::test_api_status -v
```

### Configuration

The API can be configured through environment variables:

```bash
export FLASK_ENV=development
export FLASK_DEBUG=True
export LOG_LEVEL=INFO
```

## Support

For issues or questions about the API:
1. Check the logs for error details
2. Verify data availability using `/api/status`
3. Review the API documentation at the root endpoint
4. Check test cases in `tests/test_api.py`

## Version History

- **v1.0.0**: Initial release with core functionality
  - CIP analysis endpoints
  - Risk indicator calculations
  - CISS methodology implementation
  - Custom analysis capabilities
  - Comprehensive error handling
  - Data caching and optimization
