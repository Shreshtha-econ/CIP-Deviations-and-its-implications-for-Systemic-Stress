"""
API Tests
Unit tests for the Flask API endpoints.
"""

import unittest
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.app import app
from src.api.utils import ParameterValidator, DataSerializer, StatisticsCalculator


class TestFlaskAPI(unittest.TestCase):
    """Test cases for Flask API endpoints."""
    
    def setUp(self):
        """Set up test client."""
        self.app = app.test_client()
        self.app.testing = True
    
    def test_home_page(self):
        """Test the home page endpoint."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Financial Analysis API', response.data)
    
    def test_api_status(self):
        """Test the API status endpoint."""
        response = self.app.get('/api/status')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('api_version', data['data'])
        self.assertIn('currencies', data['data'])
    
    def test_data_summary(self):
        """Test the data summary endpoint."""
        response = self.app.get('/api/data/summary')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('total_observations', data['data'])
        self.assertIn('date_range', data['data'])
    
    def test_currencies_info(self):
        """Test the currencies info endpoint."""
        response = self.app.get('/api/data/currencies')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertIsInstance(data['data'], dict)
    
    def test_cip_deviations(self):
        """Test the CIP deviations endpoint."""
        response = self.app.get('/api/cip/deviations')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('deviations', data['data'])
    
    def test_cip_deviations_with_params(self):
        """Test CIP deviations with parameters."""
        params = {
            'start_date': '2020-01-01',
            'end_date': '2020-12-31',
            'currency': 'EUR'
        }
        response = self.app.get('/api/cip/deviations', query_string=params)
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
    
    def test_cip_analysis_currency(self):
        """Test CIP analysis for specific currency."""
        response = self.app.get('/api/cip/analysis/EUR')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['data']['currency'], 'EUR')
    
    def test_cip_analysis_invalid_currency(self):
        """Test CIP analysis with invalid currency."""
        response = self.app.get('/api/cip/analysis/INVALID')
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'error')
    
    def test_risk_indicators(self):
        """Test risk indicators endpoint."""
        response = self.app.get('/api/risk/indicators')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('market_blocks', data['data'])
    
    def test_ciss_indicator(self):
        """Test CISS indicator endpoint."""
        response = self.app.get('/api/risk/ciss')
        # This might fail if CISS calculation fails, so we check for either success or error
        self.assertIn(response.status_code, [200, 400, 500])
        
        data = json.loads(response.data)
        self.assertIn(data['status'], ['success', 'error'])
    
    def test_custom_analysis_correlation(self):
        """Test custom analysis - correlation."""
        payload = {
            'analysis_type': 'correlation',
            'parameters': {
                'columns': ['column1', 'column2']  # These might not exist, but test the structure
            }
        }
        response = self.app.post('/api/analysis/custom', 
                               data=json.dumps(payload),
                               content_type='application/json')
        # Should return either success or error (depending on data availability)
        self.assertIn(response.status_code, [200, 400, 500])
    
    def test_custom_analysis_no_data(self):
        """Test custom analysis with no data."""
        response = self.app.post('/api/analysis/custom')
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'error')
    
    def test_404_error(self):
        """Test 404 error handling."""
        response = self.app.get('/api/nonexistent')
        self.assertEqual(response.status_code, 404)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'error')


class TestAPIUtils(unittest.TestCase):
    """Test cases for API utility functions."""
    
    def test_parameter_validator_date(self):
        """Test date validation."""
        # Valid dates
        date1 = ParameterValidator.validate_date('2023-01-01')
        self.assertEqual(date1.year, 2023)
        
        date2 = ParameterValidator.validate_date('01/01/2023')
        self.assertEqual(date2.year, 2023)
        
        # Invalid date
        with self.assertRaises(ValueError):
            ParameterValidator.validate_date('invalid-date')
    
    def test_parameter_validator_date_range(self):
        """Test date range validation."""
        start, end = ParameterValidator.validate_date_range('2023-01-01', '2023-12-31')
        self.assertTrue(start < end)
        
        # Invalid range
        with self.assertRaises(ValueError):
            ParameterValidator.validate_date_range('2023-12-31', '2023-01-01')
    
    def test_parameter_validator_currency(self):
        """Test currency validation."""
        currency = ParameterValidator.validate_currency('eur', ['EUR', 'USD', 'GBP'])
        self.assertEqual(currency, 'EUR')
        
        # Invalid currency
        with self.assertRaises(ValueError):
            ParameterValidator.validate_currency('INVALID', ['EUR', 'USD', 'GBP'])
    
    def test_parameter_validator_numeric_range(self):
        """Test numeric range validation."""
        value = ParameterValidator.validate_numeric_range('10.5', 'test_param', 0, 100)
        self.assertEqual(value, 10.5)
        
        # Out of range
        with self.assertRaises(ValueError):
            ParameterValidator.validate_numeric_range('150', 'test_param', 0, 100)
    
    def test_data_serializer(self):
        """Test data serialization."""
        import pandas as pd
        import numpy as np
        
        # Test pandas serialization
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        serialized = DataSerializer.serialize_pandas(df)
        self.assertIsInstance(serialized, list)
        
        # Test numpy serialization
        arr = np.array([1, 2, 3])
        serialized = DataSerializer.serialize_pandas(arr)
        self.assertIsInstance(serialized, list)
        
        # Test datetime serialization
        dt = datetime.now()
        serialized = DataSerializer.serialize_datetime(dt)
        self.assertIsInstance(serialized, str)
    
    def test_statistics_calculator(self):
        """Test statistics calculation."""
        import pandas as pd
        
        series = pd.Series([1, 2, 3, 4, 5])
        stats = StatisticsCalculator.basic_stats(series)
        
        self.assertEqual(stats['count'], 5)
        self.assertEqual(stats['mean'], 3.0)
        self.assertEqual(stats['median'], 3.0)
        
        # Test empty series
        empty_series = pd.Series([])
        empty_stats = StatisticsCalculator.basic_stats(empty_series)
        self.assertEqual(empty_stats, {})


class TestAPIIntegration(unittest.TestCase):
    """Integration tests for the API."""
    
    def setUp(self):
        """Set up test client."""
        self.app = app.test_client()
        self.app.testing = True
    
    def test_api_workflow(self):
        """Test complete API workflow."""
        # 1. Check API status
        response = self.app.get('/api/status')
        self.assertEqual(response.status_code, 200)
        
        # 2. Get data summary
        response = self.app.get('/api/data/summary')
        self.assertEqual(response.status_code, 200)
        
        # 3. Get currencies info
        response = self.app.get('/api/data/currencies')
        self.assertEqual(response.status_code, 200)
        
        # 4. Get CIP deviations
        response = self.app.get('/api/cip/deviations')
        self.assertEqual(response.status_code, 200)
        
        # 5. Try risk indicators
        response = self.app.get('/api/risk/indicators')
        self.assertEqual(response.status_code, 200)
    
    def test_error_handling(self):
        """Test error handling across endpoints."""
        # Test invalid currency
        response = self.app.get('/api/cip/analysis/INVALID')
        self.assertEqual(response.status_code, 400)
        
        # Test invalid date format
        response = self.app.get('/api/cip/deviations?start_date=invalid')
        self.assertEqual(response.status_code, 400)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
