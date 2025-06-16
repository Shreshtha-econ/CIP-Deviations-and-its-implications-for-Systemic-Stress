"""
API Configuration
Settings and configuration for the Flask API.
"""

import os
from datetime import timedelta

class APIConfig:
    """API configuration settings."""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # CORS settings
    CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # Cache settings
    CACHE_DURATION = timedelta(hours=1)
    MAX_CACHE_SIZE = 100  # MB
    
    # API rate limiting
    RATE_LIMIT_PER_MINUTE = 60
    RATE_LIMIT_PER_HOUR = 1000
    
    # Response settings
    MAX_RESPONSE_SIZE = 10000  # Maximum number of data points in response
    DEFAULT_PAGINATION_SIZE = 100
    MAX_PAGINATION_SIZE = 1000
    
    # Data settings
    SUPPORTED_DATE_FORMATS = [
        '%Y-%m-%d',
        '%Y-%m-%d %H:%M:%S',
        '%d/%m/%Y',
        '%d-%m-%Y'
    ]
    
    # Analysis settings
    DEFAULT_ANALYSIS_WINDOW = 30  # days
    MAX_ANALYSIS_WINDOW = 365 * 5  # 5 years
    
    # Logging settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


class DevelopmentConfig(APIConfig):
    """Development configuration."""
    DEBUG = True
    TESTING = False


class ProductionConfig(APIConfig):
    """Production configuration."""
    DEBUG = False
    TESTING = False
    
    # Stricter rate limiting for production
    RATE_LIMIT_PER_MINUTE = 30
    RATE_LIMIT_PER_HOUR = 500
    
    # Smaller response sizes for production
    MAX_RESPONSE_SIZE = 5000
    DEFAULT_PAGINATION_SIZE = 50


class TestingConfig(APIConfig):
    """Testing configuration."""
    DEBUG = True
    TESTING = True
    
    # Disable rate limiting for tests
    RATE_LIMIT_PER_MINUTE = 10000
    RATE_LIMIT_PER_HOUR = 100000


# Configuration mapping
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(config_name=None):
    """Get configuration class based on environment."""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default')
    
    return config_map.get(config_name, DevelopmentConfig)
