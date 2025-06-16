# Heroku Deployment Configuration
import os

class ProductionConfig:
    """Production configuration for Heroku deployment."""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
    DEBUG = False
    TESTING = False
    
    # Database (if needed in future)
    DATABASE_URL = os.environ.get('DATABASE_URL', '')
    
    # API settings
    API_RATE_LIMIT = "1000 per hour"
    API_CACHE_TIMEOUT = 300  # 5 minutes
    
    # CORS settings
    CORS_ORIGINS = ["*"]  # Configure specific domains in production
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    
    # File paths for Heroku
    DATA_PATH = 'data'
    RESULTS_PATH = 'data/results'
    
class HerokuConfig(ProductionConfig):
    """Heroku-specific configuration."""
    
    # Heroku assigns PORT dynamically
    PORT = int(os.environ.get('PORT', 5000))
    
    # Use Heroku's ephemeral filesystem carefully
    # Data should be stored in external services for production
    CACHE_TYPE = "simple"  # Use Redis in production
    
    # Security headers
    SECURE_HEADERS = True
    
    @staticmethod
    def init_app(app):
        """Initialize app for Heroku deployment."""
        
        # Log to stderr for Heroku
        import logging
        from logging import StreamHandler
        
        file_handler = StreamHandler()
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        
        # Force HTTPS in production
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(app.wsgi_app)

# Configuration dictionary
config = {
    'production': ProductionConfig,
    'heroku': HerokuConfig,
    'default': HerokuConfig
}
