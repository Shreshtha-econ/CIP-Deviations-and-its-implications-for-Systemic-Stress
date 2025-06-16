# Production Deployment Guide

This guide helps you deploy the Master Thesis Financial Analysis System in production.

## 🚀 Quick Production Setup

### 1. System Requirements
- **Python**: 3.8+ (recommended: Python 3.9-3.11)
- **Memory**: Minimum 2GB RAM (recommended: 4GB+)
- **Storage**: 1GB for data files
- **Network**: Port 5050 available

### 2. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd master-thesis-project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

```bash
# Copy and edit configuration if needed
# Configuration is in config/settings.py
# Default settings work for most use cases
```

### 4. Data Setup

```bash
# Ensure data files are in Data/ directory
# The system will automatically process Excel files
# from Data/raw/ into Data/processed/
```

### 5. Start the API Server

```bash
# Start the Flask API
python src/api/app.py

# Or use the startup script
python scripts/start_api.py

# API will be available at: http://localhost:5050
```

### 6. Verify Installation

```bash
# Run comprehensive API tests
python test_all_endpoints.py

# Expected result: 100% success rate (33/33 endpoints)
```

## 🔧 Production Configuration

### Environment Variables
Set these environment variables for production:

```bash
# Flask environment
export FLASK_ENV=production

# Optional: Custom port (default: 5050)
export PORT=5050

# Optional: Secret key for sessions
export SECRET_KEY=your-secret-key-here
```

### Security Considerations
- **API Keys**: No API keys required for local deployment
- **CORS**: Enabled by default for web integration
- **HTTPS**: Add reverse proxy (nginx/Apache) for HTTPS in production
- **Firewall**: Ensure port 5050 is accessible to intended users

## 🌐 Web Server Deployment

### Using Gunicorn (Recommended for Linux/macOS)

```bash
# Install gunicorn
pip install gunicorn

# Start with gunicorn
gunicorn --bind 0.0.0.0:5050 --workers 4 src.api.app:app
```

### Using Waitress (Cross-platform)

```bash
# Install waitress
pip install waitress

# Start with waitress
waitress-serve --host=0.0.0.0 --port=5050 src.api.app:app
```

### Using Docker (Optional)

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5050

CMD ["python", "src/api/app.py"]
```

Build and run:
```bash
docker build -t financial-analysis-api .
docker run -p 5050:5050 financial-analysis-api
```

## 📊 Monitoring & Health Checks

### Health Check Endpoint
```bash
# Check if API is running
curl http://localhost:5050/api/status

# Expected response: {"status": "ok", "timestamp": "..."}
```

### Performance Monitoring
- **Response Times**: Most endpoints respond within 1-3 seconds
- **Memory Usage**: Monitor Python process memory (~200-500MB typical)
- **CPU Usage**: Spikes during chart generation (normal)

### Log Monitoring
- **Application Logs**: Check console output for errors
- **Access Logs**: Monitor API endpoint usage
- **Error Logs**: Any 500 errors indicate issues requiring attention

## 🔄 Maintenance

### Regular Tasks
```bash
# Update processed data (monthly)
python scripts/run_analysis.py --force-reload

# Run API tests (weekly)
python test_all_endpoints.py

# Check system status
curl http://localhost:5050/api/status
```

### Backup Procedures
- **Data Files**: Backup `Data/` directory regularly
- **Configuration**: Backup `config/settings.py` if modified
- **Code**: Use version control (Git) for code changes

## 🚨 Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Find process using port 5050
netstat -ano | findstr :5050    # Windows
lsof -i :5050                   # macOS/Linux

# Kill the process and restart
```

#### 2. Missing Dependencies
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

#### 3. Data Loading Errors
```bash
# Force reload data files
python scripts/run_analysis.py --force-reload
```

#### 4. Memory Issues
```bash
# Check available memory
# Consider increasing system RAM or using data chunking
# Modify ANALYSIS_CONFIG in config/settings.py if needed
```

### Support Contacts
- **Documentation**: See `docs/` directory
- **API Reference**: `docs/API_DOCUMENTATION.md`
- **Technical Issues**: Check logs and error messages

## 📈 Scaling Considerations

### For High Traffic
- **Load Balancer**: Use nginx/HAProxy for multiple instances
- **Database**: Consider moving to PostgreSQL for large datasets
- **Caching**: Implement Redis for frequently accessed data
- **CDN**: Use CDN for static chart images

### For Large Datasets
- **Memory**: Increase RAM allocation
- **Processing**: Use chunked data processing
- **Storage**: Consider database storage for very large datasets

---

**Status**: Production Ready ✅  
**API Success Rate**: 100% (33/33 endpoints)  
**Last Updated**: June 2025
