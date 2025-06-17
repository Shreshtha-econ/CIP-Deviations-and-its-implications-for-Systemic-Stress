import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Start the Flask API server."""
    print("🏦 Financial Analysis API Startup")
    print("=" * 50)
    
    # Set environment variables if not already set
    if not os.environ.get('FLASK_ENV'):
        os.environ['FLASK_ENV'] = 'production'
    
    if not os.environ.get('FLASK_DEBUG'):
        os.environ['FLASK_DEBUG'] = 'False'
    
    # Import and run the app
    try:
        from src.shre_api import app
        
        print("✅ Flask API imported successfully")
        print("📊 Starting server...")
        print("🌐 Server will be available at:")
        print("   - Local: http://localhost:8057")
        print("   - Network: http://0.0.0.0:8057")
        print("📚 API Documentation: http://localhost:8057")
        print("")
        print("💡 Press Ctrl+C to stop the server")
        print("=" * 50)
        
        # Start the server
        app.run(
            debug=False,
            host='0.0.0.0',
            port=8057,
            use_reloader=False,
            threaded=True
        )
        
    except ImportError as e:
        print(f"❌ Failed to import Flask API: {e}")
        print("🔧 Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
