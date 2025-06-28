#!/usr/bin/env python3
"""
LandPPT Application Runner

This script starts the LandPPT FastAPI application with proper configuration.
"""

import uvicorn
import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Main entry point for running the application"""
    
    # Configuration
    config = {
        "app": "landppt.main:app",
        "host": "0.0.0.0",
        "port": 8000,
        "reload": True,
        "log_level": "info",
        "access_log": True,
    }
    
    print("🚀 Starting LandPPT Server...")
    print(f"📍 Server will be available at: http://localhost:{config['port']}")
    print(f"📚 API Documentation: http://localhost:{config['port']}/docs")
    print(f"🌐 Web Interface: http://localhost:{config['port']}/web")
    print("=" * 60)
    
    try:
        uvicorn.run(**config)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
