"""
Quick launcher for the AI Mentor backend.
Run this from the project root:
    python start_backend.py
"""
import sys
import os

# Add backend dir to path so database.py can be found
backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
sys.path.insert(0, backend_dir)

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  AI Mentor Decision Support - Backend Launcher")
    print("=" * 60)
    print(f"  Backend dir: {backend_dir}")
    print(f"  Starting on: http://localhost:8000")
    print(f"  Health check: http://localhost:8000/")
    print(f"  Ping test:    http://localhost:8000/api/ping")
    print("=" * 60)
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[backend_dir],
        app_dir=backend_dir,
    )
