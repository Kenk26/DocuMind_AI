"""
DocuMind AI - Runner Script

A simple script to launch the DocuMind AI application.
"""

import subprocess
import sys
import os

def main():
    """Run the DocuMind AI application."""
    print("Starting DocuMind AI...")
    print("Make sure Ollama is running with: ollama serve")
    print()

    try:
        from app.main import main as app_main
        app_main()
    except ImportError as e:
        print(f"Import error: {e}")
        print("\nPlease ensure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nApplication closed.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
