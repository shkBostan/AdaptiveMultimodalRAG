"""
Legacy main entry point for AdaptiveMultimodalRAG system.

NOTE: This file is deprecated. It forwards to scripts/run_pipeline.py.
For direct usage, use: python scripts/run_pipeline.py --config <config_file>

Author: s Bostan
Created on: Nov, 2025
"""

import sys
import subprocess
from pathlib import Path


def main():
    """
    Forward to scripts/run_pipeline.py.
    
    This maintains backward compatibility for legacy scripts that call main.py.
    """
    script_path = Path(__file__).parent / "scripts" / "run_pipeline.py"
    
    if not script_path.exists():
        print("Error: scripts/run_pipeline.py not found.")
        print("Please use: python scripts/run_pipeline.py --config <config_file>")
        sys.exit(1)
    
    # Forward all arguments to run_pipeline.py
    args = sys.argv[1:]
    
    # If no arguments provided, show help
    if not args:
        print("Legacy main.py entry point. Forwarding to scripts/run_pipeline.py...")
        print("For usage, run: python scripts/run_pipeline.py --help")
        print("\nExample:")
        print("  python scripts/run_pipeline.py --config configs/exp1.yaml")
        sys.exit(0)
    
    # Execute run_pipeline.py with forwarded arguments
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)] + args,
            check=False
        )
        sys.exit(result.returncode)
    except Exception as e:
        print(f"Error forwarding to run_pipeline.py: {e}")
        print("Please use directly: python scripts/run_pipeline.py --config <config_file>")
        sys.exit(1)


if __name__ == "__main__":
    main()
