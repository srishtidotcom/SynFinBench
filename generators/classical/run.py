"""
run.py - Execute the Classical ARIMA-GARCH Generator

This script executes the train.ipynb notebook programmatically.
Useful for command-line execution or automation pipelines.

Usage:
    python run.py

Outputs:
    - All outputs saved to generators/classical/outputs/
    - synthetic_returns.csv (5230 returns × 2 columns)
    - synthetic_windows.npy (5200 windows × 30 timesteps)
    - Diagnostic plots (pre_fit, post_fit, evaluation_suite, etc.)
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Execute the train.ipynb notebook."""
    
    script_dir = Path(__file__).parent
    notebook_path = script_dir / "train.ipynb"
    
    if not notebook_path.exists():
        print(f"❌ Error: {notebook_path} not found!")
        sys.exit(1)
    
    print("🚀 Starting Classical ARIMA-GARCH Generator...")
    print(f"📓 Executing: {notebook_path}")
    print("⏱️  Expected time: ~4-5 minutes (ARIMA grid search is slow)")
    print()
    
    try:
        # Execute notebook using nbconvert
        result = subprocess.run(
            [
                sys.executable, "-m", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--inplace",
                "--ExecutePreprocessor.timeout=1800",
                str(notebook_path)
            ],
            capture_output=True,
            text=True,
            cwd=script_dir
        )
        
        if result.returncode != 0:
            print("❌ Execution failed!")
            print("\nSTDERR:")
            print(result.stderr)
            sys.exit(1)
        
        print("✅ Execution completed successfully!")
        print()
        print("📊 Outputs saved to:")
        print(f"   {script_dir / 'outputs' / 'synthetic_returns.csv'}")
        print(f"   {script_dir / 'outputs' / 'synthetic_windows.npy'}")
        print()
        print("📈 Diagnostic plots:")
        outputs_dir = script_dir / "outputs"
        plots = [
            "pre_fit_diagnostics.png",
            "arima_cv_signals.png", 
            "post_fit_diagnostics.png",
            "evaluation_suite.png"
        ]
        for plot in plots:
            if (outputs_dir / plot).exists():
                print(f"   ✓ {plot}")
        
        print()
        print("🧪 To run tests:")
        print("   python -m pytest generators/classical/tests/test_arima_garch.py -v")
        
    except FileNotFoundError:
        print("❌ Error: nbconvert not found!")
        print("Install with: pip install nbconvert")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
