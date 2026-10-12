"""Compatibility entry point; PRIMEU preparation lives in examples/primeu."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.primeu.prepare_data import main

if __name__ == "__main__":
    main()
