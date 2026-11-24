"""Lets 'python scripts/xxx.py' find the pollen package without installing it."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
