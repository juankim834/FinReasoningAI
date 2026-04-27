"""
conftest.py — pytest configuration.
Adds the project root to sys.path so 'src' is importable without installation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
