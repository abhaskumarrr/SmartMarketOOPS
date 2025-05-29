#!/usr/bin/env python3
"""
SMOOPs ML System Main Entry Point

This script serves as the entry point for the SMOOPs ML system,
providing a command-line interface for training and serving models.
"""

import sys
from pathlib import Path

# Ensure the project root is in the Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import and run the CLI
from src.cli import main

if __name__ == "__main__":
    main() 