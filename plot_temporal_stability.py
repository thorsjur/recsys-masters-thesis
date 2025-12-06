#!/usr/bin/env python3
"""
Standalone script for plotting temporal stability.
Usage: ./plot_temporal_stability.py --experiment-id <id>
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from plot.temporal_stability import main

if __name__ == '__main__':
    main()
