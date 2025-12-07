#!/usr/bin/env python
"""
Standalone wrapper for dataset analysis visualization.

This script can be executed directly to analyze temporal dataset properties.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == '__main__':
    from plot.dataset_analysis import main
    main()
