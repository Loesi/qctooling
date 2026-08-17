"""
Make the ``src`` layout importable when running pytest from the folder without
installing the package first (libary loaded via LIBCINT_PATH).
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))