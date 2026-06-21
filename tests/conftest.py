"""pytest configuration: put the repo root on sys.path so the test modules can
``import box`` / ``import packing`` etc. (the modules live at the repo root)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
