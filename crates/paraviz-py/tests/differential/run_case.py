"""CLI entry: run one differential case and write its trace JSON.

Usage: python run_case.py <case_name> <out.json>
Works under real ParaView's python AND the paraviz shim venv.
"""

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

case = importlib.import_module(f"cases.{sys.argv[1]}")
case.main(sys.argv[2])
