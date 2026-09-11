"""
DEPRECATED -- validate_dataset.py

Read an 865 MB CSV that main_xgb.py rewrote on every run.

Replacement:
    python src/run.py cache   (then: python -m pytest tests/)
"""

import sys

_MSG = """
validate_dataset.py has been replaced.

  Why : superseded by the Parquet cache and the split assertions
  Use : python src/run.py cache   (then: python -m pytest tests/)

The previous implementation is preserved at
_backup/src_pre_refactor_20260826/validate_dataset.py
"""

print(_MSG, file=sys.stderr)
raise SystemExit(2)
