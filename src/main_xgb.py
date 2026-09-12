"""
DEPRECATED -- main_xgb.py

Hard-coded UNKNOWN_THRESHOLD = 0.90 (CLAUDE.md declares 0.55) and reported
metrics on the same Friday data the threshold was chosen against.

Replacement:
    python src/run.py train --model xgb --protocol crossday
"""

import sys

_MSG = """
main_xgb.py has been replaced.

  Why : uncalibrated rejection threshold, evaluated on the test day
  Use : python src/run.py train --model xgb --protocol crossday

The previous implementation is preserved at
_backup/src_pre_refactor_20260826/main_xgb.py
"""

print(_MSG, file=sys.stderr)
raise SystemExit(2)
