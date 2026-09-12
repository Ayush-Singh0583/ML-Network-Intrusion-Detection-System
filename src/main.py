"""
DEPRECATED -- main.py

Byte-identical to main_rf.py apart from a trailing newline, and it ran the
entire pipeline at import time with a random stratified split over a full
week of temporally bursty flows.

Replacement:
    python src/run.py train --model rf --protocol crossday
"""

import sys

_MSG = """
main.py has been replaced.

  Why : duplicate of main_rf.py; random split of temporal data
  Use : python src/run.py train --model rf --protocol crossday

The previous implementation is preserved at
_backup/src_pre_refactor_20260826/main.py
"""

print(_MSG, file=sys.stderr)
raise SystemExit(2)
