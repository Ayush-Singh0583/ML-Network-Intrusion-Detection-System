"""
DEPRECATED -- main_rf.py

Ran the whole pipeline at import time using train_test_split over the full
week. The 99.8% it produced measures memorisation of near-duplicate flows.

Replacement:
    python src/run.py train --model rf --protocol crossday
"""

import sys

_MSG = """
main_rf.py has been replaced.

  Why : random split of temporal data; no validation set
  Use : python src/run.py train --model rf --protocol crossday

The previous implementation is preserved at
_backup/src_pre_refactor_20260826/main_rf.py
"""

print(_MSG, file=sys.stderr)
raise SystemExit(2)
