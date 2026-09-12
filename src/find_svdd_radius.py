"""
DEPRECATED -- find_svdd_radius.py

Took the 99th percentile of distances over the TRAINING benign set, not a
held-out one, and re-ran the whole pipeline a third time to do it.

Replacement:
    python src/run.py train --model deep_svdd   (tau is calibrated on validation)
"""

import sys

_MSG = """
find_svdd_radius.py has been replaced.

  Why : threshold calibrated on training data
  Use : python src/run.py train --model deep_svdd   (tau is calibrated on validation)

The previous implementation is preserved at
_backup/src_pre_refactor_20260826/find_svdd_radius.py
"""

print(_MSG, file=sys.stderr)
raise SystemExit(2)
