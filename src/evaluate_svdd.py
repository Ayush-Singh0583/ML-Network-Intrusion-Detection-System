"""
DEPRECATED -- evaluate_svdd.py

Re-ran the entire preprocessing pipeline from the raw CSVs, then allocated the
whole test set on the device in one tensor.

Replacement:
    python src/run.py train --model deep_svdd  (evaluation is part of the run)
"""

import sys

_MSG = """
evaluate_svdd.py has been replaced.

  Why : full pipeline re-run; unbatched device allocation
  Use : python src/run.py train --model deep_svdd  (evaluation is part of the run)

The previous implementation is preserved at
_backup/src_pre_refactor_20260826/evaluate_svdd.py
"""

print(_MSG, file=sys.stderr)
raise SystemExit(2)
