"""
DEPRECATED -- main_binary_crossday.py

Produced results/classification_report.csv (0.6415 accuracy, 0.00085 ATTACK
recall) and overwrote whatever the previous run had written there.

Replacement:
    python src/run.py train --model xgb --protocol crossday
"""

import sys

_MSG = """
main_binary_crossday.py has been replaced.

  Why : fixed output path overwrote previous results
  Use : python src/run.py train --model xgb --protocol crossday

The previous implementation is preserved at
_backup/src_pre_refactor_20260826/main_binary_crossday.py
"""

print(_MSG, file=sys.stderr)
raise SystemExit(2)
