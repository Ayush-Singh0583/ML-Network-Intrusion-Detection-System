"""
DEPRECATED -- main_attack_crossday.py

Attack-only multiclass, cross-day, evaluated with the double-ground-truth
metric function.

Replacement:
    python src/run.py train --model xgb --protocol crossday
"""

import sys

_MSG = """
main_attack_crossday.py has been replaced.

  Why : metrics computed against two different y_true arrays
  Use : python src/run.py train --model xgb --protocol crossday

The previous implementation is preserved at
_backup/src_pre_refactor_20260826/main_attack_crossday.py
"""

print(_MSG, file=sys.stderr)
raise SystemExit(2)
