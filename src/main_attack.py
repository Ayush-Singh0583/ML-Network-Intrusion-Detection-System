"""
DEPRECATED -- main_attack.py

Attack-only multiclass on a random split.

Replacement:
    python src/run.py train --model xgb --protocol closedset
"""

import sys

_MSG = """
main_attack.py has been replaced.

  Why : random split of temporal data
  Use : python src/run.py train --model xgb --protocol closedset

The previous implementation is preserved at
_backup/src_pre_refactor_20260826/main_attack.py
"""

print(_MSG, file=sys.stderr)
raise SystemExit(2)
