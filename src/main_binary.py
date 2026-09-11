"""
DEPRECATED -- main_binary.py

Binary BENIGN-vs-ATTACK on a random split, with the string-truncation bug
live: rejected samples became the phantom class 'Unknow' (numpy '<U6').

Replacement:
    python src/run.py train --model xgb --protocol closedset
"""

import sys

_MSG = """
main_binary.py has been replaced.

  Why : random split; numpy fixed-width string truncation
  Use : python src/run.py train --model xgb --protocol closedset

The previous implementation is preserved at
_backup/src_pre_refactor_20260826/main_binary.py
"""

print(_MSG, file=sys.stderr)
raise SystemExit(2)
