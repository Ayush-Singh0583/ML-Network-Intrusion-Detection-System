"""
DEPRECATED -- main_lr.py

Was a stub: three star-imports and '# TODO: Implement this model'.

Replacement:
    python src/run.py train --model lr --protocol closedset
"""

import sys

_MSG = """
main_lr.py has been replaced.

  Why : unimplemented stub
  Use : python src/run.py train --model lr --protocol closedset

The previous implementation is preserved at
_backup/src_pre_refactor_20260826/main_lr.py
"""

print(_MSG, file=sys.stderr)
raise SystemExit(2)
