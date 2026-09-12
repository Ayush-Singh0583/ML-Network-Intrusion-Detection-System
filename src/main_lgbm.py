"""
DEPRECATED -- main_lgbm.py

Random split; subsample=0.8 was a no-op because subsample_freq defaulted to 0.

Replacement:
    python src/run.py train --model lgbm --protocol closedset
"""

import sys

_MSG = """
main_lgbm.py has been replaced.

  Why : random split; LightGBM bagging never actually ran
  Use : python src/run.py train --model lgbm --protocol closedset

The previous implementation is preserved at
_backup/src_pre_refactor_20260826/main_lgbm.py
"""

print(_MSG, file=sys.stderr)
raise SystemExit(2)
