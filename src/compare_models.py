"""
DEPRECATED -- compare_models.py

Crashed at line 66: sort_values(by='F1 Score') against columns named
'Macro F1' and 'Weighted F1'.

Replacement:
    python src/run.py compare --protocol closedset --models rf xgb lgbm mlp cnn lstm
"""

import sys

_MSG = """
compare_models.py has been replaced.

  Why : KeyError on a renamed column; also used the random split
  Use : python src/run.py compare --protocol closedset --models rf xgb lgbm mlp cnn lstm

The previous implementation is preserved at
_backup/src_pre_refactor_20260826/compare_models.py
"""

print(_MSG, file=sys.stderr)
raise SystemExit(2)
