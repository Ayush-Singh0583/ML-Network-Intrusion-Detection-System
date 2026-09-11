"""
DEPRECATED -- tune_xgb.py

GridSearchCV, 8 configs x 3 folds, n_jobs=1, full 2M rows, no tree_method='hist',
scoring='f1_weighted' on an 80%-benign dataset.

Replacement:
    python src/tune.py --model xgb --protocol crossday --n-iter 25
"""

import sys

_MSG = """
tune_xgb.py has been replaced.

  Why : optimised weighted F1; multi-hour search over the full training set
  Use : python src/tune.py --model xgb --protocol crossday --n-iter 25

The previous implementation is preserved at
_backup/src_pre_refactor_20260826/tune_xgb.py
"""

print(_MSG, file=sys.stderr)
raise SystemExit(2)
