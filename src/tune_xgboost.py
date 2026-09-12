"""
DEPRECATED -- tune_xgboost.py

RandomizedSearchCV with scoring='accuracy' on an 80%-benign dataset, which
selects the benign-predictor. Wrote to models/ while training wrote to
saved_models/.

Replacement:
    python src/tune.py --model xgb --protocol crossday --n-iter 25
"""

import sys

_MSG = """
tune_xgboost.py has been replaced.

  Why : optimised accuracy on imbalanced data; wrote to the stale artifact store
  Use : python src/tune.py --model xgb --protocol crossday --n-iter 25

The previous implementation is preserved at
_backup/src_pre_refactor_20260826/tune_xgboost.py
"""

print(_MSG, file=sys.stderr)
raise SystemExit(2)
