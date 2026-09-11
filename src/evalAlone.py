"""
DEPRECATED -- evalAlone.py

Crashed at line 150: confusion_matrix(y_test, y_pred) with y_test int64 and
y_pred a string array -> ValueError: Mix of label input types. Also re-derived
'the' test split from preprocessing code that had changed since training.

Replacement:
    python src/predict.py --bundle saved_models/<name> --protocol crossday
"""

import sys

_MSG = """
evalAlone.py has been replaced.

  Why : mixed int/str labels; test split re-derived from changed preprocessing
  Use : python src/predict.py --bundle saved_models/<name> --protocol crossday

The previous implementation is preserved at
_backup/src_pre_refactor_20260826/evalAlone.py
"""

print(_MSG, file=sys.stderr)
raise SystemExit(2)
