"""
DEPRECATED -- prediction.py

Loaded models/logistic_model.pkl -- a 1.5 KB artifact from a model whose entry
point was a TODO stub.

Replacement:
    python src/predict.py --bundle saved_models/<name> --csv <file>
"""

import sys

_MSG = """
prediction.py has been replaced.

  Why : loaded an orphaned artifact
  Use : python src/predict.py --bundle saved_models/<name> --csv <file>

The previous implementation is preserved at
_backup/src_pre_refactor_20260826/prediction.py
"""

print(_MSG, file=sys.stderr)
raise SystemExit(2)
