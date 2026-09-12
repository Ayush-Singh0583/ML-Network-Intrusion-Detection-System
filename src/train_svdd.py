"""
DEPRECATED -- train_svdd.py

Started a 25-epoch training run on import. The model used nn.Linear defaults
(bias=True), which lets the encoder realise a constant function -- and it did:
the saved checkpoint's final bias equals the center to 4.85e-05 and the 99th
percentile benign radius is 5.58e-09. Also printed the SUMMED loss while
discarding the computed mean, and hard-coded '/25' in the format string.

Replacement:
    python src/run.py train --model deep_svdd --pretrain-ae
"""

import sys

_MSG = """
train_svdd.py has been replaced.

  Why : hypersphere collapse (bias terms); no seeding, no validation, no best-model restore
  Use : python src/run.py train --model deep_svdd --pretrain-ae

The previous implementation is preserved at
_backup/src_pre_refactor_20260826/train_svdd.py
"""

print(_MSG, file=sys.stderr)
raise SystemExit(2)
