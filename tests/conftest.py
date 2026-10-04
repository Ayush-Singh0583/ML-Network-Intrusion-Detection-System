"""
The tests never read or write the project's own ``data/``, ``cache/``,
``runs/``, ``saved_models/`` or ``paper/``.

WHY THIS FILE EXISTS
--------------------
``src/config.py`` takes those folders from ``NIDS_*`` environment variables,
read once, when ``config`` is first imported.  The end-to-end tests already
set them for the child processes they start.  The tests that call the library
in this process did not, and for a long time nobody could tell: on a machine
whose ``data/`` holds no CSVs the default folder is as good as an empty one.

On 2026-10-05 the suite was run for the first time on a machine whose
``data/`` held the real dataset.  ``load_clean`` compared a test's small
synthetic cache with the real CSVs of the same names, found the sizes
different -- which is what it is for -- and rebuilt the test's cache from 2.8
million real flows.  Three tests failed, and a dozen more passed while
checking something other than what their names say.  Nothing in the project
folder was written; the rebuilt caches went to pytest's temporary folder.

THE FIX
-------
pytest imports this file before any test module, so the variables are set
before ``config`` exists.  They point at empty folders under one temporary
directory, which is removed when the session ends.  A test that needs data
makes its own and names it (``load_clean(cache, folder_path=...)``).

``tests/test_meta.py::test_the_suite_cannot_see_the_projects_own_folders``
fails if this file ever stops doing its job.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

SEALED_ROOT = Path(tempfile.mkdtemp(prefix="nids-tests-"))

SEALED = {
    "NIDS_DATA_DIR": SEALED_ROOT / "data",
    "NIDS_CACHE_DIR": SEALED_ROOT / "cache",
    "NIDS_RUNS_DIR": SEALED_ROOT / "runs",
    "NIDS_ARTIFACT_DIR": SEALED_ROOT / "saved_models",
    "NIDS_PAPER_DIR": SEALED_ROOT / "paper",
}

for _var, _path in SEALED.items():
    _path.mkdir(parents=True, exist_ok=True)
    # Overwritten, not defaulted: a test run must not depend on what the
    # shell that started it happened to export.
    os.environ[_var] = str(_path)


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(SEALED_ROOT, ignore_errors=True)
