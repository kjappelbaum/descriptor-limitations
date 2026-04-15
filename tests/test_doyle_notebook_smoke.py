"""Minimal smoke test: the Doyle ceiling notebook runs end-to-end."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "doyle_ceiling.py"


def test_doyle_notebook_runs(tmp_path: Path) -> None:
    """Export the marimo notebook to a script and execute it.

    Catches breakage from refactors of `information.py` or `data_loaders.py`.
    """
    script = tmp_path / "doyle_run.py"
    subprocess.run(
        ["marimo", "export", "script", str(NOTEBOOK), "-o", str(script)],
        cwd=REPO_ROOT, check=True, capture_output=True,
    )
    res = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert res.returncode == 0, (
        f"notebook script failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    )
