"""Minimal smoke test: the SMS ceiling notebook runs end-to-end."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "sms_ceiling.py"


def test_sms_notebook_runs(tmp_path: Path) -> None:
    script = tmp_path / "sms_run.py"
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
