import os
import tempfile
from pathlib import Path


def _configure_project_tempdir():
    root = Path(__file__).resolve().parents[1]
    temp_dir = root / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMPDIR", str(temp_dir))
    os.environ.setdefault("TEMP", str(temp_dir))
    os.environ.setdefault("TMP", str(temp_dir))
    tempfile.tempdir = str(temp_dir)


_configure_project_tempdir()
