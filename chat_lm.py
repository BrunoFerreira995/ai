#!/usr/bin/env python3
"""Start a chat with the project's locally trained causal language model."""

from __future__ import annotations

import os
import sys
from pathlib import Path


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    python = project_dir / ".venv/bin/python"
    if not python.exists():
        raise SystemExit("Ambiente não encontrado. Execute ./install.sh primeiro.")
    command = [str(python), str(project_dir / "chat.py"), "--mode", "local", *sys.argv[1:]]
    os.execv(command[0], command)
