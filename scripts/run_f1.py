#!/usr/bin/env python3
"""Simple entry point: ``python scripts/run_f1.py --config config/default.yaml [options]``.

Accepts the same options as ``mo438-f1 run`` (see ``--help``).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from a fresh clone without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mo438_f1.cli import main

if __name__ == "__main__":
    sys.exit(main(["run", *sys.argv[1:]]))
