#!/usr/bin/env python
"""Chạy TrackEval cũ với các bí danh NumPy đã bị loại bỏ."""

import runpy
import sys

import numpy as np


def main() -> None:
    """Khôi phục bí danh trong tiến trình chấm rồi gọi TrackEval."""
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
