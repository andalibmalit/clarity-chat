"""Runs inside llama.cpp's converter environment (not Clarity's).

Usage: python -I convert_shim.py VERSIONS_JSON CONVERTER [converter args...]

Writes the environment's Python version and every installed package to VERSIONS_JSON,
then runs CONVERTER as `__main__`. The recorded versions are the ones that converted.
"""

import json
import platform
import runpy
import sys
from importlib.metadata import distributions
from pathlib import Path


def main() -> None:
    versions_path, converter, *args = sys.argv[1:]
    packages = {d.metadata["Name"]: d.version for d in distributions()}
    with open(versions_path, "w", encoding="utf-8", newline="") as f:
        json.dump(
            {"python": platform.python_version(), "packages": dict(sorted(packages.items()))},
            f,
            indent=1,
        )
    sys.path.insert(0, str(Path(converter).resolve().parent))  # for its `conversion` package
    sys.argv = [converter, *args]
    runpy.run_path(converter, run_name="__main__")


if __name__ == "__main__":
    main()
