"""Runs inside llama.cpp's converter environment (not Clarity's).

Usage: python -I convert_shim.py VERSIONS_JSON CONVERTER [converter args...]

Writes the environment's Python version and every installed package to VERSIONS_JSON,
then runs CONVERTER as `__main__`. The recorded versions are the ones that converted: a
package installed twice (uv's overlay, and the environment under it) is recorded as the
one Python imports, the first on sys.path.
"""

import json
import platform
import runpy
import sys
from importlib.metadata import distributions
from pathlib import Path


def main() -> None:
    versions_path, converter, *args = sys.argv[1:]
    packages: dict[str, str] = {}
    for d in distributions():  # in sys.path order: the first of a name is the one imported
        packages.setdefault(d.metadata["Name"], d.version)
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
