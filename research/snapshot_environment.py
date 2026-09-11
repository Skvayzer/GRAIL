#!/usr/bin/env python3
"""Export a portable package snapshot without changing the reference environment.

Run with the reference Python. Editable project packages are intentionally
excluded: the new environment must install this fork and its own Isaac Lab tree.
This is a package snapshot, not a claim of a fully validated solver lock.
"""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import re
import subprocess
import sys


def portable_requirements(freeze):
    lines = []
    for line in freeze.splitlines():
        if not line or line.startswith(("#", "-e ")):
            continue
        if "file:" in line or line.startswith("/"):
            raise ValueError(f"Non-portable requirement: {line.split('@')[0]}")
        if " @ " in line:
            url = line.split(" @ ", 1)[1]
            if not url.startswith("git+https://github.com/") or not re.search(r"@[0-9a-f]{40}$", url):
                raise ValueError("Unpinned/unexpected direct requirement")
        lines.append(line)
    return "# Desktop package snapshot; install with --no-deps.\n" + "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite an existing package snapshot")
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    result = portable_requirements(freeze)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result)
    print(json.dumps({"python": sys.version.split()[0], "packages": len(result.splitlines())-1,
                      "isaacsim": metadata.version("isaacsim"), "torch": metadata.version("torch")}))


if __name__ == "__main__":
    main()
