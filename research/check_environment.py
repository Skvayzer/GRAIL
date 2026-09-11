#!/usr/bin/env python3
"""Report active package conflicts, allowing only three documented upstream pins.

An allowed metadata conflict is NOT proof of runtime compatibility. Physics and
training smoke tests are separate gates. No extras, simulator or service starts.
"""
import importlib.metadata as metadata
import json

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


ALLOWED = {
    ("isaacsim-kernel", "5.1.0.0", "numpy", "==1.26.0", "1.26.4"),
    ("isaacsim-kernel", "5.1.0.0", "typing-extensions", "==4.12.2", "4.15.0"),
    ("fastapi", "0.115.7", "starlette", "<0.46.0,>=0.40.0", "0.49.1"),
}


def classify(conflicts):
    return {"documented": [x for x in conflicts if tuple(x) in ALLOWED],
            "unexpected": [x for x in conflicts if tuple(x) not in ALLOWED]}


def inspect():
    distributions = list(metadata.distributions())
    installed = {canonicalize_name(d.metadata["Name"]): d.version for d in distributions}
    conflicts = []
    for dist in distributions:
        for requirement in dist.requires or []:
            req = Requirement(requirement)
            if req.marker and not req.marker.evaluate({"extra": ""}):
                continue
            name = canonicalize_name(req.name)
            version = installed.get(name)
            if version is None or (req.specifier and version not in req.specifier):
                conflicts.append((canonicalize_name(dist.metadata["Name"]), dist.version,
                                  name, str(req.specifier), version))
    return classify(sorted(set(conflicts)))


if __name__ == "__main__":
    report = inspect()
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(report["unexpected"]))
