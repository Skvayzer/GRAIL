#!/usr/bin/env python3
"""Reproduce the pinned CAT sampler fixture in CAT's own Python/JAX environment.

Read-only source extraction: does NOT import its robot environment, MuJoCo, or
deployment code. Prints numerical values; never modifies the CAT checkout.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cat-repo", type=Path, required=True)
    args = parser.parse_args()
    fixture = json.loads((Path(__file__).parent/"tests/fixtures/cat_sampler.json").read_text())
    source = args.cat_repo/fixture["file"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != fixture["sha256"]:
        raise ValueError("CAT sampler source differs from fixture pin; review instead of updating silently")
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax.numpy as jp
    methods = []
    for cls in ast.parse(source.read_text()).body:
        if isinstance(cls, ast.ClassDef):
            methods += [n for n in cls.body if isinstance(n, ast.FunctionDef)
                        and n.name in ("world_to_grid", "sample_field")]
    if len(methods) != 2:
        raise ValueError("Unexpected source layout")
    namespace = {"jp": jp}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)
    obj = types.SimpleNamespace(pf_origin=jp.array(fixture["origin"]), dx=fixture["resolution"], Nx=4, Ny=4, Nz=4)
    obj.world_to_grid = types.MethodType(namespace["world_to_grid"], obj)
    x, y, z = jp.meshgrid(*[jp.arange(4.)]*3, indexing="ij")
    field = jp.stack((x, y, z, x+10*y+100*z), -1)
    positions = obj.pf_origin+jp.array(fixture["indices"])*obj.dx
    values = namespace["sample_field"](obj, field, positions)
    if not bool(jp.allclose(values, jp.array(fixture["legacy_values"]), rtol=1e-5, atol=2e-5)):
        raise ValueError("Fixture parity failed")
    print(json.dumps({"matched": True, "device": str(values.device), "legacy_values": values.tolist()}, indent=2))


if __name__ == "__main__":
    main()
