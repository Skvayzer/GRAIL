"""Contract-bound learner/RNG saves, never the released GRAIL checkpoint.

No environment state is serialized: resume starts new episodes. This is NOT an
exact continuation of PhysX, sensors, reference phase or contact histories.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import torch


SCHEMA = "grail-cat-residual-learner-v1"
RESUME = "learner/optimizer/exploration RNG restored; simulator starts new episodes"


def validate_optimizer(model, optimizer):
    if {id(p) for group in optimizer.param_groups for p in group["params"]} != {id(p) for p in model.parameters()}:
        raise ValueError("Optimizer must own exactly the residual learner, never the frozen backbone")


def contract_hash(contract):
    if (not isinstance(contract, dict) or not {"backbone_sha256", "observation", "algorithm"} <= set(contract)
            or not isinstance(contract["backbone_sha256"], str) or len(contract["backbone_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in contract["backbone_sha256"])):
        raise ValueError("Explicit frozen-backbone, observation and algorithm contract required")
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite checkpoint tensor")
    elif isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            finite_tree(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite checkpoint metadata")


def save_checkpoint(path, model, optimizer, generator, contract, updates):
    if type(updates) is not int or updates < 0 or not isinstance(generator, torch.Generator):
        raise ValueError("Explicit nonnegative update count and learner RNG required")
    validate_optimizer(model, optimizer)
    payload = dict(schema=SCHEMA, model_contract=model.manifest(), contract=contract, contract_sha256=contract_hash(contract),
        model=model.state_dict(), optimizer=optimizer.state_dict(), rng=generator.get_state(),
        rng_device=str(generator.device), updates=updates, resume_semantics=RESUME)
    finite_tree(payload)
    path = Path(path)
    # Temporary and final names are on the same filesystem. link() atomically
    # publishes without replacing any existing checkpoint, including baseline.
    with tempfile.NamedTemporaryFile(prefix=".learner-", suffix=".tmp", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checkpoint(path, model, optimizer, generator, contract):
    validate_optimizer(model, optimizer)
    # Only state dictionaries/tensors/primitives, not executable pickled models.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("schema") != SCHEMA or payload.get("resume_semantics") != RESUME
            or payload.get("model_contract") != model.manifest()
            or payload.get("contract_sha256") != contract_hash(contract)
            or contract_hash(payload["contract"]) != payload["contract_sha256"]
            or payload.get("rng_device") != str(generator.device)
            or type(payload.get("updates")) is not int or payload["updates"] < 0):
        raise ValueError("Checkpoint contract/RNG/resume semantics mismatch")
    finite_tree(payload)
    expected = model.state_dict()
    if (set(payload["model"]) != set(expected) or any(
            payload["model"][k].shape != v.shape or payload["model"][k].dtype != v.dtype for k, v in expected.items())):
        raise ValueError("Learner architecture differs from checkpoint")
    groups = optimizer.state_dict()["param_groups"]
    if (len(groups) != len(payload["optimizer"]["param_groups"]) or any(
            len(a["params"]) != len(b["params"]) for a, b in zip(groups, payload["optimizer"]["param_groups"]))):
        raise ValueError("Optimizer parameter groups differ")
    # Validate RNG before touching the live learner; PyTorch checks state size.
    check_rng = torch.Generator(device=generator.device)
    check_rng.set_state(payload["rng"])
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    generator.set_state(payload["rng"])
    return payload["updates"]
