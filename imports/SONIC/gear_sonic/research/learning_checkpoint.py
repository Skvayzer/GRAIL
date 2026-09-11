"""Contract-bound learner/RNG saves, never the released GRAIL checkpoint.

No environment state is serialized: resume starts new episodes. This is NOT an
exact continuation of PhysX, sensors, reference phase or contact histories.
"""
import hashlib
import copy
import json
import math
import os
from pathlib import Path
import tempfile

import torch


SCHEMA = "grail-cat-residual-learner-v2"
RESUME = "learner/optimizer/exploration RNG restored; simulator starts new episodes"


def validate_optimizer(model, optimizer):
    owned = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if len(owned) != len(set(owned)) or set(owned) != {id(p) for p in model.parameters()}:
        raise ValueError("Optimizer must own exactly the residual learner, never the frozen backbone")


def optimizer_names(model, optimizer):
    validate_optimizer(model, optimizer)
    names = {id(p): name for name, p in model.named_parameters()}
    return [[names[id(p)] for p in group["params"]] for group in optimizer.param_groups]


def validate_adam_state(optimizer, state):
    """Validate moments/groups before loading anything into a live learner.

    This pilot uses fixed-hyperparameter Adam/AdamW. Scheduler state and other
    optimizers require a separately versioned contract, not a permissive load.
    """
    if type(optimizer) not in (torch.optim.Adam, torch.optim.AdamW):
        raise ValueError("Only contracted Adam/AdamW optimizers are supported")
    finite_tree(state)
    expected = optimizer.state_dict()["param_groups"]
    if (not isinstance(state, dict) or set(state) != {"state", "param_groups"}
            or not isinstance(state["state"], dict) or len(state["param_groups"]) != len(expected)):
        raise ValueError("Malformed optimizer state/groups")
    ids = []
    for live, wanted, saved in zip(optimizer.param_groups, expected, state["param_groups"]):
        if saved != wanted:
            raise ValueError("Optimizer groups/hyperparameters differ from the contract")
        for index, parameter in zip(saved["params"], live["params"]):
            ids.append(index)
            if not state["state"]:
                continue  # Untouched zero-update checkpoint.
            moment = state["state"].get(index)
            keys = {"step", "exp_avg", "exp_avg_sq"}
            if saved["amsgrad"]:
                keys.add("max_exp_avg_sq")
            if not isinstance(moment, dict) or set(moment) != keys:
                raise ValueError("Incomplete Adam moment state")
            step = moment["step"]
            if (not isinstance(step, torch.Tensor) or step.ndim != 0 or step < 0
                    or step != step.floor()):
                raise ValueError("Invalid Adam step counter")
            for key in keys-{"step"}:
                value = moment[key]
                if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                        or value.dtype != parameter.dtype
                        or (key != "exp_avg" and (value < 0).any())):
                    raise ValueError("Invalid Adam moment shape/dtype/value")
    if state["state"] and set(state["state"]) != set(ids):
        raise ValueError("Optimizer moment ownership differs from learner parameters")


def validate_update_count(state, updates):
    moments = state["state"]
    if ((updates == 0 and moments) or (updates > 0 and not moments)
            or any(float(value["step"]) != updates for value in moments.values())):
        raise ValueError("Checkpoint update count differs from Adam moment counters")


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
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    validate_optimizer(model, optimizer)
    validate_adam_state(optimizer, optimizer.state_dict())
    validate_update_count(optimizer.state_dict(), updates)
    payload = dict(schema=SCHEMA, model_contract=model.manifest(), contract=contract, contract_sha256=contract_hash(contract),
        model=model.state_dict(), optimizer=optimizer.state_dict(), rng=generator.get_state(),
        rng_device=str(generator.device), updates=updates, resume_semantics=RESUME,
        optimizer_type=type(optimizer).__name__, optimizer_parameter_names=optimizer_names(model, optimizer))
    finite_tree(payload)
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
            or payload.get("optimizer_type") != type(optimizer).__name__
            or payload.get("optimizer_parameter_names") != optimizer_names(model, optimizer)
            or type(payload.get("updates")) is not int or payload["updates"] < 0):
        raise ValueError("Checkpoint contract/RNG/resume semantics mismatch")
    finite_tree(payload)
    expected = model.state_dict()
    if (set(payload["model"]) != set(expected) or any(
            payload["model"][k].shape != v.shape or payload["model"][k].dtype != v.dtype for k, v in expected.items())):
        raise ValueError("Learner architecture differs from checkpoint")
    validate_adam_state(optimizer, payload["optimizer"])
    validate_update_count(payload["optimizer"], payload["updates"])
    # Validate RNG before touching the live learner; PyTorch checks state size.
    check_rng = torch.Generator(device=generator.device)
    check_rng.set_state(payload["rng"])
    # Exercise PyTorch's loader on detached copies before the live state changes.
    check_optimizer = copy.deepcopy(optimizer)
    check_optimizer.load_state_dict(payload["optimizer"])
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    generator.set_state(payload["rng"])
    return payload["updates"]
