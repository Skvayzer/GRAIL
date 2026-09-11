"""Read-only saved articulated-contact audit checks. No Isaac imports/launches."""
import hashlib
import json
from pathlib import Path

import numpy as np


def audit_contact_capture(run):
    run = Path(run)
    report = json.loads((run/"contact_audit.json").read_text())
    if (report["schema"] != "grail-cat-articulated-contact-audit-v1"
            or report["contact_file"] != "contact_audit.npz" or not report["capture_complete"]
            or report["error"] is not None or not report["completed"]):
        raise ValueError("Incomplete/invalid articulated contact capture")
    path = run/report["contact_file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != report["contact_sha256"]:
        raise ValueError("Contact capture checksum mismatch")
    with np.load(path, allow_pickle=False) as archive:
        data = archive["contacts"]
    if data.shape != (report["detailed_contacts"], 19) or not np.isfinite(data).all():
        raise ValueError("Invalid packed contact data")
    steps, outcomes = report["samples"], report["policy_outcomes"]
    decimation = report["decimation"]
    if (not outcomes or decimation < 1 or len(steps) != len(outcomes)*decimation
            or len(steps) != report["physics_steps"] or len(outcomes) != report["policy_steps"]
            or report["terminal_physics_samples"] != decimation
            or sum(o["terminal"] for o in outcomes) != 1 or not outcomes[-1]["terminal"]):
        raise ValueError("Missing terminal or nonterminal physics steps")
    counters = np.array([s["physics_counter"] for s in steps])
    if not (np.diff(counters) == 1).all():
        raise ValueError("Repeated/skipped physics counter")
    for i, outcome in enumerate(outcomes):
        if outcome["first_physics_sample"] != i*decimation or outcome["physics_samples"] != decimation:
            raise ValueError("Policy/physics alignment mismatch")
    ranges = {0: len(steps), 1: len(outcomes), 3: len(report["sensor_link_order"]),
              4: len(report["partner_order"]), 13: len(report["classification_order"])}
    for column, size in ranges.items():
        v = data[:, column]
        if not ((v == np.floor(v)) & (v >= 0) & (v < size)).all():
            raise ValueError("Contact index out of range")
    if not (data[:, 1] == data[:, 0]//decimation).all():
        raise ValueError("Packed policy/physics alignment mismatch")
    expected = np.array([s["contact_count"] for s in steps])
    if not np.array_equal(np.bincount(data[:, 0].astype(int), minlength=len(steps)), expected):
        raise ValueError("Packed contact counts differ from physics samples")
    for i, name in enumerate(report["classification_order"]):
        if int((data[:, 13] == i).sum()) != report["classifications"].get(name, 0):
            raise ValueError("Packed classifications differ from report")
    if ((data[:, 5] < 0).any() or (np.abs(np.linalg.norm(data[:, 9:12], axis=1)-1.) > .002).any()
            or report["classifications"].get("invalid_geometry", 0)):
        raise ValueError("Invalid force/normal/geometry in captured contacts")
    return dict(capture_complete=True, physics_steps=len(steps), policy_steps=len(outcomes),
                terminal_physics_samples=decimation, detailed_contacts=len(data),
                phase_truth_verified=False, collision_free_certified=False)
