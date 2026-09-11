#!/usr/bin/env python3
"""Export flagged contact intervals from a validated capture, without Isaac.

Consecutive physics samples are grouped per link/partner/classification, not
counted as separate collision events. The intervals are review candidates, NOT
contact truth, contact permissions or automatically approved reward labels.
Original captures are read-only; every invocation creates a fresh review run.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from contact_results import audit_contact_capture


def contact_intervals(data, report):
    dt = float(report["physics_dt"])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Expected positive finite physics dt")
    threshold = report["limits"]["force_threshold"]
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Expected finite nonnegative contact threshold")
    events = []
    for code, label in enumerate(report["classification_order"]):
        if label in ("below_force_threshold", "stance_support_candidate"):
            continue
        rows = data[(data[:, 13] == code) & (data[:, 5] > threshold)]
        for si, pi in sorted({tuple(map(int, row[3:5])) for row in rows}):
            pair = rows[(rows[:, 3] == si) & (rows[:, 4] == pi)]
            steps = np.unique(pair[:, 0]).astype(int)
            # Do not bridge gaps or merge different partners/labels/feet.
            groups = np.split(steps, np.flatnonzero(np.diff(steps) != 1)+1)
            for group in groups:
                part = pair[(pair[:, 0] >= group[0]) & (pair[:, 0] <= group[-1])]
                peak = part[np.argmax(part[:, 5])]
                events.append(dict(link=report["sensor_link_order"][si],
                    partner=report["partner_order"][pi], classification=label,
                    first_physics_sample=int(group[0]), last_physics_sample=int(group[-1]),
                    first_sample_time_s=float((group[0]+1)*dt), last_sample_time_s=float((group[-1]+1)*dt),
                    sampled_duration_s=float(len(group)*dt), physics_samples=len(group),
                    reference_frames=sorted(set(map(int, part[:, 2]))), contact_point_records=len(part),
                    peak_point_force_N=float(peak[5]), peak_physics_sample=int(peak[0]),
                    normal_z_range=[float(part[:, 11].min()), float(part[:, 11].max())],
                    peak_point_environment=peak[6:9].tolist(), peak_normal_environment=peak[9:12].tolist(),
                    peak_point_body=peak[14:17].tolist(), peak_normal_agreement=float(peak[18]),
                    max_surface_error_m=float(np.abs(part[:, 17]).max()),
                    max_penetration_m=float(max(0., -part[:, 12].min())),
                    review_status="unreviewed", verified_phase=None, reward_label=None))
    return sorted(events, key=lambda e: (e["first_physics_sample"], e["link"], e["classification"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    capture = args.capture.resolve()
    integrity = audit_contact_capture(capture)
    report = json.loads((capture/"contact_audit.json").read_text())
    with np.load(capture/"contact_audit.npz", allow_pickle=False) as archive:
        events = contact_intervals(archive["contacts"], report)
    root = Path(__file__).resolve().parent
    output = root/"runs"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")+"_contact_review")
    output.mkdir(parents=True, exist_ok=False)
    payload = dict(schema="grail-cat-contact-review-v1", simulation_only=True,
        source_capture=str(capture), capture_integrity=integrity,
        source_hashes={name: hashlib.sha256((capture/name).read_bytes()).hexdigest()
                       for name in ("contact_audit.json", "contact_audit.npz", "run.json")},
        source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)),
        physics_dt=report["physics_dt"], limits=report["limits"],
        interval_counts=dict(Counter(e["classification"] for e in events)), intervals=events,
        point_frame="environment_zero_local; axes parallel to world; body coordinates use articulated pose",
        time_contract="samples after physics update: first sample at dt; duration counts occupied time bins",
        classification_contract="existing diagnostic categories retained; no relabeling or normal changes",
        phase_truth_verified=False, contact_permissions_granted=False, avoidance_training_ready=False)
    (output/"review.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    lines = ["# Contact review queue", "", f"Capture: `{capture.name}`", "",
        "Unreviewed intervals, not verified violations. No reward labels assigned.", "",
        "Times are sampled end-of-physics-step times. Consecutive samples are grouped",
        "only for the same link, partner and label. Point counts are not event counts.", "",
        "| Sample times (s) | Link | Partner | Diagnostic label | Samples | Peak point force (N) |",
        "|---|---|---|---|---:|---:|"]
    for e in events:
        lines.append(f'| {e["first_sample_time_s"]:.3f}–{e["last_sample_time_s"]:.3f} | '
            f'{e["link"]} | {e["partner"]} | {e["classification"]} | '
            f'{e["physics_samples"]} | {e["peak_point_force_N"]:.2f} |')
    (output/"REVIEW.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"review": str(output), "interval_counts": payload["interval_counts"]}))


if __name__ == "__main__":
    main()
