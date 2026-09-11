"""Read-only verification of saved controlled-G1 contact fixtures; no Kit imports."""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from articulated_contact_fixtures import box_normal_cone_agreement, fixture_specs
from gear_sonic.research.contact_accounting import ContactLimits, classify_contact


def audit_articulated_fixtures(run):
    run = Path(run)
    report = json.loads((run/"articulated_fixture_report.json").read_text())
    if (report["schema"] != "grail-cat-articulated-fixtures-v1" or not report["passed"]
            or not report["simulation_only"] or report["policy_loaded"] or report["phase_truth_verified"]
            or report["contact_permissions_granted"] or report["avoidance_training_ready"]
            or report["contact_file"] != "articulated_fixture_contacts.json"):
        raise ValueError("Incomplete/invalid articulated fixtures")
    path = run/report["contact_file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != report["contact_sha256"]:
        raise ValueError("Fixture contact checksum mismatch")
    contacts = json.loads(path.read_text())
    specs = {s.name: s for s in fixture_specs()}
    if [r["name"] for r in report["fixtures"]] != list(specs) or report["unexpected_contacts"]:
        raise ValueError("Missing/duplicate cases or unintended contacts")
    if any(r["fixture"] not in specs for r in contacts):
        raise ValueError("Contact from unknown fixture")
    limits = ContactLimits(**report["limits"])
    for result in report["fixtures"]:
        spec = specs[result["name"]]
        if any(result[k] != v for k, v in asdict(spec).items()) or not result["passed"]:
            raise ValueError("Fixture case/expectation changed")
        rows = [r for r in contacts if r["fixture"] == spec.name]
        counts = Counter(r["classification"] for r in rows)
        if (len(rows) != result["contact_count"] or dict(counts) != result["classifications"]
                or result["other_pair_contacts"] or not result["geometry_checks_passed"]):
            raise ValueError("Inconsistent fixture counts/geometry")
        if spec.clear:
            if rows or result["first_contact_step"] is not None or result["steps"] != 30:
                raise ValueError("Clear control had a contact or skipped steps")
            if not .005 < result["actual_approach_m"] < .03:
                raise ValueError("Clear control did not approach the surface")
        elif (not rows or set(counts) != {spec.expected} or result["first_contact_step"] < 20
              or min(r["step"] for r in rows) != result["first_contact_step"]
              or result["steps"] != result["first_contact_step"]+10):
            raise ValueError("Contact case lacks measured onset/window/expected label")
        elif not .02 < result["actual_approach_m"] < .08:
            raise ValueError("Contact case did not execute the expected approach")
        solid = result["solid"]
        if not np.array_equal(solid["size"], spec.size):
            raise ValueError("Fixture box dimensions changed")
        c, s = math.cos(spec.yaw), math.sin(spec.yaw)
        rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        expected_normal = np.array({"down": [0, 0, 1], "up": [0, 0, -1], "front": [-1, 0, 0]}[spec.approach]) @ rotation.T
        for row in rows:
            if (row["link"] != spec.link or row["phase"] != spec.phase or not isinstance(row["step"], int)
                    or not result["first_contact_step"] <= row["step"] < result["steps"]):
                raise ValueError("Fixture contact identity/step mismatch")
            # Recompute both body coordinates and paired box geometry from the
            # saved measured world point and actual articulated pose.
            q = np.asarray(row["body_quaternion_wxyz"])
            if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1.) > 1e-4:
                raise ValueError("Invalid saved body quaternion")
            w, x, y, z = q
            body_r = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                               [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                               [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
            local = (np.asarray(row["point"])-row["body_position"]) @ body_r
            box = (np.asarray(row["point"])-solid["center"]) @ rotation
            distance_axes = np.abs(box)-np.asarray(solid["size"])/2
            distance = np.linalg.norm(np.maximum(distance_axes, 0))+min(0., distance_axes.max())
            agreement = box_normal_cone_agreement(row["point"], row["normal"], solid["center"], solid["size"], spec.yaw)
            if (not np.allclose(local, row["local_point"], atol=1e-6)
                    or not np.isclose(distance, row["surface_error_m"], atol=1e-6)
                    or not np.isclose(agreement, row["normal_agreement"], atol=1e-6)
                    or abs(distance) >= .005 or agreement <= .98 or np.dot(row["normal"], expected_normal) <= .98
                    or not row["separation_m"] > -.005 or not row["force_N"] > limits.force_threshold):
                raise ValueError("Saved fixture geometry/force disagrees with measured data")
            lo, hi = report["sole_regions"].get(spec.link, ([0, 0, 0], [0, 0, 0]))
            code = classify_contact(spec.partner, spec.link in report["sole_regions"], spec.phase, local, lo, hi,
                row["normal"], row["force_N"], row["separation_m"], distance, agreement, limits)
            if code != row["classification"]:
                raise ValueError("Saved contact classification could not be reproduced")
    return dict(passed=True, fixtures=len(specs), contact_point_records=len(contacts),
                phase_truth_verified=False, contact_permissions_granted=False, avoidance_training_ready=False)
