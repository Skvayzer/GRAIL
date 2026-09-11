import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observation_results import audit_observation_shadow
from gear_sonic.research.obstacle_observation import ObservationSpec


class SavedShadowTests(unittest.TestCase):
    def fixture(self):
        n = 2
        data = dict(volume=np.zeros((n, 4, 3, 3, 3), np.float32), probes=np.zeros((n, 1, 8), np.float32),
            guidance=np.zeros((n, 9), np.float32), valid=np.ones(n, bool), root=np.zeros((n, 3), np.float32),
            probe_centers=np.zeros((n, 1, 3), np.float32),
            quaternion=np.tile([1., 0., 0., 0.], (n, 1)).astype(np.float32), baseline_action=np.zeros((n, 29), np.float32),
            live_action=np.zeros((n, 29), np.float32), zero_residual_action=np.zeros((n, 29), np.float32),
            residual=np.zeros((n, 64), np.float32))
        data["volume"][:, 2:] = 1
        data["probes"][..., 6:] = 1
        data["probes"][..., 3] = .05
        data["guidance"][:, 7] = 1
        meta = dict(schema="grail-cat-observation-shadow-v1", complete=True, error=None, packet_file="observation_shadow.npz",
            adapter_applied_to_environment=False, original_actor_inputs_changed=False, rewards_changed=False,
            policy_training=False, optimizer_steps=0, physics_steps_added=0, base_backbone_unchanged=True,
            adapter_unchanged=True, frames=n, probe_order=[dict(link="pelvis", offset=[0., 0., 0.], radius=.1, collision_index=0)],
            latent_dim=64, packet_spec=ObservationSpec(shape=(3, 3, 3)).manifest(), valid_frames=n,
            samples=[dict(step=i, valid=True, parity_max_abs_error=0.) for i in range(n)],
            outcomes=[dict(step=i, terminal=i == n-1, failure=False) for i in range(n)],
            gradient_check=dict(step=0, head_gradient_norm=.01, frozen_actor_parameter_gradients=False))
        return meta, data

    def write(self, directory, meta, data):
        path = Path(directory)/"observation_shadow.npz"
        np.savez_compressed(path, **data)
        meta["packet_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (Path(directory)/"observation_shadow.json").write_text(json.dumps(meta))

    def test_valid_and_rehashed_action_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            meta, data = self.fixture()
            self.write(directory, meta, data)
            self.assertTrue(audit_observation_shadow(directory)["passed"])
            data["live_action"][0, 0] = .0001
            self.write(directory, meta, data)
            with self.assertRaisesRegex(ValueError, "parity"):
                audit_observation_shadow(directory)

    def test_mask_mismatch_unknown_placeholder_and_unsigned_terrain(self):
        for kind in ("mask", "placeholder", "unsigned", "binary", "nonfinite"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                meta, data = self.fixture()
                if kind in ("mask", "placeholder"):
                    data["volume"][0, 2, 0, 0, 0] = 0
                if kind == "placeholder":
                    data["volume"][0, 0, 0, 0, 0] = 1
                if kind == "unsigned":
                    data["volume"][0, 1, 0, 0, 0] = -.1
                if kind == "binary":
                    data["guidance"][0, 7] = .5
                if kind == "nonfinite":
                    data["root"][0, 0] = np.nan
                self.write(directory, meta, data)
                with self.assertRaises(ValueError):
                    audit_observation_shadow(directory)

    def test_terminal_pairing_gradient_and_actuation_flags(self):
        for kind in ("terminal", "pairing", "gradient", "actuation", "weights", "incomplete"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                meta, data = self.fixture()
                if kind == "terminal":
                    meta["outcomes"][0]["terminal"] = True
                if kind == "pairing":
                    meta["samples"][1]["step"] = 0
                if kind == "gradient":
                    meta["gradient_check"]["head_gradient_norm"] = 0
                if kind == "actuation":
                    meta["adapter_applied_to_environment"] = True
                if kind == "weights":
                    meta["base_backbone_unchanged"] = False
                if kind == "incomplete":
                    meta["complete"] = False
                self.write(directory, meta, data)
                with self.assertRaises(ValueError):
                    audit_observation_shadow(directory)

    def test_checksum_and_zero_residual(self):
        with tempfile.TemporaryDirectory() as directory:
            meta, data = self.fixture()
            data["residual"][0, 0] = .01
            self.write(directory, meta, data)
            with self.assertRaisesRegex(ValueError, "parity"):
                audit_observation_shadow(directory)
            path = Path(directory)/meta["packet_file"]
            path.write_bytes(path.read_bytes()+b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                audit_observation_shadow(directory)


if __name__ == "__main__":
    unittest.main()
