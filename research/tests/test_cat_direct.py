"""Direct CAT engine-comparison guards; no Isaac or hardware starts here."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cat_direct_native import CAT, ROOT
from cat_direct_results import inertia_error


class InertiaFrameTests(unittest.TestCase):
    def test_physx_inertia_is_already_in_link_axes(self):
        from scipy.spatial.transform import Rotation
        quat = Rotation.from_euler("xyz", [.4, -.7, .2]).as_quat(scalar_first=True)
        rotation = Rotation.from_quat(quat, scalar_first=True).as_matrix()
        diagonal = [.1, .2, .3]
        link_inertia = rotation@np.diag(diagonal)@rotation.T
        fixture = dict(native_inertia_quat=[quat], native_diaginertia=[diagonal],
                       imported_inertias=[link_inertia.ravel()])
        self.assertLess(inertia_error(fixture), 1e-12)
        fixture["imported_inertias"] = [np.diag(diagonal).ravel()]
        self.assertGreater(inertia_error(fixture), .01)


@unittest.skipUnless(importlib.util.find_spec("mujoco") and
    (CAT/"data/assets/TypiObs/side1/sdf.npy").exists() and
    (ROOT/"artifacts/cat_release/logs_v1/generalist_v1/checkpoints/config.json").exists(),
    "Pinned native CAT fixture not downloaded")
class NativePlayerTests(unittest.TestCase):
    def test_native_post_physics_method_is_not_a_rewritten_controller(self):
        import mujoco
        from cat_direct_native import make_player
        a, constants, _ = make_player("side1")
        b, _, _ = make_player("side1")
        sa, sb = a.reset(), b.reset()
        self.assertEqual(a.dt, .02)
        self.assertEqual(a.sim_dt, .002)
        self.assertEqual(a.mj_model.nu, 29)
        self.assertEqual(sa.obs["state"].shape, (162,))
        for _ in range(4):
            action = np.linspace(-.02, .02, 12)
            sa = a.step(sa, action)
            targets = constants.DEFAULT_QPOS[7:].copy()
            ids = b.action_joint_ids
            targets[ids] = np.clip(sb.info["motor_targets"][ids]+action*b._config.action_scale,
                                  b._soft_lowers[ids], b._soft_uppers[ids])
            sb.info["motor_targets"] = targets.copy()
            for _ in range(10):
                b.mj_data.ctrl[:] = constants.KPs*(targets-b.mj_data.qpos[7:])-constants.KDs*b.mj_data.qvel[6:]
                mujoco.mj_step(b.mj_model, b.mj_data)
            sb = b.observe_after_physics(sb, action)
            np.testing.assert_allclose(sa.obs["state"], sb.obs["state"], atol=1e-10, rtol=0)
            np.testing.assert_allclose(a.mj_data.qpos, b.mj_data.qpos, atol=1e-10, rtol=0)


if __name__ == "__main__":
    unittest.main()
