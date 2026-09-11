import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artifacts import digest, portable_asset_reference, safe_path, verify_file
from baseline import evaluation_command
from snapshot_environment import portable_requirements


class ArtifactTests(unittest.TestCase):
    def test_reject_unsafe_paths(self):
        for path in ("../secret", "/etc/passwd", "a/../../b", "a\\b"):
            with self.assertRaises(ValueError):
                safe_path(path)

    def test_known_texture_relocation_only(self):
        self.assertEqual(portable_asset_reference("/mnt/author/object_usd/textures/a.jpg", "scene"),
                         "textures/scene/a.jpg")
        with self.assertRaises(ValueError):
            portable_asset_reference("/home/user/key", "scene")

    def test_checksums_and_no_silent_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture"
            path.write_bytes(b"abc")
            item = {"path": "fixture", "size": 3, "hash_kind": "sha256", "hash": hashlib.sha256(b"abc").hexdigest()}
            verify_file(path, item)
            self.assertEqual(digest(path, "git-blob-sha1"), hashlib.sha1(b"blob 3\0abc").hexdigest())
            path.write_bytes(b"bad")
            with self.assertRaises(ValueError):
                verify_file(path, item)


class EnvironmentTests(unittest.TestCase):
    def test_snapshot_excludes_editable_projects(self):
        output = portable_requirements("numpy==1.26.4\n-e /private/project\n")
        self.assertIn("numpy==1.26.4", output)
        self.assertNotIn("private", output)

    def test_snapshot_rejects_nonportable_requirements(self):
        for line in ("pkg @ file:///tmp/a", "pkg @ git+https://github.com/example/x@main"):
            with self.assertRaises(ValueError):
                portable_requirements(line)

    def test_baseline_is_bounded_single_gpu_simulation(self):
        command = evaluation_command(Path("/run"), Path("/run/data/stair_p1"), "fixture", 1)
        self.assertIn("gear_sonic.eval_agent_trl", command)
        self.assertIn("++run_eval_loop=false", command)
        self.assertIn("++manager_env.commands.motion.motion_lib_cfg.motion_shard_world_size=1", command)
        self.assertIn("++manager_env.commands.motion.motion_lib_cfg.motion_shard_rank=0", command)
        self.assertNotIn("deploy", " ".join(command))
        self.assertNotIn("unitree_sdk", " ".join(command))


if __name__ == "__main__":
    unittest.main()
