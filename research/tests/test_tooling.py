import hashlib
import argparse
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artifacts import digest, portable_asset_reference, safe_path, verify_file
from baseline import evaluation_command
from check_environment import ALLOWED, classify
from snapshot_environment import portable_requirements
from training_smoke import smoke_overrides, UPDATES, STEPS_PER_ENV
from evaluation_audit import finite_nested
from gear_sonic.utils.app_launcher_args import consume_app_launcher_args


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
    def test_cat_teacher_is_exclusive_bounded_evaluation(self):
        options = dict(num_envs=1, cat_scene=Path("/cat"), layout_audit=True,
                       cat_teacher_contract=Path("/contract.json"),
                       cat_teacher_weights=Path("/teacher"), cat_teacher_steps=64)
        command = evaluation_command(Path("/run"), Path("/data"), "fixture", **options)
        self.assertIn('++research_cat_teacher_output="/run/cat_teacher_shadow.json"', command)
        self.assertIn("++max_render_steps=65", command)
        self.assertIn("++run_once=true", command)
        self.assertNotIn("train_agent", " ".join(command))
        for change in (dict(cat_teacher_steps=0), dict(cat_teacher_steps=501),
                       dict(num_envs=2), dict(gui=True), dict(cat_teacher_weights=None),
                       dict(residual_preflight=True), dict(contact_audit=True),
                       dict(observation_shadow=True), dict(layout_audit=False)):
            with self.assertRaises(ValueError):
                evaluation_command(Path("/run"), Path("/data"), "fixture", **{**options, **change})

    def test_residual_preflight_is_explicit_headless_no_update_and_bounded(self):
        from hydra.core.override_parser.overrides_parser import OverridesParser
        for count in (1, 4):
            command = evaluation_command(Path("/run"), Path("/data"), "fixture", count,
                cat_scene=Path("/cat"), layout_audit=True, residual_preflight=True)
            self.assertIn("++research_layout_replicated=true", command)
            self.assertIn('++research_residual_preflight_output="/run/residual_preflight.json"', command)
            self.assertIn("++run_once=true", command)
            self.assertIn("++headless=true", command)
            self.assertNotIn("train_agent", " ".join(command))
            OverridesParser.create().parse_overrides([arg for arg in command if arg.startswith("++")])
        for change in (dict(num_envs=5), dict(gui=True), dict(record_video=True), dict(layout_audit=False),
                       dict(observation_shadow=True), dict(contact_audit=True), dict(cat_scene=None)):
            options = dict(num_envs=1, cat_scene=Path("/cat"), layout_audit=True, residual_preflight=True)
            options.update(change)
            with self.assertRaises(ValueError):
                evaluation_command(Path("/run"), Path("/data"), "fixture", **options)
        with self.assertRaises(ValueError):
            evaluation_command(Path("/run"), Path("/data"), "fixture", 4, cat_scene=Path("/cat"), layout_audit=True)

    def test_video_is_opt_in_evaluation_only(self):
        from hydra.core.override_parser.overrides_parser import OverridesParser
        ordinary = evaluation_command(Path("/run"), Path("/data"), "fixture", 1)
        self.assertIn("++manager_env.config.render_results=false", ordinary)
        video = evaluation_command(Path("/run"), Path("/data"), "fixture", 1, record_video=True)
        self.assertIn("++manager_env.config.render_results=true", video)
        self.assertTrue(any("RenderEnvsRecorderCfg" in arg for arg in video))
        self.assertIn("++manager_env.recorders.dataset_export_mode=0", video)
        self.assertIn('++manager_env.recorders.dataset_export_dir_path="/run/video"', video)
        self.assertIn("gear_sonic.eval_agent_trl", video)
        # Dict JSON is not Hydra override syntax. Parse the real CLI values,
        # not just substring checks, without starting Isaac.
        OverridesParser.create().parse_overrides([arg for arg in video if arg.startswith("++")])
        for options in (dict(num_envs=2), dict(num_envs=1, gui=True)):
            with self.assertRaises(ValueError):
                evaluation_command(Path("/run"), Path("/data"), "fixture", record_video=True, **options)

    def test_clutter_is_opt_in_bounded_and_not_training(self):
        command = evaluation_command(Path("/run"), Path("/run/data/stair_p1"), "fixture", 1,
                                     clutter_audit=True)
        self.assertIn('++manager_env.config.research_clutter="stair_side_v1"', command)
        self.assertIn("++max_render_steps=501", command)
        self.assertIn("++run_once=true", command)
        self.assertIn("++eval_callbacks=[]", command)
        self.assertNotIn("train_agent", " ".join(command))
        ordinary = evaluation_command(Path("/run"), Path("/run/data/stair_p1"), "fixture", 1)
        self.assertFalse(any("research_clutter" in x for x in ordinary))

    def test_gui_repeats_without_one_shot_evaluation_callback(self):
        command = evaluation_command(Path("/run"), Path("/run/data/stair_p1"), "fixture", 1, gui=True)
        self.assertIn("++headless=false", command)
        self.assertIn("++eval_callbacks=[]", command)
        self.assertIn("++run_eval_loop=true", command)
        self.assertIn("++realtime=true", command)
        self.assertIn("++run_once=false", command)

    def test_hydra_flags_do_not_reach_kit(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--headless", action="store_true")
        argv = ["train.py", "--config-path", "/run/config", "--config-name", "smoke", "--headless", "++num_envs=4"]
        with patch.object(sys, "argv", argv):
            args = consume_app_launcher_args(parser)
            self.assertTrue(args.headless)
            self.assertEqual(sys.argv, ["train.py"])

    def test_rollout_validation_rejects_nonfinite_nested_state(self):
        self.assertTrue(finite_nested({"terminated": [False], "state": [1.0, 2.0]}))
        self.assertFalse(finite_nested({"state": [[float("nan")]]}))
        self.assertFalse(finite_nested([float("inf")]))

    def test_training_smoke_has_small_fresh_output_and_strict_load(self):
        config = smoke_overrides(Path("/run"))
        self.assertEqual(config["algo.trl.num_total_batches"], UPDATES)
        self.assertEqual(config["algo.config.num_steps_per_env"], STEPS_PER_ENV)
        self.assertNotEqual(Path(config["checkpoint"]).parent, Path(config["experiment_dir"]))
        self.assertTrue(config["warm_resume"])
        self.assertTrue(config["trainer.strict_checkpoint_load"])
        self.assertFalse(config["use_wandb"])
        self.assertEqual(set(config["callbacks"]), {"model_save"})

    def test_dependency_exceptions_are_exact_not_blanket(self):
        self.assertFalse(classify(list(ALLOWED))["unexpected"])
        changed = [("isaacsim-kernel", "5.1.0.0", "numpy", "==1.26.0", "2.0.0")]
        self.assertEqual(classify(changed)["unexpected"], changed)

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
