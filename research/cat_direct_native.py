#!/usr/bin/env python3
"""Native CAT player/checkpoint harness for the simulation-only engine comparison.

Execute the pinned player's methods directly, without importing its training
registry/JAX stack. The Isaac runner uses MuJoCo ONLY as a kinematic sensor
mirror: no mj_step call is used to advance the Isaac robot.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import shutil
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
CAT = ROOT.parent.parent/"Click-and-Traverse"
sys.path.insert(0, str(ROOT.parent/"imports/SONIC"))
SOURCES = {
    "cat_ppo/envs/g1/env_cat.py": "ae56b75e698d179e095307cd514c0b0c34dffaeec30e822d4b178d13abb8766e",
    "cat_ppo/envs/g1/play_cat.py": "5ebb22cfd6ff70c74a3338f9e37872f30113cb7161d5959b0c082c9937091723",
    "cat_ppo/envs/g1/constants.py": "93de18a873aa7a5652bbab842f24ace6b09eef1cb29cd62c75e02040c4296910",
    "data/assets/unitree_g1/g1_mjx_feetonly_torque.xml": "7e2b294fc335d388613000ff333e6b32aa81052b0fba41ee3bc311dfacc43499",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_player(cat=CAT):
    for name, expected in SOURCES.items():
        if digest(cat/name) != expected:
            raise ValueError(f"Native CAT source changed: {name}")
    const_path = cat/"cat_ppo/envs/g1/constants.py"
    tree = ast.parse(const_path.read_text())
    tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module == "cat_ppo.constant")]
    namespace = dict(PATH_ASSET=cat/"data/assets")
    exec(compile(tree, str(const_path), "exec"), namespace)
    consts = SimpleNamespace(**{k: v for k, v in namespace.items() if not k.startswith("__")})
    source = cat/"cat_ppo/envs/g1/play_cat.py"
    tree = ast.parse(source.read_text())
    tree.body = [n for n in tree.body if not (
        isinstance(n, ast.Import) and any(a.name == "cat_ppo" for a in n.names)
        or isinstance(n, ast.ImportFrom) and (n.module or "").startswith("cat_ppo"))]
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            node.decorator_list = [d for d in node.decorator_list if not ast.unparse(d).startswith("cat_ppo.")]
    namespace = dict(__name__=__name__, consts=consts)
    exec(compile(tree, str(source), "exec"), namespace)
    # The upstream source targets NumPy 2; keep the Isaac NumPy 1.26 pin intact.
    class NumpyCompat:
        concat = staticmethod(np.concatenate)
        def __getattr__(self, name):
            return getattr(np, name)
    namespace["np"] = NumpyCompat()
    cls = namespace["PlayG1CatEnv"]
    step = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls.__name__)
    step = next(n for n in step.body if isinstance(n, ast.FunctionDef) and n.name == "step")
    # Reuse the original post-physics field/gait/observation update verbatim.
    start = next(i for i, n in enumerate(step.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "head_pos" for t in n.targets))
    step.name, step.body, step.decorator_list = "observe_after_physics", step.body[start:], []
    exec(compile(ast.fix_missing_locations(ast.Module(body=[step], type_ignores=[])), str(source), "exec"), namespace)
    cls.observe_after_physics = namespace["observe_after_physics"]
    return cls, consts


def make_player(scene, cat=CAT):
    from check_cat_bridge import native_contract
    cls, consts = native_player(cat)
    cfg = json.loads((ROOT/"artifacts/cat_release/logs_v1/generalist_v1/checkpoints/config.json").read_text())["env_config"]
    def config(value):
        return SimpleNamespace(**{k: config(v) for k, v in value.items()}) if isinstance(value, dict) else value
    cfg = config(cfg)
    cfg.pf_config.path = str(cat/"data/assets/TypiObs"/scene)
    player = cls(config=cfg, headless=True)
    return player, consts, native_contract(player, cat)


def teacher():
    from gear_sonic.research.cat_teacher import CatTeacher
    path = ROOT/"artifacts/cat_teacher_v2/weights.npz"
    if digest(path) != "4fd7022efb45478e9805b154ffb07599ba0ec365913b5921efb99a3e73159440":
        raise ValueError("CAT teacher weights changed")
    torch.set_num_threads(2)
    return CatTeacher(path)


def act(policy, state):
    with torch.no_grad():
        return policy(torch.as_tensor(state.obs["state"], dtype=torch.float32)[None])[0].numpy()


def snapshot(player, state, action):
    return dict(qpos=player.mj_data.qpos.copy(), qvel=player.mj_data.qvel.copy(),
                obs=state.obs["state"].copy(), action=action.copy(),
                targets=state.info["motor_targets"].copy(),
                head=player.mj_data.site_xpos[player._head_site_id].copy(),
                clearance=np.concatenate([np.ravel(state.info[k+"df"]) for k in ("head", "feet", "hands")]))


def save_episode(run, scene, engine, rows, player, elapsed, metadata=None):
    archive = run/f"{scene}_{engine}.npz"
    arrays = {k: np.stack([row[k] for row in rows]) for k in rows[0]}
    np.savez_compressed(archive, **arrays)
    # Native player succ is collision/fall acceptance, not goal attainment.
    # Original typical field extends from x=-0.5 to 2.5, guidance goal near x=2.
    last = arrays["qpos"][-1, :3]
    result = dict(engine=engine, scene=scene, steps=len(rows), simulated_s=len(rows)*player.dt,
        elapsed_s=elapsed, final_root=last.tolist(), max_x=float(arrays["qpos"][:, 0].max()),
        min_head_height_m=float(arrays["head"][:, 2].min()),
        minimum_native_site_sdf_m=float(arrays["clearance"].min()),
        native_clearance_or_height_violation_steps=int(((arrays["head"][:, 2] <= .7)
            | (arrays["clearance"] <= -.04).any(-1)).sum()),
        reached_exit=bool(last[0] >= 1.9), fell=bool(arrays["head"][-1, 2] < .7),
        policy="unchanged released CAT generalist weights, verified PyTorch export",
        grail_loaded=False, optimizer_steps=0, robot_actuation=False,
        archive=archive.name, sha256=digest(archive))
    result.update(metadata or {})
    (run/f"{scene}_{engine}.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)
    return result


def prepare_robot_xml(run, cat=CAT):
    """Portable generated MJCF for the importer; original files untouched.

    Remove sensors/contact declarations for import only. Isaac applies the
    explicit native collision pairs separately; MuJoCo keeps the original XML.
    """
    source = cat/"data/assets/unitree_g1/g1_mjx_feetonly_torque.xml"
    tree = ET.parse(source)
    root = tree.getroot()
    for tag in ("sensor", "contact", "keyframe"):
        for node in root.findall(tag):
            root.remove(node)
    for node in root.iter("mesh"):
        if "file" in node.attrib:
            original = (source.parent/node.get("file")).resolve()
            destination = run/"assets"/original.name
            destination.parent.mkdir(exist_ok=True)
            if not destination.exists():
                shutil.copy2(original, destination)
            if digest(destination) != digest(original):
                raise ValueError("Importer mesh copy differs from native asset")
            node.set("file", "assets/"+original.name)
    path = run/"cat_robot_v2.xml"
    tree.write(path, encoding="unicode")
    return path


def main():
    import mujoco
    import time
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--scenes", nargs="+", default=["side1", "hurdle1", "crouch1"])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--verify-onnx", action="store_true", help="Verify recorded inputs against original ONNX actor; no physics")
    args = parser.parse_args()
    if not 1 <= args.steps <= 2000 or any(Path(s).name != s for s in args.scenes):
        parser.error("Use safe typical scene names and 1..2000 steps")
    args.run.mkdir(parents=True, exist_ok=True)
    policy = teacher()
    if args.verify_onnx:
        import onnxruntime as ort
        path = CAT/"data/models/generalist_v1/policy.onnx"
        expected = "17372d1d7b1c6759a2eb2eea09b9589f55746def198d744517efd05ac8665f62"
        if digest(path) != expected:
            raise ValueError("Original ONNX checkpoint changed")
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 2
        session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        reports = []
        for scene in args.scenes:
            for engine in ("mujoco", "isaac"):
                source = args.run/f"{scene}_{engine}.npz"
                with np.load(source, allow_pickle=False) as archive:
                    obs = archive["obs"].astype(np.float32)
                with torch.no_grad():
                    actual = policy(torch.from_numpy(obs)).numpy()
                native = np.concatenate([session.run(["continuous_actions"], {"obs": row[None]})[0] for row in obs])
                np.testing.assert_allclose(actual, native, atol=6e-5, rtol=2e-5)
                reports.append(dict(scene=scene, engine=engine, frames=len(obs), max_error=float(np.abs(actual-native).max()),
                                    packet_sha256=digest(source)))
        report = dict(passed=True, original_onnx_sha256=expected, cases=reports, physics_started=False)
        with (args.run/"onnx_parity.json").open("x") as stream:
            json.dump(report, stream, indent=2)
        print(json.dumps(report, indent=2))
        return
    for scene in args.scenes:
        if (args.run/f"{scene}_mujoco.npz").exists():
            raise FileExistsError("Native episode already recorded")
        player, _, _ = make_player(scene)
        np.random.seed(42)
        state = player.reset()
        rows, start = [], time.monotonic()
        for step in range(args.steps):
            action = act(policy, state)
            state = player.step(state, action)
            rows.append(snapshot(player, state, action))
            if rows[-1]["head"][2] < .7 or player.mj_data.qpos[0] >= 1.9:
                break
        save_episode(args.run, scene, "mujoco", rows, player, time.monotonic()-start)
    print(json.dumps(dict(mujoco_version=mujoco.__version__, completed=True)), flush=True)


if __name__ == "__main__":
    main()
