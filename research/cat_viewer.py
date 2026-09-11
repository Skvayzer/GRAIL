#!/usr/bin/env python3
"""Interactive desktop gallery of checksummed CAT clutter. No policy or robot.

Run with this repository's .venv Python. A supervisor bounds the GUI lifetime,
records provenance, and terminates only its own child process group on Ctrl-C.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from cat_scenes import ROOT, verify_scene


def gallery_layout(metadata):
    """Translations only, with one metre between source volumes; no rescaling."""
    if not 1 <= len(metadata) <= 4:
        raise ValueError("Choose one to four scenes")
    widths = [float(m["effective_size"][1]) for m in metadata]
    total = sum(widths) + len(widths) - 1
    cursor, placements = -total / 2, []
    for m, width in zip(metadata, widths):
        origin = m["origin_corner"]
        placements.append([0., cursor - origin[1], 0.])
        cursor += width + 1
    return placements


def worker(run, directories):
    # Kit must not receive our already-consumed CLI arguments. All simulator
    # imports, including USD, happen after AppLauncher starts the application.
    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=False, livestream=0, device="cuda:0",
                      width=1440, height=1000).app
    import asyncio
    import isaaclab.sim as sim_utils
    import omni.ui as ui
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
    from pxr import UsdGeom

    metadata = [verify_scene(d) for d in directories]
    placements = gallery_layout(metadata)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/30, device="cuda:0"))
    colors = [(0.85, 0.34, 0.12), (0.10, 0.60, 0.78), (0.55, 0.35, 0.75), (0.85, 0.65, 0.15)]
    extent = sum(m["effective_size"][1] for m in metadata) + len(metadata) - 1
    ground = sim_utils.CuboidCfg(size=(6., extent + 2., .08),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.20, .23, .27), roughness=.9))
    ground.func("/World/GalleryFloor", ground, translation=(1., 0., -.04))
    light = sim_utils.DomeLightCfg(intensity=1800., color=(.9, .93, 1.))
    light.func("/World/DomeLight", light)
    sun = sim_utils.DistantLightCfg(intensity=1800., angle=.5)
    sun.func("/World/KeyLight", sun, orientation=(.92388, .38268, 0., 0.))
    for i, (directory, meta, offset) in enumerate(zip(directories, metadata, placements)):
        path = f"/World/CAT_{i}_{meta['scene'].replace('-', '_')}"
        cfg = sim_utils.UsdFileCfg(usd_path=str(directory/"scene.usda"),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=colors[i], roughness=.65))
        cfg.func(path, cfg, translation=tuple(offset))
        # These colored markers project CAT's start/goal onto the floor for
        # orientation only. They are visual-only, not obstacles or task inputs.
        for name, color, source in (("Start", (.15, .9, .25), "start_w"),
                                    ("Goal", (.95, .13, .15), "goal_w")):
            marker = sim_utils.SphereCfg(radius=.09,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))
            point = meta[source]
            marker.func(f"/World/Markers/Scene_{i}/{name}", marker,
                        translation=(point[0] + offset[0], point[1] + offset[1], .10))
        mesh = sim.stage.GetPrimAtPath(path+"/Obstacles")
        if not mesh.IsA(UsdGeom.Mesh):
            raise RuntimeError(f"CAT mesh did not load: {path}")
    eye = [1. + extent*.9, -extent*1.15, max(3.5, extent*.9)]
    target = [1., 0., .6]
    sim.set_camera_view(eye, target)
    sim.reset()
    legend = ui.Window("CAT clutter gallery - simulation only", width=440, height=215)
    with legend.frame:
        with ui.VStack(spacing=5):
            ui.Label("Original CAT generator -> static USD / PhysX", height=23)
            for i, meta in enumerate(metadata):
                ui.Label(f"{i}: {meta['scene']} | {meta['occupied_cells']:,} occupied cells", height=22)
            ui.Label("Green = start, red = goal (projected onto floor).", height=22)
            ui.Label("4 cm voxels; each volume is 3 x 2 x 1.52 m.", height=22)
            ui.Label("No robot, policy, or avoidance training is running.", height=22)
            ui.Button("Reset overview camera", clicked_fn=lambda: sim.set_camera_view(eye, target), height=25)
    sim.stage.Export(str(run/"gallery.usda"))
    (run/"viewer_ready.json").write_text(json.dumps(dict(
        simulation_only=True, policy_loaded=False, scene_directories=[str(d) for d in directories],
        translations=placements, colors=colors[:len(metadata)], camera_eye=eye,
        camera_target=target, source_files=[m["files"] for m in metadata]), indent=2)+"\n")
    print(f"CAT_VIEWER_READY {run}", flush=True)

    async def capture():
        try:
            helper = capture_viewport_to_file(get_active_viewport(), str(run/"overview.png"))
            await helper.wait_for_result()
            print(f"CAT_VIEWER_SCREENSHOT {run/'overview.png'}", flush=True)
        except Exception as error:
            print(f"Screenshot failed (viewer remains available): {error}", flush=True)

    capture_task = None
    frame = 0
    while app.is_running():
        began = time.monotonic()
        sim.step(render=True)
        frame += 1
        if frame == 90:
            capture_task = asyncio.ensure_future(capture())
        time.sleep(max(0., 1/30 - (time.monotonic()-began)))
    if capture_task is not None and not capture_task.done():
        capture_task.cancel()
    # Static viewer has no training checkpoints or active writers on exit.
    # Avoid the known Kit teardown spin; supervisor also bounds process lifetime.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", action="append", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--accept-isaac-eula", action="store_true")
    parser.add_argument("--timeout", type=int, default=3600, help="Maximum GUI lifetime, seconds")
    parser.add_argument("--worker-run", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    directories = [d.resolve() for d in args.scene]
    metadata = [verify_scene(d) for d in directories]
    gallery_layout(metadata)
    if not 0 < args.timeout <= 14400:
        parser.error("--timeout must be between 1 and 14400 seconds")
    if not args.execute:
        print("Scenes verified. Add --execute --accept-isaac-eula to open desktop Isaac Sim.")
        return
    if not args.accept_isaac_eula:
        parser.error("Explicit --accept-isaac-eula is required")
    if not os.environ.get("DISPLAY"):
        parser.error("Desktop DISPLAY is unavailable")
    if args.worker_run:
        try:
            worker(args.worker_run, directories)
        except BaseException:
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        return
    run = ROOT/"runs"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")+"_cat_gallery")
    run.mkdir(parents=True, exist_ok=False)
    temporary = tempfile.mkdtemp(prefix="grail-view-", dir="/tmp")
    (run/"tmp").symlink_to(temporary, target_is_directory=True)
    env = os.environ.copy()
    env.update(TMPDIR=temporary, OMNI_KIT_ACCEPT_EULA="Yes", PYTHONUNBUFFERED="1",
               WANDB_MODE="offline", WANDB_DISABLED="true", OMP_NUM_THREADS="4")
    command = [sys.executable, str(Path(__file__).resolve()), "--execute", "--accept-isaac-eula",
               "--worker-run", str(run)]
    for directory in directories:
        command += ["--scene", str(directory)]
    record = dict(simulation_only=True, policy_loaded=False, status="starting", command=command,
        supervisor_pid=os.getpid(), timeout_s=args.timeout,
        source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT.parent, text=True)))
    def save():
        (run/"run.json").write_text(json.dumps(record, indent=2)+"\n")
    print(f"Gallery run: {run}", flush=True)
    with (run/"process.log").open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT.parent, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        record.update(status="running", child_pid=process.pid)
        save()
        try:
            code = process.wait(timeout=args.timeout)
            record["scene_loaded"] = (run/"viewer_ready.json").is_file()
            if code == 0 and not record["scene_loaded"]:
                code = 2
            record.update(status="closed" if code == 0 else "failed", exit_code=code)
        except (KeyboardInterrupt, subprocess.TimeoutExpired) as error:
            record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "timed_out")
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            record["exit_code"] = 130 if isinstance(error, KeyboardInterrupt) else 124
        save()
    raise SystemExit(record["exit_code"])


if __name__ == "__main__":
    main()
