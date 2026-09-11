"""Opt-in desktop overlay/capture; imported only after Kit and CAT preflight."""
import asyncio
import json
from pathlib import Path


class CatDemo:
    def __init__(self, sim, config):
        import omni.ui as ui
        self.sim = sim
        self.output = Path(config.research_cat_gui_output)
        self.eye, self.target = list(config.viewer_eye), list(config.viewer_target)
        self.capture_task = None
        self.window = ui.Window("GRAIL + CAT stairs - simulation only", width=440, height=185)
        with self.window.frame:
            with ui.VStack(spacing=5):
                ui.Label("Released GRAIL controller + physical CAT clutter", height=22)
                ui.Label("Repeating stair reference; NOT learned avoidance.", height=22)
                self.status = ui.Label("Reference preflight passed. First episode auditing...", height=22)
                ui.Label("Close Isaac Sim to stop. No real-robot connection.", height=22)
                ui.Button("Reset overview camera", clicked_fn=self.reset_camera, height=25)
        self.reset_camera()
        (self.output / "viewer_ready.json").write_text(json.dumps(dict(
            simulation_only=True, repeats_reference=True, avoidance_training=False,
            diagnostic_scope="first episode only", camera_eye=self.eye, camera_target=self.target,
            cat_scene=str(config.manager_env.config.research_cat_scene)), indent=2) + "\n")
        print(f"CAT_STAIRS_VIEWER_READY {self.output}", flush=True)

    def reset_camera(self):
        self.sim.set_camera_view(self.eye, self.target)

    def first_episode_finished(self, report):
        result = "completed without failure" if report["rollout_completed_without_failure"] else "failed"
        self.status.text = f"First episode {result}; replays are not audited."

    async def capture(self):
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
        try:
            helper = capture_viewport_to_file(get_active_viewport(), str(self.output / "overview.png"))
            await helper.wait_for_result()
            print(f"CAT_STAIRS_SCREENSHOT {self.output / 'overview.png'}", flush=True)
        except Exception as error:
            print(f"Screenshot failed (viewer remains open): {error}", flush=True)

    def step(self, step_count):
        if step_count == 120:
            self.capture_task = asyncio.ensure_future(self.capture())

    def close(self):
        if self.capture_task is not None and not self.capture_task.done():
            self.capture_task.cancel()
