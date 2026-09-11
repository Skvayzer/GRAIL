"""Isaac Lab adapters. Import only AFTER AppLauncher starts Kit."""
import math

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
import isaaclab.sim as sim_utils

from .body_envelope import collision_probes
from .scenes import stair_side_clutter


def solid_cfg(solid, prim_path):
    common = dict(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        activate_contact_sensors=True,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.15, 0.6, 0.25) if solid.support_top else (0.8, 0.2, 0.15)),
    )
    if solid.shape == "box":
        spawn = sim_utils.CuboidCfg(size=solid.size, **common)
    else:
        spawn = sim_utils.CylinderCfg(radius=solid.size[0]/2, height=solid.size[2], axis="Z", **common)
    return RigidObjectCfg(
        prim_path=prim_path, spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=solid.center,
                                                rot=(math.cos(solid.yaw/2), 0., 0., math.sin(solid.yaw/2))),
    )


def attach_clutter(scene_cfg, layout, history_length=4):
    if layout != "stair_side_v1":
        raise ValueError(f"Unknown research clutter layout {layout!r}")
    links = sorted({p.link for p in collision_probes()})
    for link in links:
        setattr(scene_cfg, "research_self_"+link, ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/"+link,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/"+other for other in links if other != link],
            history_length=history_length, update_period=0.0,
        ))
    for solid in stair_side_clutter():
        key = "research_"+solid.name
        setattr(scene_cfg, key, solid_cfg(solid, "{ENV_REGEX_NS}/Research_"+solid.name))
        # Single obstacle body sensed, separately filtered to every collision link.
        setattr(scene_cfg, key+"_contacts", ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Research_"+solid.name,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/"+link for link in links],
            history_length=history_length, update_period=0.0,
        ))
