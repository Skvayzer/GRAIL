"""Pure-data, explicit M2 task overlay. Original retention configuration unchanged."""
from .avoidance_task import FEET, LOWER


def avoidance_overrides():
    prefix = "gear_sonic.research.avoidance_mdp"
    result = {
        "manager_env.rewards._target_": prefix+".AvoidanceRewardsCfg",
        "manager_env.terminations._target_": prefix+".AvoidanceTerminationsCfg",
        "manager_env.rewards.tracking_vr_5point_local": None,
        "manager_env.rewards.tracking_relative_body_pos.params.body_names": list(LOWER),
        "manager_env.rewards.tracking_relative_body_ori.params.body_names": list(LOWER),
        "manager_env.terminations.ee_body_pos.params.body_names": list(FEET),
        # No default hand/elbow terrain-contact exemption in avoidance episodes.
        "manager_env.rewards.undesired_contacts.params.sensor_cfg.body_names":
            ["^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$).+$"],
    }
    def reward(name, func, weight, params=None):
        base = "manager_env.rewards."+name
        result[base+"._target_"] = "isaaclab.managers.RewardTermCfg"
        result[base+".func"] = func
        result[base+".weight"] = weight
        for key, value in (params or {}).items():
            result[base+".params."+key] = value
    reward("tracking_body_linvel", "gear_sonic.envs.manager_env.mdp:tracking_body_linvel_error", .2,
           dict(command_name="motion", std=1., body_names=list(LOWER)))
    reward("tracking_body_angvel", "gear_sonic.envs.manager_env.mdp:tracking_body_angvel_error", .1,
           dict(command_name="motion", std=3., body_names=list(LOWER)))
    reward("cat_feet_tracking", "gear_sonic.envs.manager_env.mdp:tracking_body_pos_error", 2.,
           dict(command_name="motion", std=.1, body_names=list(FEET)))
    reward("cat_clearance", prefix+":clearance_penalty", -2.)
    reward("cat_contact", prefix+":contact_penalty", -1.)
    reward("cat_root_progress", prefix+":root_progress", 4.)
    reward("cat_failure", prefix+":failure_event", -5.)
    for name, func in (("cat_contact", "contact_failure"), ("cat_foot_world", "foot_failure")):
        base = "manager_env.terminations."+name
        result[base+"._target_"] = "isaaclab.managers.TerminationTermCfg"
        result[base+".func"] = prefix+":"+func
    return result
