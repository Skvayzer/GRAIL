"""Isaac adapters for the opt-in posture-avoidance profile; import after Kit."""
from isaaclab.utils import configclass
from gear_sonic.envs.manager_env.mdp.rewards import RewardsCfg
from gear_sonic.envs.manager_env.mdp.terminations import TerminationsCfg
from .avoidance_task import failure_event_rate


@configclass
class AvoidanceRewardsCfg(RewardsCfg):
    cat_clearance = None
    cat_contact = None
    cat_root_progress = None
    cat_failure = None
    cat_feet_tracking = None


@configclass
class AvoidanceTerminationsCfg(TerminationsCfg):
    cat_contact = None
    cat_foot_world = None


def context(env):
    task = getattr(env, "research_avoidance", None)
    if task is None:
        raise ValueError("Avoidance task context was not attached; no zero-reward fallback")
    return task


def clearance_penalty(env):
    return context(env).measure()["clearance_penalty"]


def contact_penalty(env):
    return context(env).measure()["contact_penalty"]


def root_progress(env):
    return context(env).measure()["root_progress"]


def contact_failure(env):
    return context(env).measure()["contact_failure"]


def foot_failure(env):
    task = context(env)
    return task.measure()["foot_world_error"] > task.spec.foot_world_failure_m


def failure_event(env):
    # RewardManager multiplies by dt. This is one cost per true termination,
    # not a cost whose magnitude shrinks with the chosen action frequency.
    return failure_event_rate(env.termination_manager.terminated, env.step_dt)
