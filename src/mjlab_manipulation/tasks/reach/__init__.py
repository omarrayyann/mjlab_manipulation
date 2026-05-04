from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from mjlab_manipulation.robots.droid.droid_constants import get_droid_robot_cfg
from mjlab_manipulation.robots.gen3.gen3_constants import get_gen3_robot_cfg
from mjlab_manipulation.robots.kuka.kuka_constants import get_kuka_robot_cfg
from mjlab_manipulation.robots.ur5e.ur5e_constants import get_ur5e_robot_cfg
from mjlab_manipulation.robots.xarm.xarm_constants import get_xarm_robot_cfg
from mjlab_manipulation.tasks.lift.rl_cfg import lift_ppo_runner_cfg

from .env_cfgs import reach_env_cfg

_ROBOTS = {
  "Droid": get_droid_robot_cfg,
  "xArm": get_xarm_robot_cfg,
  "Gen3": get_gen3_robot_cfg,
  "Kuka": get_kuka_robot_cfg,
  "UR5e": get_ur5e_robot_cfg,
}

for name, robot_fn in _ROBOTS.items():
  exp = f"reach_{name.lower()}"
  register_mjlab_task(
    task_id=f"Reach-{name}",
    env_cfg=reach_env_cfg(robot_cfg_fn=robot_fn),
    play_env_cfg=reach_env_cfg(play=True, robot_cfg_fn=robot_fn),
    rl_cfg=lift_ppo_runner_cfg(experiment_name=exp),
    runner_cls=MjlabOnPolicyRunner,
  )
