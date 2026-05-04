from __future__ import annotations

from dataclasses import dataclass, field

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp import joint_pos_rel
from mjlab.envs.mdp.actions.actions import RelativeJointPositionActionCfg
from mjlab.managers import ObservationTermCfg, RewardTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_manipulation.robots.droid.droid_constants import get_droid_robot_cfg
from mjlab_manipulation.tasks.lift.env_cfgs import (
  ee_position_in_robot_base_frame,
  lift_env_cfg,
  nan_safe,
  target_position_in_robot_base_frame,
)


class ReachCommand(CommandTerm):
  cfg: ReachCommandCfg

  def __init__(self, cfg: ReachCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.robot_name]
    site_ids, _ = self.robot.find_sites(cfg.ee_site_name)
    self.ee_site_id = site_ids[0]
    self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self.episode_success = torch.zeros(self.num_envs, device=self.device)
    self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["episode_success"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.target_pos

  @property
  def ee_pos_w(self) -> torch.Tensor:
    return self.robot.data.site_pos_w[:, self.ee_site_id]

  def _update_metrics(self) -> None:
    err = torch.norm(self.target_pos - self.ee_pos_w, dim=-1)
    at_goal = (err < self.cfg.success_threshold).float()
    self.episode_success = torch.maximum(self.episode_success, at_goal)
    self.metrics["position_error"] = err
    self.metrics["at_goal"] = at_goal
    self.metrics["episode_success"] = self.episode_success

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    self.episode_success[env_ids] = 0.0
    r = self.cfg.target_position_range
    lower = torch.tensor([r.x[0], r.y[0], r.z[0]], device=self.device)
    upper = torch.tensor([r.x[1], r.y[1], r.z[1]], device=self.device)
    target = sample_uniform(lower, upper, (n, 3), device=self.device)
    self.target_pos[env_ids] = target + self._env.scene.env_origins[env_ids]

  def _update_command(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    for batch in env_indices:
      visualizer.add_sphere(
        center=self.target_pos[batch].cpu().numpy(),
        radius=0.03,
        color=(0.0, 1.0, 0.0, 0.5),
        label=f"target_position_{batch}",
      )


@dataclass(kw_only=True)
class ReachCommandCfg(CommandTermCfg):
  robot_name: str = "robot"
  ee_site_name: str
  success_threshold: float = 0.05

  @dataclass
  class TargetPositionRangeCfg:
    x: tuple[float, float] = (0.3, 0.55)
    y: tuple[float, float] = (-0.25, 0.25)
    z: tuple[float, float] = (0.6, 0.85)

  target_position_range: TargetPositionRangeCfg = field(
    default_factory=TargetPositionRangeCfg
  )

  def build(self, env: ManagerBasedRlEnv) -> ReachCommand:
    return ReachCommand(self, env)


def ee_target_distance(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  command: ReachCommand = env.command_manager.get_term(command_name)
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  distance = torch.norm(ee_pos_w - command.target_pos, dim=-1)
  return 1.0 - torch.tanh(distance / std)


def reach_env_cfg(
  play: bool = False, robot_cfg_fn=get_droid_robot_cfg
) -> ManagerBasedRlEnvCfg:
  cfg = lift_env_cfg(play=play, robot_cfg_fn=robot_cfg_fn)
  robot = robot_cfg_fn()

  cfg.actions.pop("ee_ik", None)
  cfg.actions.pop("gripper", None)
  cfg.actions["arm_joints"] = RelativeJointPositionActionCfg(
    entity_name="robot",
    actuator_names=(robot.arm_joint_pattern,),
    scale=0.05,
  )

  cfg.scene.entities.pop("box", None)
  cfg.scene.sensors = tuple(
    s for s in cfg.scene.sensors if s.name != "box_table_contact"
  )

  cfg.commands.pop("lift_height", None)
  cfg.commands["reach"] = ReachCommandCfg(
    ee_site_name=robot.ee_site,
    resampling_time_range=(5.0, 5.0),
    debug_vis=True,
  )

  ee_cfg = SceneEntityCfg("robot", site_names=(robot.ee_site,))
  cfg.rewards.clear()
  cfg.rewards["reaching_target"] = RewardTermCfg(
    func=nan_safe(ee_target_distance),
    weight=10.0,
    params={"std": 0.2, "command_name": "reach", "asset_cfg": ee_cfg},
  )
  cfg.rewards["reaching_target_fine"] = RewardTermCfg(
    func=nan_safe(ee_target_distance),
    weight=10.0,
    params={"std": 0.05, "command_name": "reach", "asset_cfg": ee_cfg},
  )

  for group in ["actor", "critic"]:
    terms = cfg.observations[group].terms
    for name in ("object_position", "object_orientation", "target_object_position"):
      terms.pop(name, None)
    terms["target_position"] = ObservationTermCfg(
      func=target_position_in_robot_base_frame,
      params={"command_name": "reach"},
    )
    terms["ee_position"] = ObservationTermCfg(
      func=ee_position_in_robot_base_frame,
      params={"asset_cfg": ee_cfg},
    )
    terms["arm_joint_pos"] = ObservationTermCfg(
      func=joint_pos_rel,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(robot.arm_joint_pattern,))
      },
    )

  cfg.observations["actor"].terms["target_position"].noise = Unoise(
    n_min=-0.01, n_max=0.01
  )

  if play:
    cfg.commands["reach"].resampling_time_range = (4.0, 4.0)

  return cfg
