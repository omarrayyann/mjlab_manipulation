from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from mjlab.entity import EntityCfg
from mjlab.envs.mdp.actions import DifferentialIKActionCfg, TendonLengthActionCfg

_DEFAULT_GRIPPER_XML: Path = (
  Path(__file__).parent / "_robotiq" / "2f85" / "2f85.xml"
)


@dataclass
class RobotCfg:
  entity_cfg: EntityCfg
  arm_joint_pattern: str = r"joint_.*"
  gripper_joint_pattern: str = r".*driver_joint"
  gripper_actuator_name: str = "split"
  gripper_scale: float = 255.0
  gripper_offset: float = 0.0
  fingertip_geom_pattern: str = r"(left|right).*pad.*"
  collision_link_pattern: str = r"link_[1-7]"
  viewer_body: str = "link_0"
  ee_site: str = "grasp_site"
  camera_name: str = "robot/gripper/wrist_camera"
  ik_damping: float = 0.1
  orientation_weight: float = 0.0
  arm_collision_geom_pattern: str = r"link_[1-7]_collision.*"
  tool_attach_link: str = "link_7"
  tool_attach_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
  tool_attach_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
  gripper_xml: Path = field(default_factory=lambda: _DEFAULT_GRIPPER_XML)

  def arm_action_cfg(self) -> DifferentialIKActionCfg:
    return DifferentialIKActionCfg(
      entity_name="robot",
      actuator_names=(self.arm_joint_pattern,),
      frame_name=self.ee_site,
      frame_type="site",
      use_relative_mode=True,
      delta_pos_scale=0.05,
      delta_ori_scale=0.25,
      position_weight=1.0,
      orientation_weight=self.orientation_weight,
      damping=self.ik_damping,
      max_dq=0.5,
      joint_limit_weight=1.0,
    )

  def gripper_action_cfg(self) -> TendonLengthActionCfg:
    return TendonLengthActionCfg(
      entity_name="robot",
      actuator_names=(self.gripper_actuator_name,),
      scale=self.gripper_scale,
      offset=self.gripper_offset,
    )
