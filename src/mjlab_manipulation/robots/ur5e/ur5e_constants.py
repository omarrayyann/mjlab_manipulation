from pathlib import Path

import mujoco
import numpy as np
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from mjlab_manipulation.robots import RobotCfg

_HERE = Path(__file__).parent
UR5E_XML: Path = _HERE / "xmls" / "ur5e.xml"
assert UR5E_XML.exists()

_ARM_JOINT_NAMES = (
  "shoulder_pan_joint",
  "shoulder_lift_joint",
  "elbow_joint",
  "wrist_1_joint",
  "wrist_2_joint",
  "wrist_3_joint",
)


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(UR5E_XML))


ARM_ACTUATORS = tuple(
  XmlActuatorCfg(target_names_expr=(name,)) for name in _ARM_JOINT_NAMES
)

ARM_HOME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.5),
  joint_pos={
    "shoulder_pan_joint": np.pi,
    "shoulder_lift_joint": -np.pi / 2,
    "elbow_joint": np.pi / 2,
    "wrist_1_joint": -np.pi / 2,
    "wrist_2_joint": -np.pi / 2,
    "wrist_3_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

ARTICULATION = EntityArticulationInfoCfg(
  actuators=ARM_ACTUATORS,
  soft_joint_pos_limit_factor=0.95,
)


def get_ur5e_robot_cfg() -> RobotCfg:
  return RobotCfg(
    entity_cfg=EntityCfg(
      init_state=ARM_HOME,
      spec_fn=get_spec,
      articulation=ARTICULATION,
    ),
    arm_joint_pattern=r"(shoulder_pan|shoulder_lift|elbow|wrist_[123])_joint",
    arm_collision_geom_pattern=r"(shoulder_link|upper_arm_link|forearm_link|wrist_[123]_link)",
    collision_link_pattern=r"(shoulder_link|upper_arm_link|forearm_link|wrist_[123]_link)",
    viewer_body="base",
    tool_attach_link="wrist_3_link",
    tool_attach_pos=(0.0, 0.1, 0.0),
    tool_attach_quat=(-1.0, 1.0, 0.0, 0.0),
  )
