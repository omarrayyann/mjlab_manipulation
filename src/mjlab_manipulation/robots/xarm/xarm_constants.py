import math
from pathlib import Path

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from mjlab_manipulation.robots import RobotCfg

_HERE = Path(__file__).parent
XARM_XML: Path = _HERE / "xmls" / "xarm.xml"
assert XARM_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(XARM_XML))


ARM_ACTUATORS = tuple(
  XmlActuatorCfg(target_names_expr=(f"joint_{i}",)) for i in range(1, 8)
)

ARM_HOME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.4),
  joint_pos={
    "joint_1": 0.0,
    "joint_2": -0.35,
    "joint_3": 0.0,
    "joint_4": 0.7,
    "joint_5": 0.0,
    "joint_6": 1.0,
    "joint_7": math.pi / 2,
  },
  joint_vel={".*": 0.0},
)

ARTICULATION = EntityArticulationInfoCfg(
  actuators=ARM_ACTUATORS,
  soft_joint_pos_limit_factor=0.95,
)


def get_xarm_robot_cfg() -> RobotCfg:
  return RobotCfg(
    entity_cfg=EntityCfg(
      init_state=ARM_HOME,
      spec_fn=get_spec,
      articulation=ARTICULATION,
    ),
    arm_joint_pattern=r"joint_[1-7]",
    arm_collision_geom_pattern=r"link_(6|7)_collision",
    collision_link_pattern=r"link_[1-7]",
    viewer_body="link_0",
    tool_attach_link="link_7",
    tool_attach_pos=(0.0, 0.0, 0.0),
    tool_attach_quat=(0.0, 0.0, 0.0, 1.0),
  )
