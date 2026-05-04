from pathlib import Path

import mujoco
import numpy as np
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from mjlab_manipulation.robots import RobotCfg

_HERE = Path(__file__).parent
DROID_XML: Path = _HERE / "xmls" / "droid.xml"
assert DROID_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(DROID_XML))


ARM_ACTUATORS = tuple(
  XmlActuatorCfg(target_names_expr=(f"joint_{i}",)) for i in range(1, 8)
)

ARM_HOME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.5),
  joint_pos={
    "joint_1": 0.0,
    "joint_2": -1 / 5 * np.pi,
    "joint_3": 0.0,
    "joint_4": -4 / 5 * np.pi,
    "joint_5": 0.0,
    "joint_6": 3 / 5 * np.pi,
    "joint_7": 0.0,
  },
  joint_vel={".*": 0.0},
)

ARTICULATION = EntityArticulationInfoCfg(
  actuators=ARM_ACTUATORS,
  soft_joint_pos_limit_factor=0.95,
)


def get_droid_robot_cfg() -> RobotCfg:
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
    tool_attach_pos=(0.0, 0.0, 0.107),
    tool_attach_quat=(0.7071068, 0.0, 0.0, 0.7071068),
  )
