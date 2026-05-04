from pathlib import Path

import mujoco
import numpy as np
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from mjlab_manipulation.robots import RobotCfg

_HERE = Path(__file__).parent
GEN3_XML: Path = _HERE / "xmls" / "gen3.xml"
assert GEN3_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(GEN3_XML))


ARM_ACTUATORS = tuple(
  XmlActuatorCfg(target_names_expr=(f"joint_{i}",)) for i in range(1, 8)
)

ARM_HOME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.5),
  joint_pos={
    "joint_1": 0.0,
    "joint_2": 0.3,
    "joint_3": np.pi,
    "joint_4": -1.8,
    "joint_5": 0.0,
    "joint_6": -1.0,
    "joint_7": np.pi / 2,
  },
  joint_vel={".*": 0.0},
)

ARTICULATION = EntityArticulationInfoCfg(
  actuators=ARM_ACTUATORS,
  soft_joint_pos_limit_factor=0.95,
)


def get_gen3_robot_cfg() -> RobotCfg:
  return RobotCfg(
    entity_cfg=EntityCfg(
      init_state=ARM_HOME,
      spec_fn=get_spec,
      articulation=ARTICULATION,
    ),
    arm_joint_pattern=r"joint_[1-7]",
    arm_collision_geom_pattern=r"(shoulder_link|half_arm_[12]_link|forearm_link|spherical_wrist_[12]_link|bracelet_link)",
    collision_link_pattern=r"(shoulder_link|half_arm_[12]_link|forearm_link|spherical_wrist_[12]_link|bracelet_link)",
    viewer_body="base_link",
    tool_attach_link="bracelet_link",
    tool_attach_pos=(0.0, 0.0, -0.061525),
    tool_attach_quat=(0.0, 1.0, 0.0, 0.0),
  )
