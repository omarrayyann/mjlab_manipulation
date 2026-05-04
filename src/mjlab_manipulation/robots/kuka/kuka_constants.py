from pathlib import Path

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from mjlab_manipulation.robots import RobotCfg

_HERE = Path(__file__).parent
KUKA_XML: Path = _HERE / "xmls" / "kuka.xml"
assert KUKA_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(KUKA_XML))


ARM_ACTUATORS = tuple(
  XmlActuatorCfg(target_names_expr=(f"joint{i}",)) for i in range(1, 8)
)

ARM_HOME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.5),
  joint_pos={
    "joint1": 0.0,
    "joint2": 0.4,
    "joint3": 0.0,
    "joint4": -1.8,
    "joint5": 0.0,
    "joint6": 0.9,
    "joint7": 0.0,
  },
  joint_vel={".*": 0.0},
)

ARTICULATION = EntityArticulationInfoCfg(
  actuators=ARM_ACTUATORS,
  soft_joint_pos_limit_factor=0.95,
)


def get_kuka_robot_cfg() -> RobotCfg:
  return RobotCfg(
    entity_cfg=EntityCfg(
      init_state=ARM_HOME,
      spec_fn=get_spec,
      articulation=ARTICULATION,
    ),
    arm_joint_pattern=r"joint[1-7]",
    arm_collision_geom_pattern=r"link[1-7]_collision.*",
    collision_link_pattern=r"link[1-7]",
    viewer_body="base",
    tool_attach_link="link7",
    tool_attach_pos=(0.0, 0.0, 0.045),
    tool_attach_quat=(0.7071068, 0.0, 0.0, 0.7071068),
  )
