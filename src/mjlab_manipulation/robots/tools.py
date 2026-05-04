from __future__ import annotations

import dataclasses
from pathlib import Path

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.utils.spec_config import CollisionCfg

from mjlab_manipulation.robots import RobotCfg

_ROBOTIQ_XML: Path = Path(__file__).parent / "_robotiq" / "2f85" / "2f85.xml"
assert _ROBOTIQ_XML.exists()


_GRIPPER_ACTUATOR = XmlActuatorCfg(
  target_names_expr=("split",),
  command_field="position",
  transmission_type=TransmissionType.TENDON,
)

_GRIPPER_INIT_JOINTS = {
  "gripper/left_driver_joint": 0.0,
  "gripper/right_driver_joint": 0.0,
}

_GRIPPER_PAD = "gripper/.*(left|right).*pad.*"


def _build_collision_cfg(arm_geom_pattern: str) -> CollisionCfg:
  enabled = f"({arm_geom_pattern}|{_GRIPPER_PAD})"
  return CollisionCfg(
    geom_names_expr=(".*",),
    contype={enabled: 1, ".*": 0},
    conaffinity={enabled: 1, ".*": 0},
    condim={_GRIPPER_PAD: 6, ".*": 3},
    friction={_GRIPPER_PAD: (1.0, 5e-3, 5e-4), ".*": (0.6,)},
    solref={_GRIPPER_PAD: (0.01, 1)},
    priority={_GRIPPER_PAD: 1},
  )


def with_robotiq(robot: RobotCfg) -> RobotCfg:
  original_spec_fn = robot.entity_cfg.spec_fn

  def new_spec_fn() -> mujoco.MjSpec:
    spec = original_spec_fn()
    gripper_spec = mujoco.MjSpec.from_file(str(_ROBOTIQ_XML))
    link = spec.body(robot.tool_attach_link)
    frame = link.add_frame(pos=robot.tool_attach_pos, quat=robot.tool_attach_quat)
    spec.attach(gripper_spec, prefix="gripper/", frame=frame)
    return spec

  art = robot.entity_cfg.articulation
  new_art = dataclasses.replace(
    art, actuators=tuple(art.actuators) + (_GRIPPER_ACTUATOR,)
  )
  init = robot.entity_cfg.init_state
  new_init = dataclasses.replace(
    init, joint_pos={**init.joint_pos, **_GRIPPER_INIT_JOINTS}
  )
  collisions = (_build_collision_cfg(robot.arm_collision_geom_pattern),)
  new_entity = dataclasses.replace(
    robot.entity_cfg,
    spec_fn=new_spec_fn,
    articulation=new_art,
    init_state=new_init,
    collisions=collisions,
  )
  return dataclasses.replace(robot, entity_cfg=new_entity)
