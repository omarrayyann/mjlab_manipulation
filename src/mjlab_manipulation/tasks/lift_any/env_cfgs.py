from __future__ import annotations

from typing import Literal

from mjlab.entity import VariantEntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_manipulation.objects import OBJECT_SPEC_FNS
from mjlab_manipulation.robots.droid.droid_constants import get_droid_robot_cfg
from mjlab_manipulation.tasks.lift.env_cfgs import lift_vision_env_cfg

_OBJECT_HALF_HEIGHT = 0.04


def lift_any_vision_env_cfg(
  cam_type: Literal["rgb", "depth"] = "rgb",
  play: bool = False,
  robot_cfg_fn=get_droid_robot_cfg,
) -> ManagerBasedRlEnvCfg:
  cfg = lift_vision_env_cfg(cam_type=cam_type, play=play, robot_cfg_fn=robot_cfg_fn)

  prev_box = cfg.scene.entities["box"]
  cfg.scene.entities["box"] = VariantEntityCfg(
    variants=OBJECT_SPEC_FNS,
    init_state=prev_box.init_state,
  )

  cfg.scene.sensors = tuple(
    s if s.name != "box_table_contact"
    else ContactSensorCfg(
      name="box_table_contact",
      primary=ContactMatch(mode="subtree", pattern="lift_object", entity="box"),
      secondary=ContactMatch(mode="subtree", pattern="table", entity="table"),
      fields=("found",),
      reduce="none",
      num_slots=1,
    )
    for s in cfg.scene.sensors
  )

  cfg.commands["lift_height"].object_half_height = _OBJECT_HALF_HEIGHT

  return cfg
