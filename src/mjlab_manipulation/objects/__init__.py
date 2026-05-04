"""Per-world mesh variants for graspable objects.

Loads each object's model.xml and normalizes the body/joint names so all
variants share the kinematic structure required by `VariantEntityCfg`. Mesh
and texture file paths are absolutized in-place because mjlab's variant merge
flattens all assets into a single spec that inherits one ``meshdir``, so
relative paths from different objects would all resolve against the same
directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import mujoco

_HERE = Path(__file__).parent

OBJECT_NAMES: tuple[str, ...] = (
  "avocado",
  "banana",
  "bowl",
  "can",
  "cup",
  "kiwi",
  "lemon",
  "mug",
  "orange",
  "peach",
  "pear",
  "tomato",
)


def _normalize(spec: mujoco.MjSpec) -> None:
  """In-place rename to canonical body / joint names so variants align."""
  for body in spec.bodies:
    if body.name == "world":
      continue
    if body.name and body.name.endswith("_object"):
      body.name = "lift_object_mesh"
    elif not body.name:
      body.name = "lift_object"
  for joint in spec.joints:
    if joint.type == mujoco.mjtJoint.mjJNT_FREE:
      joint.name = "lift_object_joint"


def _make_object_spec_fn(name: str) -> Callable[[], mujoco.MjSpec]:
  obj_dir = _HERE / name
  xml_path = obj_dir / "model.xml"
  assert xml_path.exists(), f"missing {xml_path}"

  def spec_fn() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(xml_path))
    for mesh in spec.meshes:
      if mesh.file and not mesh.file.startswith("/"):
        mesh.file = str(obj_dir / mesh.file)
    for tex in spec.textures:
      if tex.file and not tex.file.startswith("/"):
        tex.file = str(obj_dir / tex.file)
    _normalize(spec)
    return spec

  return spec_fn


OBJECT_SPEC_FNS: dict[str, Callable[[], mujoco.MjSpec]] = {
  name: _make_object_spec_fn(name) for name in OBJECT_NAMES
}
