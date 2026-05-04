# mjlab manipulation

A growing collection of manipulation tasks built with [mjlab](https://github.com/mujocolab/mjlab).

## Tasks

<table>
  <tr>
    <th align="center"><code>Reach-{Robot}</code><br/><sub>Move gripper to target.</sub></th>
    <th align="center"><code>Lift-{Robot}</code><br/><sub>Lift a cube (state-based).</sub></th>
  </tr>
  <tr>
    <td><img height="300" src="https://github.com/user-attachments/assets/a1220398-30d5-4721-82bc-3781ace44089" /></td>
    <td><img height="300" src="https://github.com/user-attachments/assets/3ffa5544-2563-4b95-8c01-983611277df5" /></td>
  </tr>
  <tr>
    <th align="center"><code>Lift-{Robot}-RGB</code><br/><sub>Lift a cube (vision-based).</sub></th>
    <th align="center"><code>Lift-Any-{Robot}-RGB</code><br/><sub>Lift 12 objaverse objects (vision-based).</sub></th>
  </tr>
  <tr>
    <td><img height="300" src="https://github.com/user-attachments/assets/72ecd644-c243-46e0-912f-1a66ea20b842" /></td>
    <td><img height="300" src="https://github.com/user-attachments/assets/a784aa22-dba8-44f7-a05f-7fd3959a9b9e" /> </td>
  </tr>
</table>

`{Robot}` is one of `Droid`, `xArm`, `Kuka`, `Gen3`, or `UR5e` ([add your own](#adding-a-robot)).

## Getting Started

```bash
git clone https://github.com/<user>/mjlab_manipulation.git && cd mjlab_manipulation
uv sync
```

Train a task:

```bash
uv run train <task-id> --num_envs 4096
```

Play back a trained policy:

```bash
uv run play <task-id> --wandb-path-checkpoint <wandb-path>
```

## Adding a robot

1. Add a bare-arm MJCF under `robots/<name>/xmls/` (no gripper, they get attached at task time).
2. Write `robots/<name>/<name>_constants.py` returning a `RobotCfg` with the home pose and tool-attach offset.
3. Add it to the `_ROBOTS` dict in each task's `__init__.py`.

## Acknowledgements

- Robot MJCF models are sourced from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
- Object meshes are sourced from [Objaverse](https://objaverse.allenai.org).
