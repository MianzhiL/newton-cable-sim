# Cable–Robot Simulation Sample

This repository provides a reference implementation for robotic cable manipulation using Isaac Lab and Newton. It brings together interactive cable simulation, teleoperated demonstration collection, and reinforcement learning for connector insertion.

Start with [teleoperation](docs/teleoperation.md) to manipulate cables and record trajectories and simulated forces, or [policy learning](docs/learning.md) to train, evaluate and run an insertion policy. Both examples use a Franka arm and Newton's VBD solver for the robot, connectors and cables.

## Setup

Use Linux and an existing Isaac Lab + Newton environment matching `src/cable_robot_sample/runtime-lock.json`. Use its Python rather than system Python; some pinned packages are prereleases. Teleoperation also needs the official Isaac Teleop / CloudXR setup and a graphical session. Video export needs FFmpeg on `PATH`.

For a new machine, follow [Environment setup](docs/setup.md) first. For headset connection and controls, see [Teleoperation](docs/teleoperation.md).

Download the prepared scene, Franka and RL bundles and the example policy from [release v0.1.0](https://github.com/MianzhiL/newton-cable-sim/releases/tag/v0.1.0). The demo also requires a separately obtained official rack USD. See [getting the assets](docs/assets.md#getting-the-inputs). Extract the scene and Franka bundles before setting the paths below; rack geometry stays external.

From the repository root:

```bash
export ISAACLAB_ROOT="/path/to/IsaacLab"
export PYTHON="$ISAACLAB_ROOT/.venv/bin/python"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export ASSETS="/path/to/cable-scene-prepared-assets"
export ROBOT="/path/to/franka-prepared-assets/robot.usda"
```

Configure the local official rack once, then check the scene dependencies:

```bash
"$PYTHON" -m cable_robot_sample configure-rack --assets "$ASSETS" \
  --source /path/to/GB300/external.usd
"$PYTHON" -m cable_robot_sample validate-assets --assets "$ASSETS" \
  --robot "$ROBOT" --usd-dependencies
```

This verifies the pinned source and writes a local reference, without copying the rack. Repeat after relocating the original file. The command neither downloads assets nor accepts asset terms for you.

Read and accept the installed Isaac Sim/Kit license yourself, then set `OMNI_KIT_ACCEPT_EULA=YES` in this terminal. The sample does not accept it for you or install/upgrade the SDK.

## Run the scene

```bash
"$PYTHON" -m cable_robot_sample run --assets "$ASSETS" --robot "$ROBOT" \
  --output "$PWD/runs/scene01"
```

This opens the scene, holds the robot pose with the gripper open, and exits after 600 control steps (10 simulated seconds). Set `--steps` for a longer run. For a short non-rendering check, add `--headless --steps 120 --reset-at 60`. Use a new output directory for each run. Asset validation also runs automatically at startup.

## Teleoperate and record

```bash
"$PYTHON" -m cable_robot_sample teleop --sdk "$ISAACLAB_ROOT" \
  --assets "$ASSETS" --robot "$ROBOT" --output "$PWD/runs/teleop01"
```

The launcher delegates input and rendering to the official `teleop_se3_agent.py`. With valid controller tracking, the first session starts automatically after 0.75 seconds. Right B pauses/resumes; tracking loss pauses control. After reset, use Play or B to resume.

For a paused first connection, set `CABLE_AUTO_START=0` before launching and use the client's Play control to start. This also disables the sample's right-B shortcut.

Recording is enabled by default: third-person camera, joints, body poses, actions and reset segments, plus simulated wrist wrench and fixed-end reactions. Use `--no-forces` to disable force observation. These are simulation reaction readings, not calibrated physical sensors.

## Replay a recording

```bash
"$PYTHON" -m cable_robot_sample replay "$PWD/runs/teleop01" --list
"$PYTHON" -m cable_robot_sample replay "$PWD/runs/teleop01" \
  --speed 0.5 --headless --output "$PWD/runs/teleop01-last.mp4"
```

Omitting `--segment` selects the last segment with control frames. To export another segment, choose an ID with `control_frames > 0` from `--list`; segment 0 may contain only initialization. Replay renders recorded states without rerunning physics. Force curves and recorded simulation throughput appear in the video; `--speed 0.5` gives half-speed playback. `replay --help` lists camera and time-range options.

## Recording files

Each session directory contains:

| Path | Contents |
| --- | --- |
| `launch.json` | Command arguments and input paths |
| `events.jsonl` | Runtime events, including resets and fault pauses |
| `trajectory/` | Scene snapshot, camera/action/body/joint data, metadata and chunk index |
| `forces/` | Sensor metadata, per-control-step `summary.csv` and raw substep NPZ chunks |
| `run-result.json` | Completion status for the `run` command |

Reset starts a new recording segment. Force and trajectory samples align by segment and control step. Keep the complete session directory for replay; an MP4 is generated only when you run `replay`. Recordings can contain composed third-party geometry, so review asset terms before sharing them.

## Adapt the demo

| Change | File in `src/cable_robot_sample/` |
| --- | --- |
| Rack/cables and timestep | `scene/runtime.py` |
| Robot and drives | `scene/robot.py` |
| Rod material and discretization | `rod_material.json`, `physics/rods.py` |
| All-VBD integration | `physics/vbd.py` |
| End-effector control | `control/ik.py` |
| Force readout | `sensors/joint_wrench.py`, `sensors/recorder.py` |

[Architecture](docs/architecture.md) explains the execution path, solver assumptions and force conventions. [Policy learning](docs/learning.md) describes the insertion task, PPO configuration and training commands.

The current preset uses 960 Hz physics, 60 Hz control and 20 VBD iterations. High-load interpenetration and contact-grasp slip remain possible; material values are demo settings. The force observer depends on private VBD APIs and checks the expected solver source before enabling itself.

For an asset dependency check or CPU regression tests:

```bash
"$PYTHON" -m cable_robot_sample validate-assets --assets "$ASSETS" --robot "$ROBOT" --usd-dependencies
"$PYTHON" -m unittest discover -s tests -v
```

CPU tests do not validate CUDA physics or headset operation. See [acknowledgments](ACKNOWLEDGMENTS.md) for referenced work.

## License

Sample code is licensed under [BSD-3-Clause](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md) and the notices included with each asset bundle.
