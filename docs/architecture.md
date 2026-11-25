# Architecture

## Execution path

`CLI → official AppLauncher → scene/config → Isaac Lab managers → Newton all-VBD → state/force recorder → offline replay`

The XR entry delegates application startup to the official teleoperation agent. The non-XR entry launches Kit before importing task modules that depend on USD/Kit. Run one scene per process: the Newton manager, rod builder and wrench observer have process-level state.

| Module | Responsibility | Change here when |
| --- | --- | --- |
| `scene/robot.py` | Franka USD, joint names and actuator configuration | Replacing the robot |
| `scene/runtime.py` | Rack/cable instances, material scaling, visual curves and timestep | Changing scene composition or solver configuration |
| `physics/rods.py` | Convert cable D6 links to Newton rod joints during import | Changing cable discretization |
| `physics/vbd.py` | All-VBD manager, robot gravity compensation, body coloring and state updates | Changing solver integration |
| `control/ik.py` | Compute differential IK once per control step | Changing controller behavior |
| `teleop/environment.py` | Per-scene friction, fault pause and event logging | Changing interaction settings |
| `teleop/reset.py` | Restore both Newton states, controls and solver/contact history | Changing reset semantics |
| `teleop/compatibility.py` | Version-specific official input/start gate and graph capture handling | Updating Isaac Teleop or Newton |
| `sensors/joint_wrench.py` | Collect VBD joint reactions without adding contact forces | Updating the observer |
| `sensors/recorder.py` | Transform reactions into sensor frames and record substeps | Changing sensor placement |
| `recording/trajectory.py` | State/camera/action snapshots and atomic chunks | Changing demonstration data |
| `recording/video.py` | Render-only reconstruction and video encoding | Changing presentation |

## Physics preset

The teleoperation preset uses 960 physics steps per simulated second, 60 control steps per simulated second, 16 substeps per control step and 20 VBD iterations. The force recorder expects exactly 16 samples per control step. Changing the timestep requires updating and validating the recorder contract.

All robot bodies, connector bodies and cable rods are solved by VBD. The arm uses implicit position drives and gravity compensation; the gripper is contact-based, without a welded grasp. Effective finger–connector friction defaults to 1.0. Configured actuator effort limits must not be interpreted as verified VBD force saturation. Rod material values are engineering demo settings, not a calibration to a particular physical cable.

Cable segment lengths scale stiffness/damping relative to a 47 mm reference. Visual curves are continuous; physical contact geometry remains segmented. High-load segment separation and interpenetration can occur. Static guide rings are collision geometry, not an inextensible tie constraint.

## Wrench observer

The observer calls the installed VBD joint force/Hessian routine at finalization, stores the selected reactions in `body_parent_f`, shifts torque from body COM to the child joint origin and rotates components into the selected frame. Wrist output uses parent-on-child sign; cable fixed-end output uses child-on-parent sign. Raw substep arrays and control-step mean/peak summaries are retained.

Forces are in newtons and torques in newton-metres. The CSV reports the wrist wrench in the child joint's local frame and fixed-end reactions in world axes, both about the corresponding child joint origin. Gravity and inertia are included; these are not compensated external-contact estimates. Each recording's sensor metadata describes the frame and channel order.

The observer reads private solver state and generates a small Warp wrapper at runtime. It does not add a physical sensor body or require hydroelastic contact. It adds computation and logging cost; deterministic equivalence is not assumed from implementation intent alone. Check source fingerprints and revalidate after SDK changes.

## Extending the sample

Start with `scene/runtime.py` for rack and cable layout, `scene/robot.py` for the robot, and `control/ik.py` for control. Keep asset paths configurable rather than copying a machine's setup into these modules.

The insertion example uses `learning/scene.py` for its single-cable scene and `learning/environment.py` to measure state, project actions, compute rewards and reset individual environments. `learning/runner.py` connects it to RSL-RL PPO and provides deterministic evaluation and GUI play. [Policy learning](learning.md) documents commands, task conditions and output files.
