# Cable insertion policy

This example trains a Franka arm to align and insert a held optical connector. Each episode starts from a physically grasped state near the port. The connector remains contact-grasped, so cable forces can move it relative to the fingers.

## Prepare the task

Follow [environment setup](setup.md) and configure the prepared Franka asset. Use `rsl-rl-lib==5.4.1` and `tensordict==0.13.0` from the matching SDK. The RL example has its own single-cable scene and initial-state bundle, `cable-insertion-assets.zip`.

Extract the bundle, then run from the repository root with `PYTHON`, `PYTHONPATH` and `ROBOT` set as in the README:

```bash
export TASK_ASSETS="/path/to/cable-insertion-assets"
"$PYTHON" -m cable_robot_sample configure-rack --assets "$TASK_ASSETS" \
  --source /path/to/GB300/external.usd
```

The bundle contains the cable and collision geometry, held-state reference and initial-state bank. Its rack appearance references the same separately obtained official USD used by the teleoperation example. Keep the bundle's notices with it.

## Train

```bash
"$PYTHON" -m cable_robot_sample train --task-assets "$TASK_ASSETS" --robot "$ROBOT" \
  --num-envs 16 --iterations 100 --headless --output "$PWD/runs/train01"
```

One process runs parallel environments on the selected GPU (`--device cuda:0` by default). Start with a small environment count on a new machine, then increase it according to available memory and measured throughput. One iteration collects 512 control steps per environment and performs one PPO update. `--iterations` counts updates, not episodes.

To continue from a checkpoint, add `--checkpoint /path/to/model_100.pt` and use a new output directory. The actor, critic and both Adam states resume; environments start new episodes from the bank. Checkpoints are saved every ten updates and at completion.

## Evaluate and view a policy

The companion `insertion-policy.pt` checkpoint is ready for evaluation, viewing or further training. Set `POLICY` to that file, or to a checkpoint from your own run:

```bash
export POLICY="/path/to/insertion-policy.pt"
"$PYTHON" -m cable_robot_sample evaluate --task-assets "$TASK_ASSETS" --robot "$ROBOT" \
  --checkpoint "$POLICY" --num-envs 4 --headless --output "$PWD/runs/eval01"

"$PYTHON" -m cable_robot_sample play --task-assets "$TASK_ASSETS" --robot "$ROBOT" \
  --checkpoint "$POLICY" --initial-ids 8159 --record --output "$PWD/runs/play01"
```

Evaluation uses deterministic actions and covers all 285 held-out states by default, once each. `--initial-ids` selects a diagnostic subset; its result reports the actual IDs and episode count. Failed episodes count as failures. Play opens the GUI and runs one held-out state, or a specified sequence, to completion.

With `--record`, play saves third-person state trajectories for the existing [replay command](../README.md#replay-a-recording). This RL recording contains the policy motion; wrench recording is provided by the teleoperation example.

## Task and learning configuration

| Component | Configuration |
| --- | --- |
| Simulation | Pure VBD, 960 Hz physics, 60 Hz control, 20 solver iterations |
| Grasp | Contact-based; effective finger–connector friction 30 |
| Initial states | 8,158 admitted training states sampled at reset; 285 separate evaluation states |
| Actor / critic | Separate 256–256 ELU MLPs; 234 actor inputs and 248 critic inputs |
| Actions | Seven arm-joint target increments; gripper targets stay closed |
| PPO | 512-step rollouts, five epochs, four minibatches; actor LR 1e-4, critic LR 3e-4 |
| Episode | At most 40 simulated seconds; terminal states do not bootstrap |

The actor observes robot and cable state, port-relative pose, grasp deviation and control history. The critic additionally observes relative motion and contact-risk features. `learning/ppo.json` contains the complete optimizer and network settings.

Success requires at least 9 mm insertion depth, less than 2 mm lateral error and 5° angular error, plus the corridor/grasp checks, held for 30 control steps. Sustained grasp loss, leaving the task region, deadline or a physical-state fault ends an episode. Joint targets use bounded increments, slew and target lead; an ordinary tracking lag is handled by control projection.

The reward combines a +20 success / −5 failure event with terminal-zero potential shaping toward the insertion pose. Time, action variation, grasp drift/rotation and persistent cable–arm contact contribute bounded auxiliary costs. Their combined episode budget is 0.1; gamma and lambda are 1.

Individual physical faults terminate the affected episode and cold-reset its solver state. Widespread or persistent faults stop the run and retain diagnostics. Training reset can quarantine and resample a rejected initial state. An evaluation reset rejection stops the run as incomplete; it never substitutes a test state or emits an aggregate rate for an incomplete evaluation.

## Outputs and modification points

`training.jsonl` records losses, rollout/update time, transitions per second and simulated/wall-clock time ratio. `episodes.jsonl` records training outcomes; `evaluation.jsonl` and `result.json` hold evaluation outcomes. Training outcomes are exploration statistics, not the deterministic evaluation success rate.

Edit `learning/scene.py` for the scene, `control.py` for action mapping, `observations.py` for policy inputs, `reward.py` / `termination.py` for the task, and `ppo.json` for learning settings. Changing body/joint layout requires regenerating the reference and initial-state bank; existing snapshots are specific to their physical model.
