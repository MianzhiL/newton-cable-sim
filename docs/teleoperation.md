# Headset connection and controls

Complete [Environment setup](setup.md) and [asset configuration](assets.md) first. These instructions use the pinned SDK's official Isaac Teleop/CloudXR integration, with the sample's Franka task and recording enabled.

## Connect a PICO headset

The pinned official guide lists **Pico 4 Ultra with Pico OS 15.4.4U or later**, using HTTPS. Other PICO models are not established by this guide. Use the release-matched web client rather than an arbitrary newer client.

1. Ensure the headset can reach the simulation host. The web client uses **TCP 49100**, **TCP 48322** and **UDP 47998**. Ask the network administrator to permit these only on the required trusted network; guest Wi-Fi isolation may block device-to-device traffic.
2. In the host's graphical desktop terminal, set the variables from the README and launch:

   ```bash
   "$PYTHON" -m cable_robot_sample teleop --sdk "$ISAACLAB_ROOT" \
     --assets "$ASSETS" --robot "$ROBOT" --output "$PWD/runs/teleop01"
   ```

3. Click **Start XR** in the Kit viewport when prompted. The official launcher starts CloudXR and its WSS proxy; do not start a second streaming server.
4. Open [the Isaac Teleop 1.4.x client](https://nvidia.github.io/IsaacTeleop/client/release-1.4.x) in the headset browser. Enter the simulation host's reachable IP in **Server IP**.
5. Follow the client's `https://<host-ip>:48322/` certificate link. Verify that it is your host, accept that host's self-signed certificate, and return to the client. Do not disable certificate checks globally.
6. Press **Connect** and allow the requested immersive-session permissions. Confirm that the rack scene is visible and the right controller is tracked.

By default, valid tracking starts control automatically after approximately 0.75 seconds. For a paused first connection, set `export CABLE_AUTO_START=0` before launching; use the client's Play control to start. This setting also disables the sample's right-B shortcut.

## Operate the robot

| Input | Effect |
| --- | --- |
| Right-controller position and orientation | Set the end-effector target through the official Franka mapping; differential IK produces arm joint targets. |
| Right index trigger | Press to close the gripper; release to open. |
| Right B, with default autostart enabled | Pause or resume control. |
| Client Play / Stop or Pause | Start or pause control through the official command channel. |
| Client Reset | Restore robot and cable states and start a new recording segment. Use Play or right B to resume. |
| Loss of controller tracking | Pause control; after tracking returns, use Play or right B to resume. |

The colored connectors `L05`, `L09`, `R05`, and `R09` are intended for manipulation. Begin with small controller motions to check the mapping. If the physics guard pauses the scene, use Reset rather than repeatedly pushing the same invalid state. Close the application normally to flush recording buffers; force-killing it can lose the final buffered samples.

Optional viewpoint alignment, set **before** launching:

```bash
export CABLE_XR_ANCHOR_POS="0,-0.9,0"
export CABLE_XR_ANCHOR_YAW_DEG="0"
```

These are the defaults. Adjust them to align the operator's XR reference frame with the scene; they do not change cable physics.

## Record and export

Recording starts with the session. It captures third-person camera, actions, joints, cable/connector body poses and force readings; it does not record headset stereo video or encode an MP4 live. Reset separates attempts into segments.

After closing the session, follow [README: Replay a recording](../README.md#replay-a-recording). Keep the complete session directory and its assets. Replay reconstructs saved states and plots recorded force data; it does not rerun physics or provide a solver checkpoint for continuing a simulation.

## Connection checks

| Symptom | Check |
| --- | --- |
| MR mode opens but no scene appears | Start XR on the host; verify the server IP, host certificate and network ports. |
| Scene renders but the arm does not move | Check right-controller tracking and Play state. Inspect the host terminal for a tracking pause or physics fault. |
| Client Play/Reset has no effect | Verify the 1.4.x client, WSS certificate and command connection; seeing video alone does not prove the control channel works. |
| Streaming port already in use | Check for another CloudXR session you own and close it normally; do not kill unrelated processes. |

See the [version-matched official guide](https://github.com/isaac-sim/IsaacLab/blob/57cc1cd08fe150ffcabaf78e834655698e403288/docs/source/how-to/cloudxr_teleoperation.rst) for host and network details. Headset interaction must be checked on the actual target machine; a headless scene check does not validate it.
