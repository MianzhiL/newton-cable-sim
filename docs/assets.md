# Asset setup

Each example uses a prepared scene bundle, the shared Franka bundle and an external rack USD. Use `cable-scene-prepared-assets.zip` for teleoperation or `cable-insertion-assets.zip` for RL. The bundles contain the task layout and connector geometry/textures, so no asset conversion is needed. The original rack is obtained separately and referenced locally.

## Prepared pack

```text
prepared-scene/
  manifest.json
  rack.usda
  rack_visuals.usdc       rack placement and instancing
  rack_reference.usdc     references to external source meshes; no rack geometry arrays
  rack_source.json        official source URL and SHA256
  Cable1.usdc
  Cable2.usdc
  ...
  rod_material.json       optional override of the packaged demo material
  textures/               connector textures
  external/gb300.usda      generated locally by configure-rack; not shipped
  robot.usda              optional location; --robot accepts another path
```

The current runtime consumes a prepared scene; it does not reconstruct the original rack layout from raw downloaded assets. The original `Intrinsic_assets` download is not a substitute for this directory: it does not contain this sample's rack layout, cable placement manifest or prepared collision geometry.

## Getting the inputs

Download the bundles and example checkpoint from [release v0.1.0](https://github.com/MianzhiL/newton-cable-sim/releases/tag/v0.1.0). The release includes `SHA256SUMS` for integrity checks. Download the desired files and compare their checksums using `sha256sum`; the values are also listed below.

1. Extract `cable-scene-prepared-assets.zip` and set `ASSETS` to the extracted `cable-scene-prepared-assets/` directory containing `manifest.json`. Keep `ASSET_SOURCES.md` and `LICENSE-AIC.txt` with it.
2. Extract `franka-prepared-assets.zip` and set `ROBOT` to `franka-prepared-assets/robot.usda`. Keep `LICENSE.txt`, `UPSTREAM_README.md`, `CHANGES.md` and `manifest.json` with the robot.
3. Obtain the official rack USD from the URL in `rack_source.json` under its applicable terms. Run `"$PYTHON" -m cable_robot_sample configure-rack --assets "$ASSETS" --source /path/to/external.usd`. The command checks its SHA256 and creates a local reference layer. It does not copy or download the source. Run the dependency check below before launching.

The original rack must remain available. Rerun `configure-rack` if it moves. When sharing the prepared bundle, exclude the locally generated `external/` directory and any downloaded rack; do not flatten the composed rack into the bundle.

Archive SHA256 values:

```text
cable-scene-prepared-assets.zip
05b7fdd9bf13a050b3524e736fcb5d04c1ca41d5ad8fee92e0872af7cbedd9ca

franka-prepared-assets.zip
c80195b3e10d3ffbebefdf93e7ade0735c5b0c1297d69eb3fd841cd64e1b4d06

cable-insertion-assets.zip
6653bea4199dd24f7ee0749e8e1def40c30dba933f59cacc5a34a8e1789106a2

insertion-policy.pt
3e1b68bc8d88eefb99a9e5575463a9e7482c5a75eb1873167d41b01a9ee5d2cb
```

### Original sources

These links identify the inputs used to build the scene, not ready-to-run sample bundles or permission to redistribute their derivatives.

| Input | Source |
| --- | --- |
| FR3 with Franka Hand | [Pinned Newton model and license](https://github.com/newton-physics/newton-assets/tree/a96f0973b6ae69c90609f9fafef9d2c1db2d6431/franka_emika_panda) |
| AIC connectors and cable | [Pinned AIC Isaac integration instructions](https://github.com/intrinsic-dev/aic/blob/6546127c1c15d5786f59c91400f5fc5ad6894448/aic_utils/aic_isaac/README.md), which link to the NVIDIA-prepared `Intrinsic_assets.zip` |
| GB300 rack | [Pinned Isaac Lab scene source](https://github.com/isaac-sim/IsaacLab/blob/23fb89d890ad667568b8301841eb5b256a6adde5/source/isaaclab_tasks/isaaclab_tasks/contrib/gb300_sharpa_insertion/visual_scene.py), which references `RizonSharpaRJ45/Environments/GB300/external.usd` |

The AIC instructions describe their own Isaac Lab 2.3.2 integration. They are an asset-source reference here, not installation instructions for this Newton sample. Use this sample's pinned runtime instead.

## Manifest and model mapping

`manifest.json` contains a `cables` list. Each item needs:

- `name`: `Cable1`, `Cable2`, ... in list order, matching the USD filenames.
- `identifier`: human-readable port label; `operable`: whether its movable connector is intended for interaction.
- `rod_count`, `segment_lengths_m`: the physical discretization, in metres.
- `jacket_radius_m`, `jacket_color`: continuous visual curve appearance; color is linear RGB.
- `connector_color`: null or the existing colored-connector metadata, including `color_linear`.
- `bodies.sc_plug_link.position` and `quaternion_xyzw`: world spawn transform for the cable assembly.

The imported body/joint paths are part of the contract. Rod bodies end in `CableN/link_K`; the movable head includes `sfp_module_link`. The fixed-end sensor identifies `joint_21_fixedJoint2`. The robot uses `fr3_joint1..7`, `fr3_finger_joint*`, `fr3_hand` and `fr3_hand_joint`. A different robot/connector requires explicit mapping changes, not just replacing a filename.

## Validate a pack

```bash
"$PYTHON" -m cable_robot_sample validate-assets --assets "$ASSETS" --robot "$ROBOT"
"$PYTHON" -m cable_robot_sample validate-assets --assets "$ASSETS" --robot "$ROBOT" --usd-dependencies
```

The first check uses Python/JSON only. The second needs `pxr` available in the SDK Python. It checks USD dependency resolution, not collision quality, legal rights or insertion feasibility.

## Licenses and redistribution

The RL example uses the separate `cable-insertion-assets.zip` bundle: a single-cable insertion scene, physical reference and held-state bank. Configure its external rack with the same `configure-rack` command, then follow [Policy learning](learning.md). Its collision geometry and initial-state ordering belong together; the 24-cable scene is a different task.

| Component | License or terms | Distribution |
| --- | --- | --- |
| FR3 with Franka Hand | Model-specific Apache-2.0 license with an included BSD notice | Prepared robot bundle; retain its license, attribution and modification notice |
| AIC connectors | Apache-2.0; license included in the scene bundle | Selected converted models and textures; retain their license, source attribution and modification notice |
| GB300 rack | NVIDIA Isaac asset terms | Original geometry is not bundled; local reference to the separately obtained official USD |

The prepared robot is standalone and needs no external texture/mesh files.
Its metadata is sanitized without changing geometry or physical properties.
Use `--robot /path/to/franka-prepared-assets/robot.usda`; it is not a complete
scene pack. The cable scene contains the selected converted connectors, their
textures and generated task geometry, not the entire downloaded conversion
archive. Rack source geometry and machine-local reference wrappers are excluded.

For NVIDIA-provided additional assets, review the applicable
[Additional Software and Materials License](https://docs.isaacsim.omniverse.nvidia.com/latest/common/license-isaac-sim-additional.html)
and any asset-specific terms. External referencing does not waive the rack's applicable use or modification terms. Third-party asset licenses do not grant a license to the sample's source code.
