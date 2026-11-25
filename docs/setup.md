# Environment setup

Use Linux x86_64 with an NVIDIA GPU and driver supported by the pinned Isaac Sim runtime. XR requires a graphical desktop session. This sample targets Python 3.12 and a specific prerelease SDK; installing the latest Isaac Lab or Newton is not equivalent.

## 1. Prepare the host

- Install Git and [uv](https://docs.astral.sh/uv/getting-started/installation/).
- Ask the machine administrator to provide the required NVIDIA driver, Vulkan runtime (`libvulkan1`), `libbsd0`, and FFmpeg. Do not replace a working driver solely for this sample.
- Check `nvidia-smi` and `ffmpeg -version`. Follow the [official rendering requirements](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/common/technical-requirements.html) for driver compatibility.

## 2. Install the pinned SDK

If a matching SDK already exists, keep it and skip to verification. Otherwise choose a new directory and run:

```bash
export ISAACLAB_ROOT="$HOME/IsaacLab-cable"
git clone --no-checkout https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_ROOT"
git -C "$ISAACLAB_ROOT" checkout --detach 57cc1cd08fe150ffcabaf78e834655698e403288
cd "$ISAACLAB_ROOT"
uv sync --frozen --extra isaacsim --extra teleop --python 3.12.13
export PYTHON="$ISAACLAB_ROOT/.venv/bin/python"
```

This installs the SDK and the Isaac Sim/Isaac Teleop extras into its project environment. Preserve the upstream `uv.lock`; do not resolve against newer package versions or combine the `ov`/`ovphysx` extras with `teleop`. Downloads require access to the package indexes configured by this SDK. If a pinned prerelease is unavailable, obtain a compatible runtime before proceeding rather than silently substituting another release.

The matching [official XR setup guide](https://github.com/isaac-sim/IsaacLab/blob/57cc1cd08fe150ffcabaf78e834655698e403288/docs/source/how-to/cloudxr_teleoperation.rst) is also available at `$ISAACLAB_ROOT/docs/source/how-to/cloudxr_teleoperation.rst`.

## 3. Verify the runtime

From the **sample repository root**, run:

```bash
git -C "$ISAACLAB_ROOT" rev-parse HEAD
"$PYTHON" - <<'PY'
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

lock = json.loads(Path('src/cable_robot_sample/runtime-lock.json').read_text())
assert sys.version_info[:2] == (3, 12), sys.version
for name, expected in lock['packages'].items():
    actual = metadata.version(name)
    assert actual == expected, f'{name}: expected {expected}, found {actual}'
distribution = metadata.distribution('newton')
for relative, expected in lock['newton_source_sha256'].items():
    path = distribution.locate_file('newton/' + relative)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, str(path)
print('Runtime versions and Newton source fingerprints match.')
PY
```

The commit printed by Git must match step 2. The source fingerprints matter because the force observer uses private VBD interfaces. Do not edit the expected hashes just to bypass a mismatch. `--no-forces` disables the observer, but does not establish compatibility with an otherwise different SDK.

## 4. Configure assets and launch

Return to [README: Setup](../README.md#setup) to extract the two prepared bundles, configure the external rack and set `PYTHONPATH`, `ASSETS` and `ROBOT`. After personally reviewing and accepting the installed Isaac Sim/Kit license, set:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
"$PYTHON" -m cable_robot_sample run --assets "$ASSETS" --robot "$ROBOT" \
  --headless --steps 120 --reset-at 60 --output "$PWD/runs/check01"
```

Use a new output directory. A completed check writes `run-result.json` with `status: completed` and `control_steps: 120`. This verifies short scene execution, not headset input; continue with [Teleoperation](teleoperation.md) when the target headset is available.

## Common setup issues

| Symptom | Check |
| --- | --- |
| Missing module or incompatible SDK API | Use the SDK's Python; verify the commit, packages and source fingerprints above. |
| Missing rack or unresolved USD dependency | Point `ASSETS` at the extracted folder containing `manifest.json`, run `configure-rack`, then `validate-assets --usd-dependencies`. |
| Output directory already exists | Choose a new session name; the sample does not overwrite recordings. |
| Video export cannot find FFmpeg | Make `ffmpeg` available on `PATH` in the export terminal. |

For a native process crash, retain the console log and session directory. A successful retry alone does not establish that the cause is resolved.
