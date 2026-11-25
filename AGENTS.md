# Instructions for coding agents

- Read README.md and docs/architecture.md before editing; see docs/assets.md for scene inputs and asset permissions.
- Keep the sample focused on scene interaction, teleoperation, recording/replay and the near-port RL task. Historical learning variants, campaign tools and engineering validation reports belong outside this repository.
- Maintain one implementation per feature. Do not add dated variants, copied SDKs, historical campaigns or machine-specific wrappers.
- Run physics through Isaac Lab + Newton. Do not replace the validation environment with standalone Newton.
- Keep SDK versions pinned. Do not update dependencies, drivers or system Python to make a test pass without approval.
- Do not start training, change rewards/termination rules, enable more GPUs or publish artifacts without a confirmed task plan.
- Asset files and recordings stay outside the source tree. License-permitted prepared assets may accompany a release as a separate bundle with provenance and required notices. Check rights separately; a code license does not license its example meshes.
- Never add credentials, internal addresses, customer details or personal paths to this repository.
- Use new output directories. Preserve failed runs and evidence outside the source tree.
- Tests distinguish import checks, physics stepping, reset, recording/replay and headset interaction. Report exactly what was executed.
- Joint reactions are an experimental, version-sensitive VBD observer. Do not claim calibrated force sensing, hard actuator force saturation or guaranteed nonpenetration.
- No Git commit, branch creation, public upload or license selection is implied by a code cleanup request.
