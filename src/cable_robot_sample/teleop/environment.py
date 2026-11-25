import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import newton
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import NewtonManager

from cable_robot_sample.scene.runtime import CableAppearance, build_config


@configclass
class EmptyTerms:
    pass


@configclass
class ManualTerminations:
    time_out = None


def make_config():
    assets = Path(os.environ['CABLE_TELEOP_ASSETS']).resolve()
    args = SimpleNamespace(assets=assets, no_render=os.environ.get('CABLE_HEADLESS') == '1', ik_damping=0.5, ik_update_rate='control')
    base = build_config(args)
    cfg = ManagerBasedRLEnvCfg(scene=base.scene, sim=base.sim, decimation=base.decimation, actions=base.actions, observations=base.observations, events=base.events, viewer=base.viewer, rewards=EmptyTerms(), terminations=ManualTerminations(), episode_length_s=1e9, ui_window_class_type=None)
    cfg.sim.device = os.environ.get('CABLE_DEVICE', 'cuda:0')
    cfg.cable_assets = str(assets)
    if os.environ.get('CABLE_XR_ENABLED', '0') != '1':
        return cfg
    from isaaclab_teleop import IsaacTeleopCfg
    from isaaclab_teleop.xr_cfg import XrCfg
    from isaaclab_tasks.contrib.stack.config.franka.stack_ik_abs_env_cfg import _build_franka_stack_pipeline

    anchor = tuple(float(value) for value in os.environ.get('CABLE_XR_ANCHOR_POS', '0,-0.9,0').split(','))
    if len(anchor) != 3 or not np.isfinite(anchor).all():
        raise ValueError('CABLE_XR_ANCHOR_POS must contain three finite comma-separated coordinates')
    yaw = np.deg2rad(float(os.environ.get('CABLE_XR_ANCHOR_YAW_DEG', '0')))
    if not np.isfinite(yaw):
        raise ValueError('CABLE_XR_ANCHOR_YAW_DEG must be finite')
    cfg.xr = XrCfg(anchor_pos=anchor, anchor_rot=(0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2))))
    cfg.isaac_teleop = IsaacTeleopCfg(pipeline_builder=_build_franka_stack_pipeline, sim_device=cfg.sim.device, xr_cfg=cfg.xr, target_frame_prim_path='/World/envs/env_0/Robot/Geometry/base/fr3_link0', teleoperation_active_default=False, app_name='CableMultilineFranka')
    cfg.cable_assets = str(assets)
    return cfg


class CableTeleopEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, **kwargs):
        if cfg.scene.num_envs != 1:
            raise ValueError('This teleoperation scene supports one environment only')
        self._cable_fault = False
        self._cable_step = 0
        self._cable_last_result = None
        self._cable_appearance = None
        self._cable_log = None
        self._force_recorder = None
        self._cable_output = Path(os.environ['CABLE_TELEOP_OUTPUT']).resolve()
        self._cable_output.mkdir(parents=True, exist_ok=True)
        force_enabled = os.environ.get('CABLE_RECORD_FORCES', '1') == '1'
        if force_enabled:
            from cable_robot_sample.sensors import recorder as force_recorder
            original_factory, provenance = force_recorder.install(self._cable_output)
        try:
            super().__init__(cfg, **kwargs)
        finally:
            if force_enabled:
                force_recorder.restore(original_factory)
        if force_enabled:
            self._force_recorder = force_recorder.ForceRecorder(self._cable_output, provenance)
        from cable_robot_sample.physics.vbd import straighten

        straighten()
        model = NewtonManager.get_model()
        flags = model.shape_flags.numpy()
        fingers = [index for index, label in enumerate(model.shape_label) if '/Robot/' in label and ('/fr3_leftfinger/' in label or '/fr3_rightfinger/' in label) and flags[index] & int(newton.ShapeFlags.COLLIDE_SHAPES)]
        heads = [index for index, label in enumerate(model.shape_label) if '/sfp_module_link/' in label and flags[index] & int(newton.ShapeFlags.COLLIDE_SHAPES)]
        friction = model.shape_material_mu.numpy()
        if not fingers or not heads or friction[heads[0]] <= 0 or not np.allclose(friction[heads], friction[heads[0]]):
            raise RuntimeError('Missing or inconsistent grip collision materials')
        grip_friction = float(os.environ.get('CABLE_GRIP_FRICTION', '1.0'))
        if not np.isfinite(grip_friction) or grip_friction < 0:
            raise ValueError('CABLE_GRIP_FRICTION must be finite and nonnegative')
        friction[fingers] = grip_friction ** 2 / friction[heads[0]]
        model.shape_material_mu.assign(friction)
        NewtonManager._solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        manifest = json.loads((Path(cfg.cable_assets) / 'manifest.json').read_text())
        self._cable_appearance = CableAppearance(model, manifest)
        self._cable_appearance.update(NewtonManager.get_state_0().body_q.numpy())
        self._cable_output = Path(os.environ['CABLE_TELEOP_OUTPUT']).resolve()
        self._cable_output.mkdir(parents=True, exist_ok=True)
        self._cable_log = (self._cable_output / 'events.jsonl').open('x', buffering=1)
        self._write_event('ready', assets=cfg.cable_assets, input='official Isaac Teleop Franka pipeline', physics_hz=960, control_hz=60, ik_damping=0.5, arm_pd=[1500, 80], headset_validated=False, force_recording=force_enabled,
                          effective_grip_friction=grip_friction, finger_shape_mu=float(friction[fingers[0]]), connector_shape_mu=float(friction[heads[0]]))

    def _write_event(self, kind, **values):
        if self._cable_log is not None:
            self._cable_log.write(json.dumps({'event': kind, 'wall_epoch_s': time.time(), 'step': self._cable_step, **values}) + '\n')

    def reset(self, *args, **kwargs):
        if self._force_recorder is not None:
            self._force_recorder.reset()
        result = super().reset(*args, **kwargs)
        from cable_robot_sample.physics.vbd import straighten

        straighten()
        self._cable_fault = False
        self._cable_last_result = None
        if self._cable_appearance is not None:
            self._cable_appearance.update(NewtonManager.get_state_0().body_q.numpy())
        self._write_event('reset')
        return result

    def step(self, action):
        if self._cable_fault:
            self.sim.render()
            return self._cable_last_result
        if not bool(torch.isfinite(action).all()):
            raise ValueError('Nonfinite teleoperation command; no physics step performed')
        start = time.perf_counter()
        if self._force_recorder is not None:
            self._force_recorder.before_step()
        result = super().step(action)
        self._cable_last_result = result
        self._cable_step += 1
        if self._force_recorder is not None:
            self._force_recorder.after_step(self._cable_step)
        state = NewtonManager.get_state_0()
        poses, velocities = state.body_q.numpy(), state.body_qd.numpy()
        finite = bool(np.isfinite(poses).all() and np.isfinite(velocities).all())
        speed = float(np.linalg.norm(velocities[:, :3], axis=1).max()) if finite else None
        if not finite or speed > 15.0:
            self._cable_fault = True
            np.savez_compressed(self._cable_output / f'fault_{self._cable_step}.npz', poses=poses, velocities=velocities)
            self._write_event('physics_paused', finite=finite, max_speed_m_s=speed)
            print('Cable physics paused. Use official R/reset to restore the scene.', flush=True)
        elif self._cable_appearance is not None:
            self._cable_appearance.update(poses)
        if self._cable_step % 60 == 0:
            self._write_event('step', max_speed_m_s=speed, wall_step_ms=1000 * (time.perf_counter() - start))
        return result

    def close(self):
        if self._force_recorder is not None:
            self._force_recorder.close()
            self._force_recorder = None
        if self._cable_log is not None:
            self._write_event('close')
            self._cable_log.close()
            self._cable_log = None
        super().close()
