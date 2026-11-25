import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom

from isaaclab_newton.physics import NewtonManager


KINDS = {'initial': 0, 'step': 1, 'paused': 2, 'reset': 3, 'final': 4}


class TrajectoryRecorder:
    def __init__(self, env):
        import isaaclab.sim as sim

        self.env = env
        self.stage = sim.get_current_stage()
        self.root = env._cable_output / 'trajectory'
        self.root.mkdir(exist_ok=False)
        self.pending = []
        self.chunk = 0
        self.segment = env._force_recorder.segment if env._force_recorder is not None else 0
        self.sim_time = 0.0
        self.started = time.monotonic()
        self.started_epoch = time.time()
        self.last_capture = self.started
        self.last_flush = self.started
        self.last_action = np.full(env.action_manager.total_action_dim, np.nan, dtype=np.float32)
        self.index = (self.root / 'chunks.jsonl').open('x', buffering=1)
        self.camera_events = (self.root / 'camera_sources.jsonl').open('x', buffering=1)
        self.last_camera_path = None
        model = NewtonManager.get_model()
        self.labels = list(model.body_label)
        scales = []
        for label in self.labels:
            prim = self.stage.GetPrimAtPath(label)
            if not prim or not prim.IsA(UsdGeom.Xformable):
                raise RuntimeError(f'Cannot map recorded rigid body to USD: {label}')
            transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            scales.append(list(Gf.Transform(transform).GetScale()))
        self.fallback = UsdGeom.Camera.Define(self.stage, '/World/TeleopThirdPersonRecordCamera')
        camera_matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*env.cfg.viewer.eye), Gf.Vec3d(*env.cfg.viewer.lookat), Gf.Vec3d(0, 0, 1)).GetInverse()
        UsdGeom.Xformable(self.fallback.GetPrim()).MakeMatrixXform().Set(camera_matrix)
        self.fallback.GetFocalLengthAttr().Set(24.0)
        self.fallback.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
        scene_path = self.root / 'scene.usdc'
        self.stage.Flatten().Export(str(scene_path))
        cable_manifest = json.loads((Path(env.cfg.cable_assets) / 'manifest.json').read_text())
        curves = []
        for cable in cable_manifest['cables']:
            ids = [next(index for index, label in enumerate(self.labels) if label.endswith(f'/{cable["name"]}/link_{segment}')) for segment in range(cable['rod_count'])]
            curves.append({'path': '/World/CableAppearance/' + cable['name'], 'body_indices': ids,
                'first_half_rod_length_m': 0.5 * cable['segment_lengths_m'][0],
                'last_half_rod_length_m': 0.5 * cable['segment_lengths_m'][-1],
                'identifier': cable['identifier'], 'operable': cable['operable'],
                'jacket_color_linear': cable['jacket_color'], 'connector_color': cable['connector_color'],
                'radius_m': cable['jacket_radius_m']})
        info = dict(schema=2, body_labels=self.labels, body_world_scale=scales,
            joint_labels=list(model.joint_label), joint_q_start=model.joint_q_start.numpy().tolist(),
            body_q_layout='world xyz + quaternion xyzw', scene_sha256=hashlib.sha256(scene_path.read_bytes()).hexdigest(),
            step_dt=float(env.step_dt), frame_kinds=KINDS, curves=curves,
            started_epoch_s=self.started_epoch, camera='desktop third person, excluding XR/head/eye cameras; fixed fallback if unavailable',
            camera_parameters=['focalLength','horizontalAperture','verticalAperture','horizontalApertureOffset','verticalApertureOffset','near','far'],
            force_alignment='segment + global control_step; no force recomputation during replay',
            physics_resume_supported=False, source_assets=str(env.cfg.cable_assets),
            scene_assets=getattr(env.cfg, 'recording_assets', None),
            storage='atomic uncompressed NPZ chunks, at most 120 frames or about 2 wall seconds buffered')
        (self.root / 'metadata.json').write_text(json.dumps(info, indent=2))

    def camera(self):
        viewport = None
        if os.environ.get('CABLE_HEADLESS') != '1':
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
        path = str(viewport.camera_path) if viewport is not None else ''
        camera = UsdGeom.Camera(self.stage.GetPrimAtPath(path)) if path else None
        if not camera or any(word in path.lower() for word in ('xr', 'head', 'eye')):
            camera = self.fallback
            path = str(camera.GetPath())
        if path != self.last_camera_path:
            self.camera_events.write(json.dumps({'wall_s': time.monotonic() - self.started, 'camera_path': path}) + '\n')
            self.last_camera_path = path
        matrix = camera.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        clip = camera.GetClippingRangeAttr().Get()
        parameters = [camera.GetFocalLengthAttr().Get(), camera.GetHorizontalApertureAttr().Get(),
            camera.GetVerticalApertureAttr().Get(), camera.GetHorizontalApertureOffsetAttr().Get(),
            camera.GetVerticalApertureOffsetAttr().Get(), clip[0], clip[1]]
        return np.asarray(matrix, dtype=np.float64), np.asarray(parameters, dtype=np.float64), camera.GetProjectionAttr().Get() == 'orthographic'

    def capture(self, kind, action=None):
        if kind == 'step':
            self.sim_time += float(self.env.step_dt)
        if action is not None:
            self.last_action = action.detach().cpu().numpy().reshape(-1).copy()
        state = NewtonManager.get_state_0()
        control = NewtonManager.get_control()
        matrix, parameters, orthographic = self.camera()
        pose = state.body_q.numpy().copy()
        now = time.monotonic()
        self.pending.append(dict(body_q=pose, body_qd=state.body_qd.numpy().copy(),
            joint_q=state.joint_q.numpy().copy(), joint_qd=state.joint_qd.numpy().copy(), joint_target_q=control.joint_target_q.numpy().copy(),
            action=self.last_action.copy(), camera_matrix=matrix, camera_parameters=parameters,
            camera_orthographic=orthographic, sim_time_s=self.sim_time, wall_time_s=now - self.started,
            wall_epoch_s=time.time(), control_step=self.env._cable_step, kind=KINDS[kind],
            finite=bool(np.isfinite(pose).all())))
        self.last_capture = now
        if len(self.pending) >= 120 or now - self.last_flush >= 2.0:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        name = f'segment_{self.segment:04d}_chunk_{self.chunk:06d}.npz'
        temporary = self.root / (name + '.partial')
        with temporary.open('xb') as stream:
            np.savez(stream, **{key: np.stack([frame[key] for frame in self.pending]) for key in self.pending[0]})
        os.replace(temporary, self.root / name)
        self.index.write(json.dumps(dict(file=name, segment=self.segment, frames=len(self.pending),
            first_sim_s=self.pending[0]['sim_time_s'], last_sim_s=self.pending[-1]['sim_time_s'],
            first_wall_s=self.pending[0]['wall_time_s'], last_wall_s=self.pending[-1]['wall_time_s'])) + '\n')
        self.pending.clear()
        self.chunk += 1
        self.last_flush = time.monotonic()

    def new_segment(self):
        self.flush()
        self.segment = self.env._force_recorder.segment if self.env._force_recorder is not None else self.segment + 1
        self.sim_time = 0.0
        self.last_action.fill(np.nan)

    def close(self):
        try:
            self.flush()
        finally:
            self.index.close()
            self.camera_events.close()
