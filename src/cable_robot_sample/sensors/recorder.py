import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import warp as wp
from newton.solvers import SolverVBD
from isaaclab_newton.physics import NewtonManager

import cable_robot_sample.sensors.joint_wrench as core
from cable_robot_sample.physics.vbd import NewtonAllVBDManager


CAPACITY = 32
SENSORS = []
SELECTED_LABELS = []


@wp.kernel
def sample_joints(selected: wp.array[wp.int32], children: wp.array[wp.int32],
                  child_frames: wp.array[wp.transform], com: wp.array[wp.vec3],
                  poses: wp.array[wp.transform], raw: wp.array[wp.spatial_vector],
                  signs: wp.array[float], cursor: wp.array[wp.int32],
                  local_samples: wp.array2d[wp.spatial_vector],
                  world_samples: wp.array2d[wp.spatial_vector]):
    index = wp.tid()
    slot = cursor[0]
    if slot < 32:
        joint = selected[index]
        body = children[joint]
        pose = poses[body]
        frame = pose * child_frames[joint]
        force = wp.vec3(raw[body][0], raw[body][1], raw[body][2])
        torque = wp.vec3(raw[body][3], raw[body][4], raw[body][5])
        torque += wp.cross(wp.transform_point(pose, com[body]) - wp.transform_get_translation(frame), force)
        force *= signs[index]
        torque *= signs[index]
        inverse = wp.quat_inverse(wp.transform_get_rotation(frame))
        local_force = wp.quat_rotate(inverse, force)
        local_torque = wp.quat_rotate(inverse, torque)
        local_samples[slot, index] = wp.spatial_vector(local_force[0], local_force[1], local_force[2], local_torque[0], local_torque[1], local_torque[2])
        world_samples[slot, index] = wp.spatial_vector(force[0], force[1], force[2], torque[0], torque[1], torque[2])


@wp.kernel
def advance(cursor: wp.array[wp.int32]):
    cursor[0] += 1


class MultiSensorVBD(core.ProbeVBD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.record_cursor = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.record_local = wp.zeros((CAPACITY, len(SENSORS)), dtype=wp.spatial_vector, device=self.device)
        self.record_world = wp.zeros_like(self.record_local)
        self.record_signs = wp.array([sensor[3] for sensor in SENSORS], dtype=float, device=self.device)

    def _finalize_rigid_bodies(self, state_in, state_out, dt):
        super()._finalize_rigid_bodies(state_in, state_out, dt)
        wp.launch(sample_joints, dim=len(SENSORS), inputs=[self.probe_selected, self.model.joint_child,
            self.model.joint_X_c, self.model.body_com, state_out.body_q, self.probe_body_parent_f,
            self.record_signs, self.record_cursor, self.record_local, self.record_world], device=self.device)
        wp.launch(advance, dim=1, inputs=[self.record_cursor], device=self.device)


def create_solver(cls, model, solver_cfg):
    selected = []
    for name, path, suffix, sign in SENSORS:
        matches = [index for index, label in enumerate(model.joint_label) if path in label and label.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f'Expected one {name} joint, found {len(matches)}')
        selected.extend(matches)
    if len(set(model.joint_child.numpy()[selected].tolist())) != len(SENSORS):
        raise ValueError('Sensor child bodies must be distinct')
    core.JOINTS[:] = selected
    SELECTED_LABELS[:] = [model.joint_label[index] for index in selected]
    return MultiSensorVBD(model, **cls._filter_solver_kwargs(SolverVBD, solver_cfg))


def install(output):
    manifest = json.loads((Path(os.environ['CABLE_TELEOP_ASSETS']) / 'manifest.json').read_text())
    SENSORS[:] = [('wrist', '/Robot/', '/fr3_hand_joint', 1.0)] + [
        (f'cable{index}_fixed', f'/Cable{index}/', '/joint_21_fixedJoint2', -1.0)
        for index in range(1, len(manifest['cables']) + 1)
    ]
    original = NewtonAllVBDManager.__dict__.get('_create_solver')
    core.OUTPUT = Path(output) / 'force_kernel'
    core.OUTPUT.mkdir(parents=True, exist_ok=False)
    provenance = core.build_kernel()
    NewtonManager.request_extended_state_attribute('body_parent_f')
    NewtonAllVBDManager._create_solver = classmethod(create_solver)
    return original, provenance


def restore(original):
    if original is None:
        delattr(NewtonAllVBDManager, '_create_solver')
    else:
        NewtonAllVBDManager._create_solver = original


class ForceRecorder:
    def __init__(self, output, provenance):
        self.output = Path(output) / 'forces'
        self.output.mkdir(exist_ok=False)
        self.segment = 0
        self.physics_step = 0
        self.chunk = 0
        self.pending = []
        self.stream = (self.output / 'summary.csv').open('x', newline='', buffering=1)
        self.writer = csv.writer(self.stream)
        header = ['segment', 'control_step', 'sim_time_s', 'wall_epoch_s', 'valid']
        for name, path, suffix, sign in SENSORS:
            header.extend(f'{name}_{component}_mean' for component in ('Fx_N', 'Fy_N', 'Fz_N', 'Tx_Nm', 'Ty_Nm', 'Tz_Nm'))
            header.extend([f'{name}_force_norm_max_N', f'{name}_torque_norm_max_Nm'])
        self.writer.writerow(header)
        metadata = dict(schema=1, physics_hz=960, control_hz=60, provenance=provenance,
            sensors=[dict(name=sensor[0], joint=label, sign=sensor[3]) for sensor, label in zip(SENSORS, SELECTED_LABELS)],
            components=['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz'], force_units='N', torque_units='N m',
            local_frame='child joint frame; torque about child joint origin',
            world_frame='world axes; torque about the same child joint origin',
            sign_convention='wrist: parent on child; fixed cable ends: child on parent',
            csv_frame='wrist local, fixed ends world', gravity_or_inertia_compensated=False,
            filter='none', timing='raw arrays: each physical substep end; sim time restarts on reset',
            qualification='VBD simulated joint reaction; not calibrated hardware or isolated insertion force')
        manifest = json.loads((Path(os.environ['CABLE_TELEOP_ASSETS']) / 'manifest.json').read_text())
        for sensor, cable in zip(metadata['sensors'][1:], manifest['cables'], strict=True):
            sensor.update(cable_name=cable['name'], identifier=cable['identifier'], operable=cable['operable'],
                          jacket_color_linear=cable['jacket_color'], connector_color=cable['connector_color'])
        (self.output / 'metadata.json').write_text(json.dumps(metadata, indent=2))

    def before_step(self):
        NewtonManager._solver.record_cursor.zero_()

    def after_step(self, control_step):
        solver = NewtonManager._solver
        count = int(solver.record_cursor.numpy()[0])
        if count != 16:
            raise RuntimeError(f'Expected 16 force samples, found {count}')
        local = solver.record_local.numpy()[:count].copy()
        world = solver.record_world.numpy()[:count].copy()
        times = (self.physics_step + np.arange(1, count + 1)) / 960.0
        self.physics_step += count
        epoch = time.time()
        valid = bool(np.isfinite(local).all() and np.isfinite(world).all())
        self.pending.append((control_step, epoch, times, local, world))
        row = [self.segment, control_step, times[-1], epoch, int(valid)]
        for index in range(len(SENSORS)):
            values = local[:, index] if index == 0 else world[:, index]
            row.extend(values.mean(axis=0).tolist())
            row.extend([float(np.linalg.norm(values[:, :3], axis=1).max()), float(np.linalg.norm(values[:, 3:], axis=1).max())])
        self.writer.writerow(row)
        if len(self.pending) >= 120:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        path = self.output / f'segment_{self.segment:04d}_chunk_{self.chunk:06d}.npz'
        with path.open('xb') as stream:
            np.savez(stream, control_step=np.array([entry[0] for entry in self.pending]),
                wall_epoch_s=np.array([entry[1] for entry in self.pending]),
                sim_time_s=np.stack([entry[2] for entry in self.pending]),
                local=np.stack([entry[3] for entry in self.pending]),
                world=np.stack([entry[4] for entry in self.pending]))
        self.pending.clear()
        self.chunk += 1

    def reset(self):
        self.flush()
        self.segment += 1
        self.physics_step = 0

    def close(self):
        try:
            self.flush()
        finally:
            self.stream.close()
