import numpy as np
import torch
import warp as wp

from isaaclab_newton.physics import NewtonManager
from cable_robot_sample.teleop.environment import CableTeleopEnv, make_config as original_config


def make_config():
    cfg = original_config()
    if getattr(cfg, 'isaac_teleop', None) is not None:
        from isaacteleop.teleop_session_manager import RetargetingExecutionConfig
        cfg.isaac_teleop.retargeting_execution = RetargetingExecutionConfig(mode='sync')
    return cfg


class ResettableCableEnv(CableTeleopEnv):
    def __init__(self, *args, **kwargs):
        self._cable_reset_snapshot = None
        super().__init__(*args, **kwargs)

    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        env_ids = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        for articulation in self.scene.articulations.values():
            articulation.set_joint_position_target_index(
                target=articulation.data.default_joint_pos.torch.clone(), env_ids=env_ids)
            articulation.set_joint_velocity_target_index(
                target=articulation.data.default_joint_vel.torch.clone(), env_ids=env_ids)
        self.scene.write_data_to_sim()
        state = NewtonManager.get_state_0()
        control = NewtonManager.get_control()
        if self._cable_reset_snapshot is None:
            self._cable_reset_snapshot = {
                'state': {name: getattr(state, name).numpy().copy()
                          for name in ('body_q', 'joint_q', 'joint_qd')},
                'control': {name: value.numpy().copy() for name, value in vars(control).items()
                            if isinstance(value, wp.array)},
            }
        snapshot = self._cable_reset_snapshot
        for target in (NewtonManager.get_state_0(), NewtonManager.get_state_1()):
            if target is None:
                continue
            for name, value in snapshot['state'].items():
                getattr(target, name).assign(value)
            target.body_qd.zero_()
            target.body_f.zero_()
            if getattr(target, 'body_parent_f', None) is not None:
                target.body_parent_f.zero_()
        for name, value in snapshot['control'].items():
            getattr(control, name).assign(value)
        solver = NewtonManager._solver
        solver.reset(state, world_mask=None, flags=0)
        pipeline = NewtonManager._collision_pipeline
        if pipeline is not None:
            pipeline.reset_contact_matching()
        NewtonManager.invalidate_body_state()
        if NewtonManager._fk_reset_mask is not None:
            NewtonManager._fk_reset_mask.zero_()
        if NewtonManager._world_reset_mask is not None:
            NewtonManager._world_reset_mask.zero_()
        if hasattr(solver, 'record_cursor'):
            solver.record_cursor.zero_()
        self.scene.update(self.physics_dt)
        self._cable_appearance.update(state.body_q.numpy())
        pose_error = float(np.max(np.abs(state.body_q.numpy() - snapshot['state']['body_q'])))
        velocity = float(np.max(np.abs(state.body_qd.numpy())))
        if not np.isfinite([pose_error, velocity]).all() or pose_error > 1e-6 or velocity > 1e-6:
            raise RuntimeError(f'Full cable reset failed: pose_error={pose_error}, velocity={velocity}')
        self._write_event('full_reset_verified', max_pose_component_error=pose_error,
                          max_velocity_component=velocity, all_bodies=True, solver_history_reset=True)
        self.obs_buf = self.observation_manager.compute(update_history=False)
        return self.obs_buf, result[1]
