import newton
import numpy as np
import torch
import warp as wp

from isaaclab_newton.physics import NewtonManager

from .reference import map_worlds


class BatchedHeldState:
    """Fixed-model, per-world cold reset inside Isaac Lab + Newton.

    The caller chooses IDs and implements training admission/retry policy.
    No state is silently substituted, and no solver warm cache is cloned.
    """

    def __init__(self, base, reference, bank):
        self.base, self.reference, self.bank = base, reference, bank
        self.device = base.device
        model = NewtonManager.get_model()
        solver = NewtonManager._solver
        if not isinstance(solver, newton.solvers.SolverVBD):
            raise TypeError('Held-state restoration requires the all-VBD solver')
        arrays = {name: getattr(model, name).numpy() for name in (
            'body_world', 'body_mass', 'body_inertia', 'body_com', 'joint_type', 'joint_q_start', 'joint_qd_start')}
        arrays.update(joint_coord_count=model.joint_coord_count, joint_dof_count=model.joint_dof_count)
        self.mapping = map_worlds(reference, model.body_label, model.joint_label, arrays,
                                  base.num_envs, int(newton.JointType.FREE))
        self.body_matrix = torch.as_tensor(self.mapping['bodies'], device=self.device, dtype=torch.long)
        self.q_matrix = torch.as_tensor(self.mapping['coordinates'], device=self.device, dtype=torch.long)
        self.qd_matrix = torch.as_tensor(self.mapping['velocities'], device=self.device, dtype=torch.long)
        for name, matrix, source in (
            ('body_q', self.body_matrix, reference.rest_poses),
            ('joint_q', self.q_matrix, reference.rest_coordinates),
        ):
            value = torch.tensor(source, device=self.device).expand(base.num_envs, *source.shape).clone()
            if name == 'body_q':
                value[:, :, :3] += base.scene.env_origins[:, None, :]
            wp.to_torch(getattr(model, name))[matrix] = value
        wp.to_torch(solver.joint_rest_angle)[self.qd_matrix] = torch.tensor(reference.rest_angles, device=self.device)
        self.targets = torch.tensor(reference.targets, device=self.device).repeat(base.num_envs, 1)
        self.initial_ids = torch.full((base.num_envs,), -1, device=self.device, dtype=torch.long)

    def restore(self, worlds, initial_ids, *, split):
        selected = torch.as_tensor(worlds, device=self.device)
        if selected.ndim != 1 or selected.dtype not in (torch.int32, torch.int64):
            raise ValueError('World IDs must be a one-dimensional integer tensor')
        selected = selected.long()
        if len(selected) == 0 or selected.unique().numel() != selected.numel():
            raise ValueError('Reset requires distinct nonempty world IDs')
        if bool(((selected < 0) | (selected >= self.base.num_envs)).any()):
            raise ValueError('World ID outside scene')
        ids = np.asarray(initial_ids)
        values = self.bank.select(ids, split=split)
        if len(ids) != len(selected):
            raise ValueError('One initial state is required per selected world')
        values = {name: torch.tensor(value, device=self.device) for name, value in values.items()}
        self.base._reset_idx(selected)
        for state in (NewtonManager.get_state_0(), NewtonManager.get_state_1()):
            for name in ('body_q', 'body_qd', 'body_f', 'joint_q', 'joint_qd'):
                matrix = self.body_matrix if name.startswith('body_') else self.q_matrix if name == 'joint_q' else self.qd_matrix
                value = values[name].clone()
                if name == 'body_q':
                    value[:, :, :3] += self.base.scene.env_origins[selected, None, :]
                wp.to_torch(getattr(state, name))[matrix[selected]] = value
        mask = torch.zeros(self.base.num_envs + 1, dtype=torch.bool, device=self.device)
        mask[selected] = True
        world_mask = wp.from_torch(mask, dtype=wp.bool)
        NewtonManager._solver.reset(NewtonManager.get_state_0(), world_mask=world_mask, flags=0)
        NewtonManager._collision_pipeline.reset_contact_matching(world_mask=world_mask)
        NewtonManager.invalidate_body_state(wp.from_torch(selected.to(torch.int32), dtype=wp.int32))
        NewtonManager._collision_pipeline.collide(NewtonManager.get_state_0(), NewtonManager.get_contacts())
        self.base.scene.update(0.0)
        self.targets[selected] = values['targets']
        self.initial_ids[selected] = torch.as_tensor(ids, device=self.device, dtype=torch.long)

    def capture(self, buffer=0):
        state = NewtonManager.get_state_0() if buffer == 0 else NewtonManager.get_state_1()
        result = {}
        for name in ('body_q', 'body_qd', 'body_f', 'joint_q', 'joint_qd'):
            matrix = self.body_matrix if name.startswith('body_') else self.q_matrix if name == 'joint_q' else self.qd_matrix
            result[name] = wp.to_torch(getattr(state, name))[matrix].clone()
        result['body_q'][:, :, :3] -= self.base.scene.env_origins[:, None, :]
        return result
