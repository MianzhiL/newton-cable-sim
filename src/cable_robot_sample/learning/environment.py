from collections import deque
import json
import math

import numpy as np
import torch
from tensordict import TensorDict
import warp as wp
from isaaclab_newton.physics import NewtonManager

from .control import project_control, relative_quaternion, task_potential
from .geometry import inverse_rotate, rotate
from .measurements import NewtonMeasurements
from .observations import ActorState, EPISODE_STEPS, actor_observation, risk_features, terminal_observation
from .reward import RiskConfig, transition_reward
from .snapshot import BatchedHeldState
from .termination import TaskMetrics, evaluate_termination


class InsertionEnv:
    num_actions = 7
    max_episode_length = EPISODE_STEPS

    def __init__(self, base, reference, bank, args):
        self.base, self.bank, self.output = base, bank, args.output
        self.device, self.num_envs = base.device, base.num_envs
        self.snapshot = BatchedHeldState(base, reference, bank)
        self.measure = NewtonMeasurements(base, self.snapshot, args.task_assets, args.robot)
        self.rng = np.random.default_rng(args.seed)
        self.active_ids = list(bank.train_ids)
        self.autoreset = args.command == 'train'
        self.step_number = 0
        self.fault_window = deque()
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_return = torch.zeros(self.num_envs, device=self.device)
        self.previous_action = torch.zeros((self.num_envs, 7), device=self.device)
        self.previous_increment = torch.zeros_like(self.previous_action)
        self.contact_age = torch.full((self.num_envs, 2), 60., device=self.device)
        self.unsupported_steps = torch.zeros_like(self.episode_length_buf)
        self.escape_steps = torch.zeros_like(self.episode_length_buf)
        self.success_hold = torch.zeros_like(self.episode_length_buf)
        self.risk_age = torch.zeros_like(self.episode_return)
        self.privileged = torch.zeros((self.num_envs, 14), device=self.device)
        self.reference_angles = wp.to_torch(NewtonManager._solver.joint_rest_angle)[self.measure.velocities[:, :7]].clone()
        from pxr import Usd, UsdPhysics
        stage = Usd.Stage.Open(str(args.robot))
        joints = {prim.GetName(): UsdPhysics.RevoluteJoint(prim) for prim in stage.Traverse()
                  if prim.IsA(UsdPhysics.RevoluteJoint)}
        lower = torch.tensor([math.radians(joints[f'fr3_joint{index}'].GetLowerLimitAttr().Get()) + .01
                              for index in range(1, 8)], device=self.device)
        upper = torch.tensor([math.radians(joints[f'fr3_joint{index}'].GetUpperLimitAttr().Get()) - .01
                              for index in range(1, 8)], device=self.device)
        self.lower = torch.maximum(lower, self.reference_angles - math.pi + math.radians(10))
        self.upper = torch.minimum(upper, self.reference_angles + math.pi - math.radians(10))
        poses = torch.tensor(reference.states['body_q'], device=self.device)
        hand = poses[next(index for index, label in enumerate(reference.body_labels) if label.endswith('/fr3_hand'))]
        head = poses[next(index for index, label in enumerate(reference.body_labels) if label.endswith('/sfp_module_link'))]
        self.relative_reference = inverse_rotate(hand[3:], head[:3] - hand[:3]).repeat(self.num_envs, 1)
        self.rotation_reference = relative_quaternion(hand[3:], head[3:]).repeat(self.num_envs, 1)
        self.previous_relative = self.relative_reference.clone()
        self.previous_rotation = self.rotation_reference.clone()
        self.phi = torch.zeros_like(self.episode_return)
        self.cfg = dict(task='near-port insertion', physics_hz=960, control_hz=60, vbd_iterations=20,
                        actor_observations=234, critic_observations=248, episode_steps=EPISODE_STEPS,
                        effective_grip_friction=30., seed=args.seed, gamma=1., timeout_bootstrap=False)

    def health(self):
        model = NewtonManager.get_model()
        shared = wp.to_torch(model.body_world) < 0
        finite = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        for state in (NewtonManager.get_state_0(), NewtonManager.get_state_1()):
            for name in ('body_q', 'body_qd', 'body_f', 'joint_q', 'joint_qd'):
                matrix = self.snapshot.body_matrix if name.startswith('body_') else (
                    self.snapshot.q_matrix if name == 'joint_q' else self.snapshot.qd_matrix)
                values = wp.to_torch(getattr(state, name))
                finite &= torch.isfinite(values[matrix]).flatten(1).all(-1)
                if name.startswith('body_') and not bool(torch.isfinite(values[shared]).all()):
                    raise RuntimeError('Nonfinite shared state cannot be isolated to one environment')
        state = NewtonManager.get_state_0()
        poses = wp.to_torch(state.body_q)
        speed = torch.linalg.vector_norm(wp.to_torch(state.body_qd)[self.snapshot.body_matrix, :3], dim=-1).amax(-1)
        parent, child = poses[self.measure.anchor_parent.clamp_min(0)], poses[self.measure.anchor_child]
        first = parent[:, :3] + rotate(parent[:, 3:], self.measure.anchor_first)
        first = torch.where((self.measure.anchor_parent < 0)[:, None], self.measure.anchor_first, first)
        second = child[:, :3] + rotate(child[:, 3:], self.measure.anchor_second)
        anchor = torch.zeros_like(speed)
        anchor.scatter_reduce_(0, self.measure.anchor_world, torch.linalg.vector_norm(first - second, dim=-1),
                               reduce='amax', include_self=True)
        position = wp.to_torch(state.joint_q)[self.measure.coordinates[:, :7]]
        margin = math.pi - (position - self.reference_angles).abs().amax(-1)
        reasons = dict(nonfinite=~finite, speed=speed > 20, anchor=anchor > .02, chart=margin < math.radians(10))
        return torch.stack(list(reasons.values())).any(0), reasons

    def reset(self, selected=None, initial_ids=None, split='train'):
        selected = torch.arange(self.num_envs, device=self.device) if selected is None else torch.as_tensor(selected, device=self.device)
        selected = selected.long()
        pending = selected
        for attempt in range(3):
            ids = self.rng.choice(self.active_ids, len(pending)).tolist() if initial_ids is None else list(initial_ids)
            self.snapshot.restore(pending, ids, split=split)
            failed, reasons = self.health()
            values = self.measure.sample(include_contacts=False)
            drift = torch.linalg.vector_norm(values['relative_sfp'] - self.relative_reference, dim=-1)
            angle = torch.rad2deg(2 * torch.acos((values['rotation_sfp'] * self.rotation_reference).sum(-1).abs().clamp(0, 1)))
            rejected = failed[pending] | (drift[pending] > .003) | (angle[pending] > 5)
            if split == 'test':
                if bool(failed.any()):
                    raise RuntimeError('Evaluation initial state failed physics validation')
                if bool(rejected.any()):
                    raise RuntimeError('Evaluation initial state failed grasp validation')
                break
            if not bool(rejected.any()):
                break
            bad_ids = self.snapshot.initial_ids[pending[rejected]].cpu().tolist()
            with (self.output / 'reset-rejections.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(attempt=attempt + 1, initial_ids=bad_ids)) + '\n')
            self.active_ids = [initial for initial in self.active_ids if initial not in bad_ids]
            if not self.active_ids:
                raise RuntimeError('No admitted training states remain')
            pending = pending[rejected]
        else:
            raise RuntimeError('Initial-state reset retry budget exhausted')
        for value in (self.episode_length_buf, self.episode_return, self.previous_action, self.previous_increment,
                      self.unsupported_steps, self.escape_steps, self.success_hold, self.risk_age, self.privileged):
            value[selected] = 0
        self.contact_age[selected] = 60
        self.previous_relative[selected] = values['relative_sfp'][selected]
        self.previous_rotation[selected] = values['rotation_sfp'][selected]
        self.phi[selected] = task_potential(values['depth'], values['lateral'], values['angle_deg'])[selected]
        self.values = values
        return self.get_observations()

    def get_observations(self):
        values = self.values
        observation = actor_observation(ActorState(
            **{name: values[name] for name in ('joint_positions', 'joint_velocities', 'cable_poses', 'cable_velocities',
                                               'front_delta', 'relative_grip', 'relative_sfp', 'rotation_sfp', 'corridor', 'backstop')},
            entrance=self.measure.entrance, goal_rotation=self.measure.goal_rotation, targets=self.snapshot.targets,
            previous_action=self.previous_action, previous_increment=self.previous_increment,
            relative_reference=self.relative_reference, rotation_reference=self.rotation_reference,
            contact_age=self.contact_age, unsupported_steps=self.unsupported_steps, escape_steps=self.escape_steps,
            success_hold=self.success_hold, episode_steps=self.episode_length_buf))
        return TensorDict(dict(policy=observation, critic=torch.cat((observation, self.privileged), -1)),
                          batch_size=[self.num_envs])

    def step(self, actions):
        if actions.shape != (self.num_envs, 7) or not bool(torch.isfinite(actions).all()):
            raise RuntimeError('Invalid policy action')
        previous = self.get_observations().clone()
        old_increment = self.previous_increment.clone()
        old_phi = self.phi.clone()
        target, increment = project_control(actions, self.values['joint_positions'][:, :7],
                                           self.snapshot.targets[:, :7], old_increment, self.lower, self.upper)
        self.snapshot.targets[:, :7] = target
        self.base.step(self.snapshot.targets)
        self.step_number += 1
        self.episode_length_buf += 1
        failed, reasons = self.health()
        if bool(failed.any()):
            count = int(failed.sum())
            self.fault_window.append((self.step_number, count))
            while self.fault_window and self.fault_window[0][0] <= self.step_number - 600:
                self.fault_window.popleft()
            rate = sum(value for _, value in self.fault_window) / (min(600, self.step_number) * self.num_envs)
            with (self.output / 'physics-events.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(step=self.step_number, worlds=failed.nonzero().flatten().tolist(),
                    reasons={key: value.nonzero().flatten().tolist() for key, value in reasons.items()}, rate=rate)) + '\n')
            if count >= max(4, math.ceil(.1 * self.num_envs)) or (self.step_number >= 60 and rate > .01):
                raise RuntimeError('Systemic physics failure; inspect physics-events.jsonl')
            selected = failed.nonzero().flatten()
            ids = self.snapshot.initial_ids[selected].cpu().tolist()
            self.snapshot.restore(selected, ids, split='train' if self.autoreset else 'test')
            if bool(self.health()[0].any()):
                raise RuntimeError('Per-environment physics recovery failed')
        values = self.measure.sample()
        values['support'][failed] = 0
        values['force_groups'][failed] = 0
        supported = values['support'] > .01
        self.contact_age = torch.where(supported, 0, (self.contact_age + 1).clamp(max=60))
        drift = torch.linalg.vector_norm(values['relative_sfp'] - self.relative_reference, dim=-1)
        angle = torch.rad2deg(2 * torch.acos((values['rotation_sfp'] * self.rotation_reference).sum(-1).abs().clamp(0, 1)))
        metrics = TaskMetrics(**{name: values[name] for name in ('depth', 'lateral', 'angle_deg', 'position_error',
                                                                 'corridor', 'backstop', 'anchor', 'gap')},
                              drift=drift, rotation_error=angle)
        events = evaluate_termination(metrics, supported, failed, self.episode_length_buf,
                                      self.unsupported_steps, self.escape_steps, self.success_hold)
        self.unsupported_steps, self.escape_steps, self.success_hold = events.unsupported_steps, events.escape_steps, events.success_hold
        self.risk_age = torch.where(values['force_groups'][:, 0] > RiskConfig().contact_deadzone_n, self.risk_age + 1 / 60, 0)
        self.privileged = risk_features(values['relative_sfp'], self.previous_relative, values['rotation_sfp'],
            self.previous_rotation, self.relative_reference, self.rotation_reference,
            values['force_groups'], values['support'], self.risk_age)
        self.previous_relative, self.previous_rotation = values['relative_sfp'].clone(), values['rotation_sfp'].clone()
        next_phi = task_potential(values['depth'], values['lateral'], values['angle_deg'])
        reward, parts = transition_reward(old_phi, next_phi, events, increment, old_increment,
                                          drift, angle, values['force_groups'][:, 0], self.risk_age)
        self.values = values
        self.previous_action.copy_(actions.clamp(-1, 1))
        self.previous_increment.copy_(increment)
        self.phi.copy_(next_phi)
        self.episode_return += reward
        terminal = self.get_observations().clone()
        for key in terminal.keys():
            terminal[key] = terminal_observation(terminal[key], previous[key], failed)
        records = []
        for world in events.done.nonzero().flatten().tolist():
            reason = next(name for name in ('success', 'grip_loss', 'outside', 'deadline', 'physics_failure')
                          if bool(getattr(events, name)[world]))
            records.append(dict(world=world, initial_id=int(self.snapshot.initial_ids[world]), reason=reason,
                                steps=int(self.episode_length_buf[world]), return_total=float(self.episode_return[world]),
                                depth_m=None if bool(failed[world]) else float(values['depth'][world])))
        if records and self.autoreset:
            with (self.output / 'episodes.jsonl').open('a') as stream:
                for record in records:
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
        extras = dict(time_outs=events.time_outs, terminal_observation=terminal,
                      terminal_observation_valid=events.terminal_valid, episodes=records, reward_parts=parts)
        if self.autoreset and bool(events.done.any()):
            self.reset(events.done.nonzero().flatten())
        return self.get_observations(), reward, events.done, extras
