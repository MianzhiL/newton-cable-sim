import json
from pathlib import Path

import numpy as np
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

from cable_robot_sample.physics.vbd import NewtonAllVBDManager
from .contacts import contact_roles, contact_totals
from .control import relative_quaternion
from .geometry import collision_corners, grip_clearance, inverse_rotate, rotate
from .observations import CABLE_BODY_NAMES
from .reference import file_digest
from .scene import JOINT_NAMES


def task_frames(directory):
    """Resolve the insertion frame from the scene's connector and port geometry."""
    from scipy.spatial.transform import Rotation

    directory = Path(directory)
    paths = [directory / 'cable/frames.json', directory / 'cable/layout.json', directory / 'launch_arguments.json']
    frames, layout, arguments = [json.loads(path.read_text()) for path in paths]
    if not np.allclose(frames['insertion_axis'], [0, 1, 0], atol=1e-7, rtol=0):
        raise ValueError('This corridor requires insertion along +Y')
    quaternion = Rotation.from_quat(frames['sfp_rotation_xyzw'])
    if arguments.get('downward_flip', False):
        hang = Rotation.from_quat(layout.get('horizontal_hang_rotation_xyzw', layout['hang_rotation_xyzw']))
        insertion = Rotation.from_euler('x', 180, degrees=True) * hang
        if arguments.get('insertion_roll_deg', 0):
            insertion = insertion * Rotation.from_euler('y', arguments['insertion_roll_deg'], degrees=True)
        quaternion = insertion * quaternion
    elif arguments.get('insertion_roll_deg', 0):
        raise ValueError('Insertion roll requires downward flip')
    entrance = np.asarray(frames['entrance_local_m'], dtype=np.float64)
    entrance[2] += frames['scene_height_m']
    front = np.asarray(frames['front_body_m'], dtype=np.float64)
    if entrance.shape != (3,) or front.shape != (3,) or not np.isfinite([entrance, front]).all():
        raise ValueError('Invalid insertion frame')
    return entrance, quaternion.as_quat(), front, {path.relative_to(directory).as_posix(): file_digest(path) for path in paths}


class NewtonMeasurements:
    """Read near-port task measurements after a completed physics step.

    This adapter does not reset worlds, advance counters, terminate episodes or
    recover faults. Contact force collection uses the pinned VBD private API.
    """

    def __init__(self, base, snapshot, task_assets, robot):
        self.base, self.snapshot = base, snapshot
        self.device, self.num_envs = base.device, base.num_envs
        model = NewtonManager.get_model()
        labels = model.body_label
        first_labels = [labels[index] for index in snapshot.mapping['bodies'][0]]

        def bodies(name):
            matches = [index for index, label in enumerate(first_labels) if label.endswith('/' + name)]
            if len(matches) != 1:
                raise ValueError(f'Ambiguous body mapping: {name}')
            return snapshot.body_matrix[:, matches[0]]

        self.sfp, self.hand, self.head = [bodies(name) for name in ('sfp_module_link', 'fr3_hand', 'lc_plug_link')]
        self.cable = torch.stack([bodies(name) for name in CABLE_BODY_NAMES], -1)
        joints = snapshot.mapping['joints']
        names = [model.joint_label[index] for index in joints[0]]
        selected = []
        for name in JOINT_NAMES:
            matches = [index for index, label in enumerate(names) if label.endswith('/' + name)]
            if len(matches) != 1:
                raise ValueError(f'Ambiguous joint mapping: {name}')
            selected.append(matches[0])
        self.coordinates = torch.tensor(model.joint_q_start.numpy()[joints[:, selected]], device=self.device, dtype=torch.long)
        self.velocities = torch.tensor(model.joint_qd_start.numpy()[joints[:, selected]], device=self.device, dtype=torch.long)
        entrance, goal_rotation, front, self.frame_hashes = task_frames(task_assets)
        self.entrance = torch.tensor(entrance, device=self.device, dtype=torch.float32).repeat(self.num_envs, 1)
        self.entrance += base.scene.env_origins
        self.goal = self.entrance + torch.tensor([0, .01, 0], device=self.device)
        self.goal_rotation = torch.tensor(goal_rotation, device=self.device, dtype=torch.float32).repeat(self.num_envs, 1)
        self.front_offset = torch.tensor(front, device=self.device, dtype=torch.float32).repeat(self.num_envs, 1)
        self.body_world = wp.to_torch(model.body_world).long()
        self.roles = contact_roles(labels, self.device)
        self.fingers = [bodies(name) for name in ('fr3_leftfinger', 'fr3_rightfinger')]
        self.finger_side = torch.full((len(labels),), -1, device=self.device, dtype=torch.long)
        self.finger_corners = []
        for side, name in enumerate(('fr3_leftfinger', 'fr3_rightfinger')):
            self.finger_side[self.fingers[side]] = side
            self.finger_corners.append(torch.tensor(collision_corners(robot, name), device=self.device).flatten(0, 1))
        self.sfp_corners = torch.tensor(collision_corners(Path(task_assets) / 'cable/supported_cable.usda', 'sfp_module_link'), device=self.device)
        anchors = [index for index, child in enumerate(model.joint_child.numpy()) if child >= 0 and '/Cable/' in labels[child]]
        anchor_ids = torch.tensor(anchors, device=self.device)
        self.anchor_parent = wp.to_torch(model.joint_parent)[anchor_ids].long()
        self.anchor_child = wp.to_torch(model.joint_child)[anchor_ids].long()
        self.anchor_first = wp.to_torch(model.joint_X_p)[anchor_ids, :3]
        self.anchor_second = wp.to_torch(model.joint_X_c)[anchor_ids, :3]
        self.anchor_world = self.body_world[self.anchor_child]

    def contacts(self):
        state, solver, contacts = NewtonManager.get_state_0(), NewtonManager._solver, NewtonManager.get_contacts()
        first, second, _, _, forces, count = solver.collect_rigid_contact_forces(
            state.body_q, NewtonAllVBDManager.contact_previous_pose, contacts, 1 / 960)
        active = int(wp.to_torch(count)[0])
        if not 0 <= active <= min(first.shape[0], second.shape[0], forces.shape[0]):
            raise RuntimeError('Invalid collected contact count')
        support, _ = contact_totals(wp.to_torch(first)[:active].long(), wp.to_torch(second)[:active].long(),
                                   wp.to_torch(forces)[:active], self.body_world, self.finger_side, self.roles, self.num_envs)
        active = int(wp.to_torch(contacts.rigid_contact_count)[0])
        first, second, forces = solver._rigid_contact_body0, solver._rigid_contact_body1, contacts.rigid_contact_force
        if not 0 <= active <= min(first.shape[0], second.shape[0], forces.shape[0]):
            raise RuntimeError('Invalid risk contact count')
        _, groups = contact_totals(wp.to_torch(first)[:active].long(), wp.to_torch(second)[:active].long(),
                                  wp.to_torch(forces)[:active], self.body_world, self.finger_side, self.roles, self.num_envs)
        return support, groups

    def sample(self, include_contacts=True):
        state = NewtonManager.get_state_0()
        poses = wp.to_torch(state.body_q)
        sfp, hand, head = poses[self.sfp], poses[self.hand], poses[self.head]
        front = sfp[:, :3] + rotate(sfp[:, 3:], self.front_offset)
        delta = front - self.entrance
        dot = (sfp[:, 3:] * self.goal_rotation).sum(-1).abs().clamp(0, 1)
        corridor, backstop, gap = grip_clearance(poses, self.sfp, self.hand, self.entrance,
                                               self.sfp_corners, self.fingers, self.finger_corners)
        parent, child = poses[self.anchor_parent.clamp_min(0)], poses[self.anchor_child]
        first = parent[:, :3] + rotate(parent[:, 3:], self.anchor_first)
        first = torch.where((self.anchor_parent < 0)[:, None], self.anchor_first, first)
        second = child[:, :3] + rotate(child[:, 3:], self.anchor_second)
        anchor = torch.zeros(self.num_envs, device=self.device)
        anchor.scatter_reduce_(0, self.anchor_world, torch.linalg.vector_norm(first - second, dim=-1), reduce='amax', include_self=True)
        support, groups = self.contacts() if include_contacts else (
            torch.zeros((self.num_envs, 2), device=self.device),
            torch.zeros((self.num_envs, 5), device=self.device))
        return dict(joint_positions=wp.to_torch(state.joint_q)[self.coordinates],
                    joint_velocities=wp.to_torch(state.joint_qd)[self.velocities],
                    cable_poses=poses[self.cable], cable_velocities=wp.to_torch(state.body_qd)[self.cable],
                    front_delta=delta, depth=delta[:, 1], lateral=torch.linalg.vector_norm(delta[:, [0, 2]], dim=-1),
                    angle_deg=2 * torch.acos(dot) * (180 / torch.pi), position_error=torch.linalg.vector_norm(front - self.goal, dim=-1),
                    relative_grip=inverse_rotate(hand[:, 3:], head[:, :3] - hand[:, :3]),
                    relative_sfp=inverse_rotate(hand[:, 3:], sfp[:, :3] - hand[:, :3]),
                    rotation_sfp=relative_quaternion(hand[:, 3:], sfp[:, 3:]),
                    corridor=corridor, backstop=backstop, gap=gap, anchor=anchor,
                    support=support, force_groups=groups)
