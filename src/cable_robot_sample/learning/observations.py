from dataclasses import dataclass
import math

import torch

from .control import relative_quaternion


CABLE_BODY_NAMES = (
    'sc_plug_link', 'link_8', 'link_7', 'link_6', 'link_5', 'link_4',
    'link_3', 'link_2', 'link_1', 'link_0', 'lc_plug_link',
    'sfp_module_link', 'sfp_tip_link',
)
ACTOR_DIM = 234
CRITIC_DIM = 248
EPISODE_STEPS = 2400


@dataclass(frozen=True)
class ActorState:
    joint_positions: torch.Tensor
    joint_velocities: torch.Tensor
    cable_poses: torch.Tensor
    cable_velocities: torch.Tensor
    entrance: torch.Tensor
    front_delta: torch.Tensor
    goal_rotation: torch.Tensor
    relative_grip: torch.Tensor
    targets: torch.Tensor
    previous_action: torch.Tensor
    previous_increment: torch.Tensor
    relative_sfp: torch.Tensor
    relative_reference: torch.Tensor
    rotation_sfp: torch.Tensor
    rotation_reference: torch.Tensor
    contact_age: torch.Tensor
    unsupported_steps: torch.Tensor
    escape_steps: torch.Tensor
    success_hold: torch.Tensor
    episode_steps: torch.Tensor
    corridor: torch.Tensor
    backstop: torch.Tensor


def actor_observation(state: ActorState):
    """Assemble policy inputs from world-frame poses and xyzw rotations.

    Cable bodies must be in CABLE_BODY_NAMES order, not numeric rod order.
    relative_grip refers to lc_plug_link; relative_sfp refers to sfp_module_link.
    Both are expressed in the hand frame. This function does not sample physics.
    """
    batch = state.joint_positions.shape[0]
    widths = {
        'joint_positions': (9,), 'joint_velocities': (9,),
        'cable_poses': (13, 7), 'cable_velocities': (13, 6),
        'entrance': (3,), 'front_delta': (3,), 'goal_rotation': (4,),
        'relative_grip': (3,), 'targets': (9,), 'previous_action': (7,),
        'previous_increment': (7,), 'relative_sfp': (3,),
        'relative_reference': (3,), 'rotation_sfp': (4,),
        'rotation_reference': (4,), 'contact_age': (2,),
        'unsupported_steps': (), 'escape_steps': (), 'success_hold': (),
        'episode_steps': (), 'corridor': (), 'backstop': (),
    }
    for name, width in widths.items():
        if getattr(state, name).shape != (batch, *width):
            raise ValueError(f'Invalid actor input shape: {name}')
    poses = state.cable_poses.clone()
    poses[:, :, :3] -= state.entrance[:, None, :]
    observation = torch.cat((
        state.joint_positions / math.pi, state.joint_velocities / 2,
        poses.flatten(1), state.cable_velocities.flatten(1) / 2,
        state.front_delta, state.goal_rotation, state.relative_grip,
        state.targets[:, :7] - state.joint_positions[:, :7],
        state.previous_action, state.previous_increment / .002,
    ), -1)
    observation[:, 18:109].reshape(batch, 13, 7)[:, :, :3] /= .5
    velocities = observation[:, 109:187].reshape(batch, 13, 6)
    velocities[:, :, :3] *= 2
    velocities[:, :, 3:] *= .4
    observation[:, 187:190] /= .05
    observation[:, 194:197] /= .1
    observation[:, 197:204] /= .04
    opening = state.joint_positions[:, 7:].sum(-1) / .08
    additional = torch.cat((
        (state.relative_sfp - state.relative_reference) / .03,
        relative_quaternion(state.rotation_reference, state.rotation_sfp),
        state.contact_age / 60, opening[:, None],
        state.unsupported_steps[:, None] / 15, state.escape_steps[:, None] / 15,
        state.success_hold[:, None] / 30,
        1 - state.episode_steps[:, None] / EPISODE_STEPS,
        state.corridor[:, None].clamp(-.05, .05) / .01,
        state.backstop[:, None] / .06,
    ), -1)
    result = torch.cat((observation, additional), -1).float()
    if result.shape != (batch, ACTOR_DIM) or not bool(torch.isfinite(result).all()):
        raise ValueError('Invalid actor observation')
    return result


def risk_features(relative, previous_relative, rotation, previous_rotation,
                  reference_relative, reference_rotation, force_groups, support, contact_age):
    """Return the 14 privileged features, using last-substep contact totals.

    Force columns: rod-arm, rod-hand, head-static, arm-static, rod-static.
    These contact magnitudes are not cable tension measurements.
    """
    angle_step = 2 * torch.acos((rotation * previous_rotation).sum(-1).abs().clamp(0, 1)) * (180 / torch.pi)
    drift = torch.linalg.vector_norm(relative - reference_relative, dim=-1)
    angle = 2 * torch.acos((rotation * reference_rotation).sum(-1).abs().clamp(0, 1)) * (180 / torch.pi)
    result = torch.cat((
        (relative - previous_relative) / ((1 / 60) * .1),
        (angle_step / ((1 / 60) * 90))[:, None],
        torch.log1p(force_groups / 10), torch.log1p(support / 10),
        contact_age[:, None].clamp(max=5), drift[:, None] / .01, angle[:, None] / 10,
    ), -1)
    if result.shape != (relative.shape[0], 14) or not bool(torch.isfinite(result).all()):
        raise ValueError('Invalid critic risk features')
    return result


def critic_observation(policy, features):
    if policy.ndim != 2 or policy.shape[1] != ACTOR_DIM or features.shape != (policy.shape[0], 14):
        raise ValueError('Invalid critic inputs')
    result = torch.cat((policy, features), -1)
    if not bool(torch.isfinite(result).all()):
        raise ValueError('Invalid critic observation')
    return result


def terminal_observation(current, previous, physics_failed):
    """Call before task reset, after failed worlds have been recovered.

    A physics-failure fallback is finite but not a valid terminal measurement.
    The caller must mark it invalid and must not bootstrap any terminal state.
    """
    result = current.clone()
    result[physics_failed] = previous[physics_failed]
    if not bool(torch.isfinite(result).all()):
        raise ValueError('Nonfinite terminal observation')
    return result
