from dataclasses import dataclass

import torch


AUXILIARY_NAMES = ('time', 'amplitude', 'slew', 'grip_drift', 'grip_rotation', 'cable_arm_contact')
HORIZON_SECONDS = 40.0
MAX_AUXILIARY_PER_SECOND = .05 + .005 + .005 + .05 + .05 + .025
AUXILIARY_SCALE = .1 / (HORIZON_SECONDS * MAX_AUXILIARY_PER_SECOND)


@dataclass(frozen=True)
class RiskConfig:
    control_dt: float = 1 / 60
    drift_deadzone_m: float = .003
    drift_full_m: float = .015
    rotation_deadzone_deg: float = 5.0
    rotation_full_deg: float = 20.0
    contact_deadzone_n: float = 1.0
    contact_full_n: float = 10.0
    contact_persistence_s: float = .1
    drift_cost_per_s: float = .05
    rotation_cost_per_s: float = .05
    contact_cost_per_s: float = .025


def ramp(value, lower, upper):
    if upper <= lower:
        raise ValueError('Risk ramp requires upper > lower')
    scaled = ((value - lower) / (upper - lower)).clamp(0, 1)
    return scaled.square() * (3 - 2 * scaled)


def risk_costs(drift, rotation, arm_contact_force, contact_age, valid, config):
    values = {
        'grip_drift': config.drift_cost_per_s * ramp(drift, config.drift_deadzone_m, config.drift_full_m),
        'grip_rotation': config.rotation_cost_per_s * ramp(rotation, config.rotation_deadzone_deg, config.rotation_full_deg),
        'cable_arm_contact': config.contact_cost_per_s * ramp(arm_contact_force, config.contact_deadzone_n, config.contact_full_n)
                             * (contact_age >= config.contact_persistence_s),
    }
    return {name: torch.where(valid, value * config.control_dt, 0) for name, value in values.items()}


def objective_reward(previous_phi, next_phi, done, raw_parts):
    """Combine terminal events, potential shaping and budgeted auxiliary penalties."""
    if set(raw_parts) != {'event', 'shaping', *AUXILIARY_NAMES}:
        raise ValueError('Unexpected reward components; re-audit the auxiliary bound')
    auxiliaries = {name: raw_parts[name] * AUXILIARY_SCALE for name in AUXILIARY_NAMES}
    parts = dict(event=raw_parts['event'],
                 shaping=2 * (torch.where(done, torch.zeros_like(next_phi), next_phi) - previous_phi),
                 **auxiliaries)
    return sum(parts.values()), parts


def transition_reward(previous_phi, next_phi, events, increment, previous_increment,
                      drift, rotation, arm_contact_force, contact_age):
    """Compute the final D objective before resetting any completed world."""
    costs = risk_costs(drift, rotation, arm_contact_force, contact_age,
                       ~events.physics_failure, RiskConfig())
    raw_parts = {
        'event': 20 * events.success.float() - 5 * (events.done & ~events.success).float(),
        'time': torch.full_like(previous_phi, -.05 * (1 / 60)),
        'amplitude': -.005 * (1 / 60) * (increment / .002).square().mean(-1).clamp(max=1),
        'slew': -.005 * (1 / 60) * ((increment - previous_increment) / .0005).square().mean(-1).clamp(max=1),
        'shaping': torch.zeros_like(previous_phi),
        **{name: -cost for name, cost in costs.items()},
    }
    raw_parts = {name: value.float() for name, value in raw_parts.items()}
    reward, parts = objective_reward(previous_phi, next_phi, events.done, raw_parts)
    if not bool(torch.isfinite(reward).all()):
        raise RuntimeError('Nonfinite transition reward')
    return reward, parts
