from dataclasses import dataclass

import torch

from .control import grip_events
from .observations import EPISODE_STEPS


@dataclass(frozen=True)
class TaskMetrics:
    depth: torch.Tensor
    lateral: torch.Tensor
    angle_deg: torch.Tensor
    position_error: torch.Tensor
    corridor: torch.Tensor
    backstop: torch.Tensor
    anchor: torch.Tensor
    gap: torch.Tensor
    drift: torch.Tensor
    rotation_error: torch.Tensor


@dataclass(frozen=True)
class TaskEvents:
    success: torch.Tensor
    grip_loss: torch.Tensor
    outside: torch.Tensor
    deadline: torch.Tensor
    physics_failure: torch.Tensor
    unsupported_steps: torch.Tensor
    escape_steps: torch.Tensor
    success_hold: torch.Tensor

    @property
    def done(self):
        return self.success | self.grip_loss | self.outside | self.deadline | self.physics_failure

    @property
    def terminal_valid(self):
        return self.done & ~self.physics_failure

    @property
    def time_outs(self):
        return torch.zeros_like(self.done)


def evaluate_termination(metrics, supported, physics_failed, episode_steps,
                         unsupported_steps, escape_steps, success_hold):
    """Evaluate task termination after physics recovery and contact sampling.

    episode_steps includes the current action. Counters belong to each world;
    reset them only after capturing terminal observations and event rewards.
    """
    lost, unsupported, escaped = grip_events(
        metrics.gap, metrics.drift, supported.any(-1), unsupported_steps, escape_steps)
    lost &= ~physics_failed
    outside = (metrics.position_error > .15) & ~physics_failed & ~lost
    inside = (
        (metrics.depth >= .009) & (metrics.lateral < .002) & (metrics.angle_deg < 5)
        & (metrics.corridor >= -.0005) & (metrics.backstop >= -.0005)
        & (metrics.anchor <= .002) & (metrics.gap <= .01) & (metrics.drift < .03)
        & (metrics.rotation_error < 30) & ~physics_failed & ~lost & ~outside
    )
    hold = torch.where(inside, success_hold + 1, 0)
    success = hold >= 30
    deadline = (episode_steps >= EPISODE_STEPS) & ~success & ~lost & ~outside & ~physics_failed
    return TaskEvents(success, lost, outside, deadline, physics_failed.clone(), unsupported, escaped, hold)
