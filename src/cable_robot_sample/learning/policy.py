from pathlib import Path

import torch
from torch import nn


ACTOR_OBSERVATIONS = 234
CRITIC_OBSERVATIONS = 248
ACTIONS = 7


class InsertionPolicy(nn.Module):
    """Insertion actor; joint-control projection is applied by the environment."""

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(ACTOR_OBSERVATIONS, 256), nn.ELU(),
                                 nn.Linear(256, 256), nn.ELU(), nn.Linear(256, ACTIONS))

    def forward(self, observations):
        if observations.ndim != 2 or observations.shape[-1] != ACTOR_OBSERVATIONS:
            raise ValueError('Expected a batch of 234 ordered policy observations')
        if not bool(torch.isfinite(observations).all()):
            raise ValueError('Nonfinite policy observations')
        actions = self.mlp(observations)
        if not bool(torch.isfinite(actions).all()):
            raise ValueError('Nonfinite policy actions')
        return actions


def load_policy(path: str | Path, device='cpu'):
    """Load a trusted project checkpoint on CPU before moving the frozen actor."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    state = checkpoint['actor_state_dict']
    policy = InsertionPolicy()
    expected = policy.state_dict()
    if set(state) != {*expected, 'distribution.log_std_param'}:
        raise ValueError('Unexpected actor schema or observation normalization')
    for name, tensor in state.items():
        shape = expected[name].shape if name in expected else (ACTIONS,)
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != shape
                or tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all())):
            raise ValueError(f'Invalid actor tensor: {name}')
    policy.load_state_dict({name: state[name] for name in expected}, strict=True)
    return policy.to(device).eval().requires_grad_(False)
