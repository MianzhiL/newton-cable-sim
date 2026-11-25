import torch


def project_control(actions, positions, targets, previous_increment, lower, upper):
    lower, upper = lower.to(targets.dtype), upper.to(targets.dtype)
    actions, positions = actions.to(targets.dtype), positions.to(targets.dtype)
    desired = targets + 0.002 * actions.clamp(-1, 1)
    desired = torch.maximum(torch.minimum(desired, positions + 0.04), positions - 0.04)
    increment = torch.maximum(torch.minimum(desired - targets, previous_increment + 0.0005),
                              previous_increment - 0.0005).clamp(-0.002, 0.002)
    target = torch.maximum(torch.minimum(targets + increment, upper), lower)
    return target, target - targets


def task_potential(depth, lateral, angle_deg):
    error = torch.sqrt((depth - 0.01).square() + lateral.square()
                       + (0.04 * torch.sin(torch.deg2rad(angle_deg) / 2)).square())
    return -(error / 0.05).clamp(0, 1)


def relative_quaternion(parent, child):
    vector = parent[..., 3:] * child[..., :3] - child[..., 3:] * parent[..., :3]
    vector -= torch.cross(parent[..., :3], child[..., :3], dim=-1)
    scalar = parent[..., 3:] * child[..., 3:] + (parent[..., :3] * child[..., :3]).sum(-1, keepdim=True)
    result = torch.cat((vector, scalar), dim=-1)
    return torch.where(result[..., 3:] < 0, -result, result)


def grip_events(gap, drift, support, unsupported_steps, escape_steps):
    unsupported = torch.where(~support, (unsupported_steps + 1).clamp(max=15), 0)
    escaped = torch.where((gap > 0.03) | (drift > 0.08), (escape_steps + 1).clamp(max=15), 0)
    lost = ((gap > 0.01) & (unsupported >= 15)) | (escaped >= 15)
    return lost, unsupported, escaped
