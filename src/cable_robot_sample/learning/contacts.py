import torch


def contact_roles(labels, device):
    """Assign bodies to finger, arm, connector, cable and hand contact groups."""
    return torch.tensor([
        1 if label.endswith(('/fr3_leftfinger', '/fr3_rightfinger')) else
        5 if label.endswith('/fr3_hand') else
        2 if '/Robot/' in label and '/fr3_link' in label else
        3 if label.endswith(('/lc_plug_link', '/sfp_module_link', '/sfp_tip_link')) else
        4 if '/Cable/link_' in label else 0 for label in labels], device=device)


def contact_totals(first, second, forces, body_world, finger_side, roles, num_envs):
    """Aggregate active last-substep contacts, without advancing history.

    Static contact partners use body ID -1.
    Invalid force magnitudes are excluded as in the source implementation.
    The totals are contact magnitudes, not cable tension or wrist wrench.
    """
    count, bodies = first.numel(), body_world.numel()
    if first.shape != (count,) or second.shape != (count,) or forces.shape != (count, 3):
        raise ValueError('Contact arrays must have matching active lengths')
    if bodies == 0 or num_envs < 1 or finger_side.shape != (bodies,) or roles.shape != (bodies,):
        raise ValueError('Invalid contact body maps')
    if bool(((first < -1) | (second < -1) | (first >= bodies) | (second >= bodies)).any()):
        raise ValueError('Invalid active contact body ID')
    first_safe, second_safe = first.clamp_min(0), second.clamp_min(0)
    first_world, second_world = body_world[first_safe], body_world[second_safe]
    dynamic_pair = (first >= 0) & (second >= 0) & (first_world >= 0) & (second_world >= 0)
    if bool((dynamic_pair & (first_world != second_world)).any()):
        raise RuntimeError('Contact crossed world boundaries')
    magnitude = torch.linalg.vector_norm(forces, dim=-1)
    finite = torch.isfinite(magnitude)
    first_role = torch.where(first >= 0, roles[first_safe], -1)
    second_role = torch.where(second >= 0, roles[second_safe], -1)
    support = torch.zeros(num_envs * 2, device=forces.device, dtype=forces.dtype)
    valid = (first >= 0) & (second >= 0) & finite & (magnitude > .01)
    for finger, head in ((first_safe, second_safe), (second_safe, first_safe)):
        side = finger_side[finger]
        mask = valid & (roles[head] == 3) & (side >= 0)
        indices = (2 * body_world[finger] + side).clamp(0, num_envs * 2 - 1)
        support.scatter_add_(0, indices, torch.where(mask, magnitude, 0))
    groups = torch.zeros((num_envs, 5), device=forces.device, dtype=forces.dtype)
    for column, (affected_role, other_role) in enumerate(((4, 2), (4, 5), (3, -1), (2, -1), (4, -1))):
        forward = (first_role == affected_role) & (second_role == other_role)
        reverse = (second_role == affected_role) & (first_role == other_role)
        affected = torch.where(forward, first_safe, second_safe)
        world = body_world[affected]
        mask = (forward | reverse) & (world >= 0) & finite
        groups[:, column].scatter_add_(0, world.clamp(0, num_envs - 1), torch.where(mask, magnitude, 0))
    return support.reshape(num_envs, 2), groups
