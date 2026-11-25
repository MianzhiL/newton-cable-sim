import itertools

import numpy as np
import torch


def rotate(quaternion, vector):
    cross = 2 * torch.cross(quaternion[..., :3], vector, dim=-1)
    return vector + quaternion[..., 3:] * cross + torch.cross(quaternion[..., :3], cross, dim=-1)


def inverse_rotate(quaternion, vector):
    return rotate(torch.cat((-quaternion[..., :3], quaternion[..., 3:]), -1), vector)


def collision_corners(path, body_name):
    """Read body-relative collision-box bounds, not render-mesh silhouettes."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f'Cannot open collision asset: {path}')
    bodies = [prim for prim in stage.Traverse()
              if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.GetName() == body_name]
    if len(bodies) != 1:
        raise ValueError(f'Expected one rigid body: {body_name}')
    body = bodies[0]
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy', 'guide'], False, True)
    corners = []
    for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        ancestor = prim
        while ancestor and not ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
            ancestor = ancestor.GetParent()
        if ancestor != body:
            continue
        bound = cache.ComputeRelativeBound(prim, body).ComputeAlignedRange()
        corners.append(list(itertools.product(*zip(bound.GetMin(), bound.GetMax()))))
    if not corners or not np.isfinite(corners).all():
        raise ValueError(f'Invalid collider bounds: {body_name}')
    return np.asarray(corners, dtype=np.float32)


def transform_points(pose, points):
    quaternion = pose[..., None, 3:].expand(*points.shape[:-1], 4)
    return rotate(quaternion, points) + pose[..., None, :3]


def corridor_clearance(points, half_width=.011, half_height=.006, port_depth=.06):
    """Screen collider AABBs against the axis-aligned port corridor.

    Include box-edge intersections with the entrance plane. A negative result
    indicates an overlap of these bounds, not an exact mesh penetration test.
    """
    shape_count = points.shape[1]
    flattened = points.flatten(1, 2)
    side = torch.minimum(half_width - flattened[..., 0].abs(), half_height - flattened[..., 2].abs())
    clearance = torch.where(flattened[..., 1] >= 0, side, torch.inf).amin(-1)
    first_ids, second_ids = [], []
    for first in range(8):
        for bit in (1, 2, 4):
            second = first ^ bit
            if first < second:
                first_ids.append(first)
                second_ids.append(second)
    first, second = points[:, :, first_ids], points[:, :, second_ids]
    crossing = (first[..., 1] < 0) != (second[..., 1] < 0)
    denominator = second[..., 1] - first[..., 1]
    fraction = -first[..., 1] / torch.where(denominator.abs() > 1e-12, denominator, 1)
    intersection = first + fraction[..., None] * (second - first)
    edge_gap = torch.minimum(half_width - intersection[..., 0].abs(), half_height - intersection[..., 2].abs())
    clearance = torch.minimum(clearance, torch.where(crossing, edge_gap, torch.inf).reshape(-1, shape_count * 12).amin(-1))
    return clearance.clamp(-1, 1), port_depth - flattened[..., 1].amax(-1)


def grip_clearance(poses, sfp, hand, entrance, sfp_corners, fingers, finger_corners):
    batch = len(sfp)
    local_pose = poses[sfp].clone()
    local_pose[:, :3] -= entrance
    points = transform_points(local_pose[:, None], sfp_corners[None].expand(batch, -1, -1, -1))
    corridor, backstop = corridor_clearance(points)
    hand_pose = poses[hand]
    world_points = points + entrance[:, None, None, :]
    sfp_hand = inverse_rotate(hand_pose[:, None, None, 3:].expand(-1, *world_points.shape[1:3], -1),
                              world_points - hand_pose[:, None, None, :3])
    pads = []
    for indices, corners in zip(fingers, finger_corners):
        world = transform_points(poses[indices], corners[None].expand(batch, -1, -1))
        pads.append(inverse_rotate(hand_pose[:, None, 3:].expand(-1, world.shape[1], -1), world - hand_pose[:, None, :3]))
    pads = torch.cat(pads, 1)
    lower, upper = pads.amin(1), pads.amax(1)
    gaps = torch.maximum(lower[:, None] - sfp_hand.amax(2), sfp_hand.amin(2) - upper[:, None]).clamp(min=0)
    return corridor, backstop, torch.linalg.vector_norm(gaps, dim=-1).amin(-1)
