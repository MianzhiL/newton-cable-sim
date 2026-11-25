import importlib.util
import itertools
import unittest


HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from cable_robot_sample.learning.contacts import contact_roles, contact_totals
    from cable_robot_sample.learning.geometry import corridor_clearance, inverse_rotate, rotate, transform_points


@unittest.skipUnless(HAS_TORCH, 'Run tensor checks with the pinned SDK Python')
class MeasurementTests(unittest.TestCase):
    def test_point_transform_and_inverse_rotation(self):
        quaternion = torch.tensor([[0., 0., 2 ** -.5, 2 ** -.5]])
        points = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])
        pose = torch.cat((torch.tensor([[1., 2., 3.]]), quaternion), -1)
        transformed = transform_points(pose, points)
        torch.testing.assert_close(transformed, torch.tensor([[[1., 3., 3.], [0., 2., 3.]]]))
        expanded = quaternion[:, None].expand(1, 2, 4)
        torch.testing.assert_close(inverse_rotate(expanded, rotate(expanded, points)), points)

    def test_corridor_crossing_and_backstop(self):
        points = torch.tensor(list(itertools.product((-.010, .010), (-.001, .003), (-.005, .005))))[None, None]
        clearance, backstop = corridor_clearance(points)
        torch.testing.assert_close(clearance, torch.tensor([.001]))
        torch.testing.assert_close(backstop, torch.tensor([.057]))
        points[..., 0] += .002
        clearance, _ = corridor_clearance(points)
        torch.testing.assert_close(clearance, torch.tensor([-.001]))
        points[..., 1] -= 1
        clearance, _ = corridor_clearance(points)
        torch.testing.assert_close(clearance, torch.ones(1))

    def maps(self):
        labels = ['/World/envs/env_0/Robot/fr3_leftfinger', '/World/envs/env_0/Robot/fr3_rightfinger',
                  '/World/envs/env_0/Cable/sfp_module_link', '/World/envs/env_0/Cable/link_0',
                  '/World/envs/env_0/Robot/fr3_link3', '/World/envs/env_0/Robot/fr3_hand',
                  '/World/envs/env_1/Cable/sfp_module_link', '/shared/rack']
        return torch.tensor([0, 0, 0, 0, 0, 0, 1, -1]), torch.tensor([0, 1, -1, -1, -1, -1, -1, -1]), contact_roles(labels, 'cpu')

    def test_support_and_all_five_risk_groups(self):
        world, sides, roles = self.maps()
        first = torch.tensor([0, 2, 3, 5, 2, -1, 3, 3, 2])
        second = torch.tensor([2, 1, 4, 3, -1, 4, -1, 0, 7])
        forces = torch.zeros(9, 3)
        forces[:, 0] = torch.arange(1, 10)
        support, groups = contact_totals(first, second, forces, world, sides, roles, 2)
        torch.testing.assert_close(support, torch.tensor([[1., 2.], [0., 0.]]))
        torch.testing.assert_close(groups, torch.tensor([[3., 4., 5., 6., 7.], [0., 0., 0., 0., 0.]]))

    def test_invalid_ids_and_cross_world_contact_rejected(self):
        world, sides, roles = self.maps()
        for second in (-2, 8):
            with self.assertRaises(ValueError):
                contact_totals(torch.tensor([0]), torch.tensor([second]), torch.ones(1, 3), world, sides, roles, 2)
        with self.assertRaisesRegex(RuntimeError, 'world'):
            contact_totals(torch.tensor([0]), torch.tensor([6]), torch.ones(1, 3), world, sides, roles, 2)

    def test_empty_nonfinite_and_small_contacts(self):
        world, sides, roles = self.maps()
        empty = torch.empty(0, dtype=torch.long)
        support, groups = contact_totals(empty, empty, torch.empty(0, 3), world, sides, roles, 2)
        self.assertEqual(float(support.sum() + groups.sum()), 0)
        forces = torch.tensor([[float('nan'), 0., 0.], [.005, 0., 0.]])
        support, groups = contact_totals(torch.tensor([0, 1]), torch.tensor([2, 2]), forces, world, sides, roles, 2)
        self.assertEqual(float(support.sum() + groups.sum()), 0)
