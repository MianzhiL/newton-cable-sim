from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from cable_robot_sample.learning.reference import array_digest, checked_member, map_worlds


class ReferenceMappingTests(unittest.TestCase):
    def fixture(self):
        body_labels = [f'/World/envs/env_0/body{index}' for index in range(24)]
        joint_labels = [f'/World/envs/env_0/joint{index}' for index in range(24)]
        structure = dict(body_mass=np.arange(24, dtype=np.float32),
                         body_inertia=np.tile(np.eye(3, dtype=np.float32), (24, 1, 1)),
                         body_com=np.zeros((24, 3), dtype=np.float32),
                         joint_type=np.zeros(24, dtype=np.int32))
        reference = SimpleNamespace(body_labels=body_labels, joint_labels=joint_labels, structure=structure,
                                    manifest={'structure_sha256': {name: array_digest(value) for name, value in structure.items()}})
        counts = np.array([2] * 17 + [1] * 7)
        starts = np.concatenate(([0], np.cumsum(np.tile(counts, 2)))).astype(np.int32)
        arrays = {name: np.concatenate((value, value)) for name, value in structure.items()}
        arrays.update(body_world=np.repeat([0, 1], 24), joint_q_start=starts, joint_qd_start=starts,
                      joint_coord_count=82, joint_dof_count=82)
        bodies = body_labels + [label.replace('/env_0/', '/env_1/') for label in body_labels]
        joints = joint_labels + [label.replace('/env_0/', '/env_1/') for label in joint_labels]
        return reference, bodies, joints, arrays

    def test_world_mapping_and_coordinate_ranges(self):
        reference, bodies, joints, arrays = self.fixture()
        result = map_worlds(reference, bodies, joints, arrays, 2, 99)
        np.testing.assert_array_equal(result['bodies'], np.arange(48).reshape(2, 24))
        np.testing.assert_array_equal(result['coordinates'], np.arange(82).reshape(2, 41))
        arrays['body_com'][25, 0] = .5e-7
        result = map_worlds(reference, bodies, joints, arrays, 2, 99)
        self.assertGreater(result['com_errors'][1], 0)

    def test_mismatch_rejected_before_model_mutation(self):
        reference, bodies, joints, arrays = self.fixture()
        changes = [('body_mass', (25,), 99), ('body_world', (25,), 0),
                   ('body_com', (25, 0), 2e-7), ('joint_type', (25,), 99),
                   ('joint_q_start', (-1,), 83)]
        for name, index, value in changes:
            modified = deepcopy(arrays)
            modified[name][index] = value
            with self.subTest(field=name), self.assertRaises(ValueError):
                map_worlds(reference, bodies, joints, modified, 2, 99)
        with self.assertRaises(ValueError):
            map_worlds(reference, bodies[:-1], joints, arrays, 2, 99)

    def test_reference_members_cannot_escape_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inside = root / 'reference'
            inside.mkdir()
            (root / 'outside').write_bytes(b'test')
            (inside / 'value').write_bytes(b'test')
            self.assertEqual(checked_member(inside, 'value'), inside / 'value')
            with self.assertRaises(ValueError):
                checked_member(inside, '../outside')


if __name__ == '__main__':
    unittest.main()
