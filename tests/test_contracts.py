import json
from pathlib import Path
import tempfile
import unittest

from cable_robot_sample.assets import validate_assets
from cable_robot_sample.recording.reader import Recording, schedule


class AssetContractTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for name in ('rack.usda', 'robot.usda', 'Cable1.usdc'):
            (self.root / name).touch()
        self.manifest = {'cables': [{'name': 'Cable1', 'identifier': 'L01', 'operable': True,
            'rod_count': 2, 'segment_lengths_m': [0.04, 0.04], 'jacket_color': [0, 0.5, 1],
            'jacket_radius_m': 0.003, 'connector_color': None, 'bodies': {'sc_plug_link': {
                'position': [0, 0, 1], 'quaternion_xyzw': [0, 0, 0, 1]}}}]}

    def validate(self):
        (self.root / 'manifest.json').write_text(json.dumps(self.manifest))
        return validate_assets(self.root, self.root / 'robot.usda')

    def test_valid_manifest_uses_packaged_material(self):
        self.assertEqual(self.validate()['rod_segments'], 2)

    def test_bad_lengths(self):
        self.manifest['cables'][0]['segment_lengths_m'][0] = -1
        with self.assertRaises(ValueError):
            self.validate()

    def test_missing_connector_color(self):
        del self.manifest['cables'][0]['connector_color']
        with self.assertRaisesRegex(ValueError, 'connector_color is required'):
            self.validate()

    def test_connector_color(self):
        cable = self.manifest['cables'][0]
        cable['connector_color'] = {'color_linear': [0.1, 0.4, 1.0]}
        self.validate()
        for invalid in ([1, 0, 0], {}, {'color_linear': [2, 0, 0]}):
            cable['connector_color'] = invalid
            with self.assertRaises(ValueError):
                self.validate()

    def test_bad_quaternion(self):
        self.manifest['cables'][0]['bodies']['sc_plug_link']['quaternion_xyzw'] = [0, 0, 0, 0]
        with self.assertRaises(ValueError):
            self.validate()

    def test_unsafe_name(self):
        self.manifest['cables'][0]['name'] = '../Cable1'
        with self.assertRaises(ValueError):
            self.validate()


class RecordingContractTests(unittest.TestCase):
    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'trajectory'
            root.mkdir()
            (root / 'metadata.json').write_text('{"schema": 2}')
            (root / 'chunks.jsonl').write_text('{"file":"../outside.npz","frames":1,"segment":0}\n')
            with self.assertRaisesRegex(ValueError, 'Invalid chunk path'):
                Recording(directory)

    def test_half_speed_doubles_samples(self):
        rows = [('chunk', index, index / 60, 100 + index / 30, index, 1) for index in range(61)]
        normal, _, _ = schedule(rows, 'sim', 0, 1, 30, 1)
        slow, _, _ = schedule(rows, 'sim', 0, 1, 30, 0.5)
        self.assertEqual(len(slow) - 1, 2 * (len(normal) - 1))

    def test_invalid_schedule(self):
        for rows, timebase in (([], 'sim'), ([('chunk', 0, 0, 0, 0, 1)], 'invalid')):
            with self.assertRaises(ValueError):
                schedule(rows, timebase, 0, 1, 30, 1)

    def test_schedule_includes_endpoint_with_float_roundoff(self):
        rows = [('chunk', index, index / 60 + 1e-15, index / 60, index, 1) for index in range(61)]
        times, indices, _ = schedule(rows, 'sim', 0, 1, 60, 1)
        self.assertEqual(indices[-1], 60)
        self.assertEqual(indices[30], 30)
        self.assertEqual(times[-1], 1)


if __name__ == '__main__':
    unittest.main()
