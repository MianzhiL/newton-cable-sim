import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from cable_robot_sample.recording.overlay import ForceOverlay


class OverlayTests(unittest.TestCase):
    def test_endpoint_roundoff_keeps_force_bound_and_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            forces = root / 'forces'
            forces.mkdir()
            (forces / 'metadata.json').write_text(json.dumps({'physics_hz': 960, 'control_hz': 60}))
            (root / 'events.jsonl').write_text(json.dumps({'event': 'step', 'step': 120, 'wall_step_ms': 80}) + '\n')
            components = ('Fx_N', 'Fy_N', 'Fz_N', 'Tx_Nm', 'Ty_Nm', 'Tz_Nm')
            headers = ['segment', 'control_step', 'valid'] + [f'wrist_{name}_mean' for name in components] + ['cable1_fixed_force_norm_max_N']
            with (forces / 'summary.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(headers)
                writer.writerow([2, 120, 1, 20, 0, 0, 0, 0, 0, 5])
            recording = SimpleNamespace(session=root, metadata={'curves': [{'path': '/World/Cable1', 'identifier': 'L01'}]})
            rows = [('chunk', 0, 1.0000000000000013, 0, 120, 1)]
            overlay = ForceOverlay(recording, 2, rows, 'sim', 0, 1, cable_colors=[(255, 0, 0)])
            self.assertEqual(overlay.performance_summary['samples'], 1)
            self.assertEqual(overlay.performance_summary['sampled_env_steps_per_second'], 12.5)
            self.assertAlmostEqual(overlay.bounds[0], 22)
            self.assertAlmostEqual(overlay.bounds[2], 5.5)
