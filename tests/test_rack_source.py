import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from cable_robot_sample.rack_source import configure_rack


@unittest.skipUnless(importlib.util.find_spec('pxr'), 'Requires SDK USD Python')
class RackSourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'original.usda'
        self.source.write_text('#usda 1.0\ndef Xform "external" {}\n')
        self.assets = self.root / 'scene'
        self.assets.mkdir()
        metadata = {'sha256': hashlib.sha256(self.source.read_bytes()).hexdigest()}
        (self.assets / 'rack_source.json').write_text(json.dumps(metadata))

    def test_reference_resolves_without_copying_geometry(self):
        from pxr import Sdf, Usd

        result = configure_rack(self.assets, self.source)
        stage = Usd.Stage.Open(result['reference'])
        self.assertTrue(stage.GetPrimAtPath('/external'))
        self.assertEqual(len(Sdf.Layer.FindOrOpen(result['reference']).rootPrims), 0)
        self.assertEqual(result, configure_rack(self.assets, self.source))

    def test_wrong_source_leaves_no_reference(self):
        self.source.write_text('#usda 1.0\n')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            configure_rack(self.assets, self.source)
        self.assertFalse((self.assets / 'external/gb300.usda').exists())

    def test_cli_writes_usda_in_fresh_process(self):
        subprocess.run([sys.executable, '-m', 'cable_robot_sample', 'configure-rack',
                        '--assets', str(self.assets), '--source', str(self.source)],
                       check=True, capture_output=True, text=True)
        self.assertTrue((self.assets / 'external/gb300.usda').read_text().startswith('#usda 1.0'))

    def test_preserves_unknown_existing_file(self):
        target = self.assets / 'external/gb300.usda'
        target.parent.mkdir()
        target.write_text('#usda 1.0\n')
        with self.assertRaises(FileExistsError):
            configure_rack(self.assets, self.source)
        self.assertEqual(target.read_text(), '#usda 1.0\n')
