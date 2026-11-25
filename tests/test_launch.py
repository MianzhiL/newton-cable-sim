import json
from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from cable_robot_sample.launch import run
from cable_robot_sample.cli import main


class LaunchTests(unittest.TestCase):
    def check_failed_initialization(self, headless, visualizer):
        app = Mock()
        launcher = Mock(return_value=SimpleNamespace(app=app))
        environment = Mock(side_effect=RuntimeError('initialization failed'))
        exports = {
            'isaaclab.app': {'AppLauncher': launcher},
            'torch': {},
            'cable_robot_sample.teleop.compatibility': {'install_capture_fix': Mock()},
            'cable_robot_sample.physics.vbd': {'NewtonAllVBDManager': Mock()},
            'cable_robot_sample.teleop.reset': {'make_config': Mock()},
            'cable_robot_sample.teleop.recorded': {'RecordedCableEnv': environment},
        }
        modules = {}
        for name, attributes in exports.items():
            module = ModuleType(name)
            module.__dict__.update(attributes)
            modules[name] = module
        with tempfile.TemporaryDirectory() as directory, patch.dict('sys.modules', modules):
            args = SimpleNamespace(headless=headless, device='cuda:0', output=Path(directory), reset_at=1)
            with self.assertRaisesRegex(RuntimeError, 'initialization failed'):
                run(args)
            result = json.loads((args.output / 'run-result.json').read_text())
        launcher.assert_called_once_with(headless=headless, visualizer=visualizer, device='cuda:0')
        app.close.assert_called_once_with(exit_code=1)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['control_steps'], 0)

    def test_headless_disables_visualizer_and_reports_failure(self):
        self.check_failed_initialization(True, 'none')

    def test_desktop_keeps_kit_and_reports_failure(self):
        self.check_failed_initialization(False, 'kit')


class DemoCliTests(unittest.TestCase):
    def arguments(self, output):
        result = ['sample', 'run', '--headless']
        for name in ('assets', 'robot'):
            result.extend(['--' + name, 'fixture'])
        return result + ['--output', str(output)]

    def test_license_and_bounds_fail_before_runtime_import(self):
        for accepted, extra in (('', []), ('YES', ['--steps', '0']),
                                ('YES', ['--grip-friction', 'nan']), ('YES', ['--reset-at', '600'])):
            with self.subTest(accepted=accepted, extra=extra), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'new-run'
                with patch.dict('os.environ', {'OMNI_KIT_ACCEPT_EULA': accepted}), \
                        patch('sys.argv', self.arguments(output) + extra), \
                        patch('cable_robot_sample.assets.validate_assets', return_value={}), \
                        patch('cable_robot_sample.launch.run') as runner, redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as failure:
                        main()
                self.assertEqual(failure.exception.code, 2)
                self.assertFalse(output.exists())
                runner.assert_not_called()

    def test_dispatch_records_launch_and_rejects_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'new-run'
            with patch('cable_robot_sample.assets.validate_assets', return_value={}), \
                    patch('cable_robot_sample.launch.run') as runner, \
                    patch.dict('os.environ', {'OMNI_KIT_ACCEPT_EULA': 'YES'}), \
                    patch('sys.argv', self.arguments(output)):
                main()
                runner.assert_called_once()
                saved = json.loads((output / 'launch.json').read_text())
                self.assertTrue(saved['headless'])
                self.assertEqual(saved['steps'], 600)
                self.assertEqual(saved['grip_friction'], 1.0)
                with self.assertRaises(FileExistsError):
                    main()
                runner.assert_called_once()
