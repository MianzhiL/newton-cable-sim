import ast
from pathlib import Path
import unittest


class SourceTests(unittest.TestCase):
    def test_package_imports_are_self_contained(self):
        package = Path(__file__).resolve().parents[1] / 'src/cable_robot_sample'
        for path in sorted(package.rglob('*.py')):
            with self.subTest(module=path.name):
                tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('cable_robot_sample.'):
                        target = package.parent.joinpath(*node.module.split('.'))
                        self.assertTrue(target.with_suffix('.py').is_file() or (target / '__init__.py').is_file(), node.module)

    def test_public_source_has_no_machine_paths(self):
        package = Path(__file__).resolve().parents[1] / 'src/cable_robot_sample'
        for path in sorted(package.rglob('*.py')):
            with self.subTest(module=path.name):
                source = path.read_text(encoding='utf-8')
                self.assertNotIn('/home/', source)
                self.assertNotIn('/opt/cable_project', source)
                self.assertNotIn('sys.path.insert', source)


if __name__ == '__main__':
    unittest.main()
