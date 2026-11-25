import os
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from cable_robot_sample.learning.runner import parser, validate_arguments, save_checkpoint, restore_checkpoint


class LearningCliTests(unittest.TestCase):
    def test_valid_train_and_output_protection(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, OMNI_KIT_ACCEPT_EULA='YES', WORLD_SIZE='1'):
            root = Path(directory)
            robot = root / 'robot.usda'
            robot.touch()
            args = parser().parse_args(['train', '--task-assets', str(root), '--robot', str(robot),
                                         '--output', str(root / 'new'), '--headless', '--iterations', '1'])
            validate_arguments(args)
            args.output.mkdir()
            with self.assertRaises(FileExistsError):
                validate_arguments(args)

    def test_invalid_combinations(self):
        base = ['--task-assets', '.', '--robot', '.', '--output', 'unused-output']
        for command in (['evaluate'], ['play', '--num-envs', '2'], ['train', '--num-envs', '0'],
                        ['train', '--initial-ids', '8159'], ['train', '--record']):
            with self.subTest(command=command), self.assertRaises(ValueError):
                validate_arguments(parser().parse_args([*command, *base]))


@unittest.skipUnless(all(importlib.util.find_spec(name) is not None for name in ('torch', 'rsl_rl', 'tensordict')),
                     'Run checkpoint checks with the pinned SDK Python')
class CheckpointTests(unittest.TestCase):
    def test_models_and_both_adam_states_resume(self):
        import torch
        from tensordict import TensorDict
        from rsl_rl.algorithms import PPO

        config_path = Path(__file__).resolve().parents[1] / 'src/cable_robot_sample/learning/ppo.json'
        observations = TensorDict({'policy': torch.zeros(2, 234), 'critic': torch.zeros(2, 248)}, [2])
        def construct():
            config = json.loads(config_path.read_text())
            config.update(num_steps_per_env=4, multi_gpu=None)
            return PPO.construct_algorithm(observations, SimpleNamespace(num_envs=2, num_actions=7), config, 'cpu')

        original, restored = construct(), construct()
        for module in (original.actor, original.critic):
            for parameter in module.parameters():
                parameter.grad = torch.ones_like(parameter)
        original.optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            save_checkpoint(original, path, 7)
            self.assertEqual(restore_checkpoint(restored, path, 'cpu'), 7)
        for name in ('actor', 'critic'):
            for key, value in getattr(original, name).state_dict().items():
                torch.testing.assert_close(value, getattr(restored, name).state_dict()[key], rtol=0, atol=0)
            expected = getattr(original.optimizer, name).state_dict()
            actual = getattr(restored.optimizer, name).state_dict()
            self.assertEqual(expected['param_groups'], actual['param_groups'])
            self.assertEqual(expected['state'].keys(), actual['state'].keys())
            for index, state in expected['state'].items():
                for key, value in state.items():
                    torch.testing.assert_close(value, actual['state'][index][key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
