import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from cable_robot_sample.learning.admission import validate_admission


HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from cable_robot_sample.learning.control import grip_events, project_control
    from cable_robot_sample.learning.policy import InsertionPolicy, load_policy
    from cable_robot_sample.learning.reward import AUXILIARY_NAMES, AUXILIARY_SCALE, objective_reward, RiskConfig, risk_costs


class LearningContractTests(unittest.TestCase):
    def test_retained_configuration(self):
        path = Path(__file__).resolve().parents[1] / 'src/cable_robot_sample/learning/ppo.json'
        config = json.loads(path.read_text())
        self.assertEqual(config['num_steps_per_env'], 512)
        self.assertEqual(config['actor']['hidden_dims'], [256, 256])
        self.assertEqual(config['algorithm']['gamma'], 1)
        self.assertEqual(config['algorithm']['lam'], 1)
        self.assertFalse(config['actor']['obs_normalization'])
        self.assertNotIn('multi_gpu', config)

    def test_admission_preserves_held_out_set(self):
        train, test = list(range(2049)), list(range(2049, 2334))
        manifest = {'sha256': 'test-only', 'splits': {'train': train, 'test': test}}
        admission = dict(mode='per_state_admission', bank_sha256='test-only', infrastructure_passed=True,
                         canonical_grip_unchanged=True, exact_retained_state_copy=True,
                         mapped_parent_scan_count=2334, peer_reset_checks=3, train_test_separation=True,
                         quarantined_train_ids=[0], active_train_ids=train[1:], evaluation_ids=test,
                         maximum_reset_attempts=3)
        self.assertEqual(validate_admission(manifest, admission), train[1:])
        for field, replacement in [('evaluation_ids', test[1:]), ('quarantined_train_ids', [2049]),
                                   ('maximum_reset_attempts', 4), ('bank_sha256', 'wrong')]:
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                validate_admission(manifest, dict(admission, **{field: replacement}))


@unittest.skipUnless(HAS_TORCH, 'Run tensor checks with the pinned SDK Python')
class LearningTensorTests(unittest.TestCase):
    def test_control_and_persistent_grip_loss(self):
        zeros = torch.zeros(3, 7)
        target, increment = project_control(torch.ones_like(zeros), zeros, zeros, zeros,
                                            -torch.ones_like(zeros), torch.ones_like(zeros))
        torch.testing.assert_close(target, torch.full_like(zeros, .0005))
        torch.testing.assert_close(target, increment)
        lost, unsupported, escaped = grip_events(torch.tensor([.02, .02, .04]), torch.zeros(3),
                                                 torch.tensor([False, False, True]),
                                                 torch.tensor([13, 14, 0]), torch.tensor([0, 0, 14]))
        self.assertEqual(lost.tolist(), [False, True, True])

    def test_terminal_shaping_and_risk_budget(self):
        previous = torch.tensor([-.5, -.2])
        following = torch.tensor([-.2, -.1])
        done = torch.tensor([False, True])
        raw = {name: torch.full((2,), -.001) for name in AUXILIARY_NAMES}
        raw.update(event=torch.tensor([0., 20.]), shaping=torch.zeros(2))
        reward, parts = objective_reward(previous, following, done, raw)
        torch.testing.assert_close(parts['shaping'], torch.tensor([.6, .4]))
        torch.testing.assert_close(reward, raw['event'] + parts['shaping'] - .006 * AUXILIARY_SCALE)
        costs = risk_costs(torch.tensor([1., float('nan')]), torch.tensor([90., float('nan')]),
                           torch.tensor([100., float('nan')]), torch.ones(2),
                           torch.tensor([True, False]), RiskConfig())
        self.assertAlmostEqual(float(sum(costs.values())[0]), .125 / 60)
        self.assertEqual(float(sum(costs.values())[1]), 0)

    def test_checkpoint_roundtrip_and_rejection(self):
        torch.manual_seed(7)
        source = InsertionPolicy().eval()
        state = dict(source.state_dict(), **{'distribution.log_std_param': torch.zeros(7)})
        observations = torch.randn(4, 234)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policy.pt'
            torch.save({'actor_state_dict': state}, path)
            loaded = load_policy(path)
            torch.testing.assert_close(source(observations), loaded(observations), rtol=0, atol=0)
            self.assertTrue(all(not parameter.requires_grad for parameter in loaded.parameters()))
            invalid = copy.deepcopy(state)
            invalid['mlp.0.weight'][0, 0] = float('nan')
            torch.save({'actor_state_dict': invalid}, path)
            with self.assertRaises(ValueError):
                load_policy(path)
        with self.assertRaises(ValueError):
            source(torch.zeros(4, 233))


if __name__ == '__main__':
    unittest.main()
