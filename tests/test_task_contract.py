import importlib.util
import unittest


HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from cable_robot_sample.learning.observations import (
        ActorState, actor_observation, critic_observation, risk_features, terminal_observation,
    )
    from cable_robot_sample.learning.termination import TaskMetrics, evaluate_termination
    from cable_robot_sample.learning.reward import AUXILIARY_SCALE, transition_reward


@unittest.skipUnless(HAS_TORCH, 'Run tensor checks with the pinned SDK Python')
class TaskContractTests(unittest.TestCase):
    def make_state(self, batch=4):
        scalar = torch.zeros(batch)
        quaternion = torch.tensor([0., 0., 0., 1.]).repeat(batch, 1)
        return ActorState(
            torch.zeros(batch, 9), torch.zeros(batch, 9), torch.zeros(batch, 13, 7),
            torch.zeros(batch, 13, 6), torch.zeros(batch, 3), torch.zeros(batch, 3),
            quaternion, torch.zeros(batch, 3), torch.zeros(batch, 9),
            torch.zeros(batch, 7), torch.zeros(batch, 7), torch.zeros(batch, 3),
            torch.zeros(batch, 3), quaternion, quaternion, torch.full((batch, 2), 60.),
            scalar, scalar, scalar, scalar, scalar, scalar,
        )

    def test_observation_layout_scaling_and_no_input_mutation(self):
        state = self.make_state()
        state.cable_poses[:, :, :3] = 3
        state.entrance[:] = 1
        state.cable_velocities[:] = 10
        state.front_delta[:] = .01
        state.relative_grip[:] = .02
        state.targets[:, :7] = .04
        policy = actor_observation(state)
        self.assertEqual(policy.shape, (4, 234))
        torch.testing.assert_close(policy[:, 18:109].reshape(4, 13, 7)[:, :, :3], torch.full((4, 13, 3), 4.))
        torch.testing.assert_close(policy[:, 109:187].reshape(4, 13, 6)[:, :, :3], torch.full((4, 13, 3), 10.))
        torch.testing.assert_close(policy[:, 109:187].reshape(4, 13, 6)[:, :, 3:], torch.full((4, 13, 3), 2.))
        torch.testing.assert_close(policy[:, 187:190], torch.full((4, 3), .2))
        torch.testing.assert_close(policy[:, 194:197], torch.full((4, 3), .2))
        torch.testing.assert_close(policy[:, 197:204], torch.ones(4, 7))
        torch.testing.assert_close(policy[:, 221:225], torch.tensor([0., 0., 0., 1.]).repeat(4, 1))
        torch.testing.assert_close(policy[:, 225:227], torch.ones(4, 2))
        torch.testing.assert_close(policy[:, 231], torch.ones(4))
        torch.testing.assert_close(state.cable_poses[:, :, :3], torch.full((4, 13, 3), 3.))
        state.front_delta[0, 0] = float('nan')
        with self.assertRaises(ValueError):
            actor_observation(state)

    def test_risk_features_and_failure_terminal_fallback(self):
        state = self.make_state()
        policy = actor_observation(state)
        features = risk_features(state.relative_sfp, state.relative_sfp, state.rotation_sfp,
                                 state.rotation_sfp, state.relative_reference, state.rotation_reference,
                                 torch.zeros(4, 5), torch.zeros(4, 2), torch.zeros(4))
        torch.testing.assert_close(features, torch.zeros(4, 14))
        critic = critic_observation(policy, features)
        self.assertEqual(critic.shape, (4, 248))
        failed = torch.tensor([True, False, False, False])
        current = critic.clone()
        current[0] = float('nan')
        restored = terminal_observation(current, critic, failed)
        torch.testing.assert_close(restored, critic)
        self.assertTrue(torch.isnan(current[0]).all())

    def good_metrics(self, batch):
        zeros = torch.zeros(batch)
        return TaskMetrics(torch.full((batch,), .01), zeros.clone(), zeros.clone(), zeros.clone(),
                           zeros.clone(), zeros.clone(), zeros.clone(), zeros.clone(), zeros.clone(), zeros.clone())

    def test_terminal_precedence_and_deadline_not_bootstrapped(self):
        metrics = self.good_metrics(5)
        metrics.gap[1] = .04
        metrics.position_error[2] = .16
        metrics.lateral[3] = .003
        failed = torch.tensor([False, False, False, False, True])
        events = evaluate_termination(metrics, torch.zeros(5, 2, dtype=torch.bool), failed,
                                      torch.full((5,), 2400), torch.full((5,), 14),
                                      torch.full((5,), 14), torch.full((5,), 29))
        for index, name in enumerate(('success', 'grip_loss', 'outside', 'deadline', 'physics_failure')):
            self.assertEqual(getattr(events, name).nonzero().flatten().tolist(), [index])
        self.assertTrue(events.done.all())
        self.assertFalse(events.time_outs.any())
        self.assertEqual(events.terminal_valid.tolist(), [True, True, True, True, False])
        reward, parts = transition_reward(torch.full((5,), -.5), torch.zeros(5), events,
                                          torch.zeros(5, 7), torch.zeros(5, 7),
                                          metrics.drift, metrics.rotation_error, torch.zeros(5), torch.zeros(5))
        torch.testing.assert_close(parts['event'], torch.tensor([20., -5., -5., -5., -5.]))
        torch.testing.assert_close(reward, parts['event'] + 1 - .05 / 60 * AUXILIARY_SCALE)

    def test_success_hold_restarts_and_small_grip_drift_is_not_failure(self):
        metrics = self.good_metrics(1)
        zeros = torch.zeros(1, dtype=torch.long)
        metrics.drift[:] = .02
        metrics.rotation_error[:] = 20
        hold = zeros
        for step in range(29):
            events = evaluate_termination(metrics, torch.ones(1, 2, dtype=torch.bool), zeros.bool(),
                                          zeros + step + 1, zeros, zeros, hold)
            hold = events.success_hold
            self.assertFalse(events.done.any())
        metrics.angle_deg[:] = 5
        events = evaluate_termination(metrics, torch.ones(1, 2, dtype=torch.bool), zeros.bool(),
                                      zeros + 30, zeros, zeros, hold)
        self.assertEqual(events.success_hold.item(), 0)
        self.assertFalse(events.done.any())


if __name__ == '__main__':
    unittest.main()
