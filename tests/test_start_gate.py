import unittest

import numpy as np

from cable_robot_sample.teleop.compatibility import StartGate, install_capture_fix


class StartGateTests(unittest.TestCase):
    def test_start_pause_tracking_loss_and_reset(self):
        gate = StartGate()
        events = [
            (False, False, -1, False, True, (None, False)),
            (False, False, 0, False, False, (None, False)),
            (True, False, 1, False, False, (None, False)),
            (True, False, 1.8, False, False, ('start', False)),
            (True, False, 1.9, True, False, (None, True)),
            (True, True, 2, True, False, ('stop', False)),
            (True, True, 3, False, False, (None, False)),
            (True, False, 3.1, True, False, (None, True)),
            (True, True, 4, False, False, ('start', False)),
            (True, False, 4.8, True, False, (None, True)),
            (False, False, 5, True, False, ('stop', False)),
            (True, False, 6, False, False, (None, False)),
            (True, False, 6.8, False, False, (None, False)),
            (True, False, 7, True, False, (None, True)),
            (True, False, 8, False, True, (None, False)),
            (True, False, 9, False, False, (None, False)),
            (True, False, 10, False, False, (None, False)),
            (True, False, 11, True, False, (None, True)),
        ]
        for valid, button, timestamp, active, reset, expected in events:
            with self.subTest(timestamp=timestamp):
                self.assertEqual(gate.update(valid, button, timestamp, active=active, reset=reset), expected)

    def test_capture_cursor_restored_on_success_and_error(self):
        class Cursor:
            def __init__(self):
                self.value = np.array([3], dtype=np.int32)

            def numpy(self):
                return self.value.copy()

            def assign(self, value):
                self.value[:] = value

        class Solver:
            record_cursor = Cursor()

        class Manager:
            _solver = Solver()

            @classmethod
            def _capture_relaxed_graph(cls, device, capture_target=None):
                cls._solver.record_cursor.value += 1
                if device == 'fail':
                    raise RuntimeError('synthetic capture failure')
                return 'graph'

        install_capture_fix(Manager)
        install_capture_fix(Manager)
        self.assertEqual(Manager._capture_relaxed_graph('test'), 'graph')
        self.assertEqual(Manager._solver.record_cursor.numpy()[0], 3)
        with self.assertRaises(RuntimeError):
            Manager._capture_relaxed_graph('fail')
        self.assertEqual(Manager._solver.record_cursor.numpy()[0], 3)


if __name__ == '__main__':
    unittest.main()
