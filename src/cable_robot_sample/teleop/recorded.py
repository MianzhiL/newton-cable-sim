import time

from cable_robot_sample.teleop.reset import ResettableCableEnv as CableTeleopEnv
from cable_robot_sample.recording.trajectory import TrajectoryRecorder


class RecordedCableEnv(CableTeleopEnv):
    def __init__(self, *args, **kwargs):
        self._trajectory = None
        self._trajectory_busy = True
        super().__init__(*args, **kwargs)
        self._trajectory = TrajectoryRecorder(self)
        self._trajectory.capture('initial')
        self.sim.add_render_callback('cable_trajectory_recording', self._record_paused_camera)
        self._trajectory_busy = False

    def _record_paused_camera(self, event):
        if self._trajectory is not None and not self._trajectory_busy:
            if time.monotonic() - self._trajectory.last_capture >= 0.1:
                self._trajectory.capture('paused')

    def step(self, action):
        advanced = not self._cable_fault
        self._trajectory_busy = True
        try:
            result = super().step(action)
            self._trajectory.capture('step' if advanced else 'paused', action)
            return result
        finally:
            self._trajectory_busy = False

    def reset(self, *args, **kwargs):
        self._trajectory_busy = True
        try:
            if self._trajectory is not None:
                self._trajectory.flush()
            result = super().reset(*args, **kwargs)
            if self._trajectory is not None:
                self._trajectory.new_segment()
                self._trajectory.capture('reset')
            return result
        finally:
            self._trajectory_busy = False

    def close(self):
        try:
            if self._trajectory is not None:
                self.sim.remove_render_callback('cable_trajectory_recording')
                self._trajectory.capture('final')
                self._trajectory.close()
                self._trajectory = None
        finally:
            super().close()
