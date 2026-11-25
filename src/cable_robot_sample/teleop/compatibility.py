import os
import time


class StartGate:
    def __init__(self):
        self.ready_since = None
        self.auto_start_pending = True
        self.has_run = False
        self.was_active = False
        self.button_down = False
        self.last_toggle = float('-inf')

    def update(self, valid, button, now, active=False, reset=False):
        if reset:
            if self.has_run:
                self.auto_start_pending = False
            self.ready_since = None
            self.was_active = False
            return None, False
        if active:
            self.auto_start_pending = False
            self.has_run = True
        elif self.was_active:
            self.auto_start_pending = False
        previously_active = self.was_active
        self.was_active = bool(active)
        if not valid:
            command = 'stop' if previously_active else None
            self.ready_since = None
            self.button_down = False
            return command, False
        if self.ready_since is None:
            self.ready_since = now
        rising = button and not self.button_down
        self.button_down = button
        if rising and now - self.last_toggle >= 0.4:
            self.last_toggle = now
            self.auto_start_pending = False
            return ('stop' if active else 'start'), False
        if now - self.ready_since < 0.75:
            return None, False
        if self.auto_start_pending:
            self.auto_start_pending = False
            self.has_run = True
            return 'start', False
        return None, bool(active)


def install_capture_fix(manager):
    if getattr(manager, '_cable_capture_recording_fix', False):
        return
    original = manager._capture_relaxed_graph.__func__

    def capture(cls, device, capture_target=None):
        solver = cls._solver
        cursor = getattr(solver, 'record_cursor', None)
        if capture_target is not None or cursor is None:
            return original(cls, device, capture_target)
        saved = cursor.numpy().copy()
        try:
            return original(cls, device, capture_target)
        finally:
            warmup_count = int(cursor.numpy()[0]) - int(saved[0])
            cursor.assign(saved)
            print(f'[Cable recording] Excluded {warmup_count} graph-warmup force samples.', flush=True)

    manager._capture_relaxed_graph = classmethod(capture)
    manager._cable_capture_recording_fix = True


def install_autostart():
    if os.environ.get('CABLE_AUTO_START', '1') != '1':
        return
    import torch
    from isaaclab_teleop.isaac_teleop_device import IsaacTeleopDevice
    from isaacteleop.retargeting_engine.tensor_types import ControllerInputIndex

    if getattr(IsaacTeleopDevice, '_cable_autostart_installed', False):
        return
    original = IsaacTeleopDevice.advance

    def advance(self, *args, **kwargs):
        action = original(self, *args, **kwargs)
        if self._cfg.app_name != 'CableMultilineFranka':
            return action
        gate = getattr(self, '_cable_start_gate', None)
        if gate is None:
            gate = self._cable_start_gate = StartGate()
        controller = self._session_lifecycle.last_right_controller
        valid = action is not None and bool(torch.isfinite(action).all())
        valid = valid and controller is not None and not controller.is_none
        valid = valid and bool(controller[ControllerInputIndex.GRIP_IS_VALID])
        button = valid and float(controller[ControllerInputIndex.SECONDARY_CLICK]) > 0.5
        events = self.last_control_events
        command, allow = gate.update(valid, button, time.monotonic(),
                                     active=events.is_active, reset=events.should_reset)
        if command == 'start':
            self.request_start()
            print('[Cable teleop] Right controller ready: automatic START. Right B pauses/resumes.', flush=True)
        elif command == 'stop':
            self.request_stop()
            print('[Cable teleop] Paused by right B or lost controller tracking.', flush=True)
        return action if allow else None

    IsaacTeleopDevice.advance = advance
    IsaacTeleopDevice._cable_autostart_installed = True


def install():
    from cable_robot_sample.physics.vbd import NewtonAllVBDManager
    from isaaclab_teleop.teleop_message_processor import TeleopMessageProcessor

    if not getattr(TeleopMessageProcessor, '_cable_command_log', False):
        original = TeleopMessageProcessor._apply_command_kind

        def apply_command(self, kind):
            if kind is not None:
                print(f'[Cable control] Official control command: {kind}', flush=True)
            return original(self, kind)

        TeleopMessageProcessor._apply_command_kind = apply_command
        TeleopMessageProcessor._cable_command_log = True

    install_capture_fix(NewtonAllVBDManager)
    install_autostart()
