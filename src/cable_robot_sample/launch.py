import json
import time


def run(args):
    started = time.perf_counter()
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=args.headless, visualizer='none' if args.headless else 'kit', device=args.device)
    app = launcher.app
    env = None
    report = {'status': 'started', 'control_steps': 0, 'reset_requested': args.reset_at}
    try:
        import torch
        from cable_robot_sample.teleop.compatibility import install_capture_fix
        from cable_robot_sample.physics.vbd import NewtonAllVBDManager
        from cable_robot_sample.teleop.reset import make_config
        from cable_robot_sample.teleop.recorded import RecordedCableEnv

        install_capture_fix(NewtonAllVBDManager)
        env = RecordedCableEnv(make_config())
        env.reset()
        arm = env.action_manager.get_term('arm')

        def hold_action():
            position, quaternion = arm._compute_frame_pose()
            return torch.cat((position, quaternion, torch.ones((1, 1), device=env.device)), dim=-1).clone()

        action = hold_action()
        with torch.inference_mode():
            for step in range(args.steps):
                if not app.is_running():
                    raise RuntimeError('Application closed before requested steps completed')
                if step == args.reset_at:
                    env.reset()
                    action = hold_action()
                env.step(action)
                report['control_steps'] += 1
                if env._cable_fault:
                    raise RuntimeError('Physics guard paused this run; inspect events.jsonl and fault snapshot')
        report['status'] = 'completed'
    except Exception as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['wall_seconds_including_startup'] = time.perf_counter() - started
        try:
            if env is not None:
                env.close()
        finally:
            (args.output / 'run-result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            app.close(exit_code=0 if report['status'] == 'completed' else 1)
