import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ('train', 'evaluate', 'play'):
        from cable_robot_sample.learning.runner import main as learning
        learning(sys.argv[1:])
        return
    parser = argparse.ArgumentParser(description='Cable manipulation sample: inspect, run, teleoperate and replay')
    commands = parser.add_subparsers(dest='command', required=True)
    for name, help_text in [('train', 'Train the cable insertion policy'), ('evaluate', 'Evaluate a policy on held-out initial states'),
                            ('play', 'Run and record a policy in the scene')]:
        commands.add_parser(name, help=help_text)
    rack = commands.add_parser('configure-rack', help='Reference a locally obtained official rack USD without copying it')
    rack.add_argument('--assets', type=Path, required=True)
    rack.add_argument('--source', type=Path, required=True)
    assets = commands.add_parser('validate-assets', help='Check a user-supplied asset pack')
    assets.add_argument('--assets', type=Path, required=True)
    assets.add_argument('--robot', type=Path, required=True)
    assets.add_argument('--usd-dependencies', action='store_true')
    commands.add_parser('replay', help='List recorded segments or render a video; use replay --help')
    for name in ('run', 'teleop'):
        launch = commands.add_parser(name, help='Hold the robot pose' if name == 'run' else 'Run the official Isaac Teleop agent')
        launch.add_argument('--assets', type=Path, required=True)
        launch.add_argument('--robot', type=Path, required=True)
        launch.add_argument('--output', type=Path, required=True)
        launch.add_argument('--device', default='cuda:0')
        launch.add_argument('--grip-friction', type=float, default=1.0)
        launch.add_argument('--no-forces', action='store_true')
        if name == 'run':
            launch.add_argument('--steps', type=int, default=600)
            launch.add_argument('--reset-at', type=int)
            launch.add_argument('--headless', action='store_true')
        else:
            launch.add_argument('--sdk', type=Path, required=True)
    if len(sys.argv) > 1 and sys.argv[1] == 'replay':
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        from cable_robot_sample.recording.video import main as replay
        replay()
        return
    args = parser.parse_args()
    if args.command == 'configure-rack':
        from cable_robot_sample.rack_source import configure_rack
        print(json.dumps(configure_rack(args.assets, args.source), indent=2))
        return
    from cable_robot_sample.assets import validate_assets
    report = validate_assets(args.assets, args.robot)
    if args.command == 'validate-assets':
        if args.usd_dependencies:
            from cable_robot_sample.assets import check_usd_dependencies
            report.update(check_usd_dependencies(args.assets, args.robot))
        print(json.dumps(report, indent=2))
        return
    if not math.isfinite(args.grip_friction) or args.grip_friction < 0:
        parser.error('--grip-friction must be finite and nonnegative')
    if os.environ.get('OMNI_KIT_ACCEPT_EULA', '').upper() != 'YES':
        parser.error('Review and accept the installed Isaac Sim/Kit license yourself, then set OMNI_KIT_ACCEPT_EULA=YES in this session. No simulation started.')
    if args.command == 'run':
        if args.steps < 1 or (args.reset_at is not None and not 0 < args.reset_at < args.steps):
            parser.error('--steps must be positive and --reset-at must be within the run')
    else:
        agent = args.sdk.resolve() / 'scripts/environments/teleoperation/teleop_se3_agent.py'
        if not agent.is_file():
            parser.error('Official teleop agent not found under --sdk')
        if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
            parser.error('Launch teleop in a graphical Linux session')
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    args.output = output
    os.environ.update(CABLE_TELEOP_ASSETS=str(args.assets.resolve()), CABLE_ROBOT_USD=str(args.robot.resolve()),
        CABLE_TELEOP_OUTPUT=str(output), CABLE_DEVICE=args.device, CABLE_RECORD_FORCES='0' if args.no_forces else '1',
        CABLE_RECORD_TRAJECTORY='1', CABLE_GRIP_FRICTION=str(args.grip_friction),
        CABLE_XR_ENABLED='1' if args.command == 'teleop' else '0',
        CABLE_HEADLESS='1' if getattr(args, 'headless', False) else '0')
    (output / 'launch.json').write_text(json.dumps(vars(args), default=str, indent=2), encoding='utf-8')
    print(f'Session: {output}', flush=True)
    if args.command == 'teleop':
        command = [sys.executable, str(agent), '--external_callback', 'cable_robot_sample.teleop.register.register',
            '--task', 'Cable-Robot-Sample-Franka-v0', '--num_envs', '1', '--visualizer', 'kit',
            '--xr', '--cloudxr_env', 'cloudxrjs', '--device', args.device]
        raise SystemExit(subprocess.call(command, cwd=args.sdk.resolve()))
    from cable_robot_sample.launch import run
    run(args)
