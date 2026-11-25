import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import time


def parser():
    result = argparse.ArgumentParser(description='Train, evaluate or demonstrate cable insertion')
    result.add_argument('command', choices=('train', 'evaluate', 'play'))
    result.add_argument('--task-assets', type=Path, required=True)
    result.add_argument('--robot', type=Path, required=True)
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--num-envs', type=int, default=1)
    result.add_argument('--seed', type=int, default=42)
    result.add_argument('--checkpoint', type=Path)
    result.add_argument('--iterations', type=int, default=100)
    result.add_argument('--headless', action='store_true')
    result.add_argument('--initial-ids', type=int, nargs='+')
    result.add_argument('--record', action='store_true', help='Record one-environment play for the replay command')
    return result


def validate_arguments(args):
    if args.num_envs < 1 or args.iterations < 1:
        raise ValueError('Environment and iteration counts must be positive')
    if args.command != 'train' and args.checkpoint is None:
        raise ValueError('Evaluation and play require --checkpoint')
    if args.command == 'play' and args.num_envs != 1:
        raise ValueError('Play uses one environment')
    if args.record and args.command != 'play':
        raise ValueError('--record is a play option')
    if args.initial_ids is not None and (args.command == 'train' or len(set(args.initial_ids)) != len(args.initial_ids)):
        raise ValueError('Explicit initial IDs must be distinct evaluation states')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Launch one process; choose parallel worlds with --num-envs')
    for path in (args.task_assets, args.robot, args.checkpoint):
        if path is not None and not path.exists():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(f'Choose a new output directory: {args.output}')
    if os.environ.get('OMNI_KIT_ACCEPT_EULA', '').upper() != 'YES':
        raise RuntimeError('Review the installed runtime license and set OMNI_KIT_ACCEPT_EULA=YES after accepting it')


def save_checkpoint(algorithm, path, iteration):
    import torch
    state = algorithm.save()
    state.update(iter=iteration, infos=dict(task='cable-insertion'), torch_rng_state=torch.get_rng_state())
    temporary = path.with_suffix('.pt.partial')
    torch.save(state, temporary)
    os.replace(temporary, path)


def restore_checkpoint(algorithm, path, device):
    import torch
    state = torch.load(path, map_location=device, weights_only=True)
    algorithm.load(state, load_cfg=None, strict=True)
    if 'torch_rng_state' in state:
        torch.set_rng_state(state['torch_rng_state'].cpu())
    return int(state['iter'])


def train(env, args, app):
    import torch
    from rsl_rl.algorithms import PPO
    from .ppo import IndependentCriticPPO

    config = json.loads(Path(__file__).with_name('ppo.json').read_text())
    config['seed'] = args.seed
    (args.output / 'ppo.json').write_text(json.dumps(config, indent=2))
    configuration = copy.deepcopy(config)
    configuration['multi_gpu'] = None
    algorithm = PPO.construct_algorithm(env.get_observations(), env, configuration, args.device)
    if not isinstance(algorithm, IndependentCriticPPO):
        raise TypeError('Unexpected PPO implementation')
    iteration = 0
    if args.checkpoint:
        iteration = restore_checkpoint(algorithm, args.checkpoint, args.device)
    algorithm.train_mode()
    observation = env.get_observations()
    with (args.output / 'training.jsonl').open('x', buffering=1) as log:
        for iteration in range(iteration + 1, iteration + args.iterations + 1):
            started = time.perf_counter()
            with torch.inference_mode():
                for _ in range(config['num_steps_per_env']):
                    if not app.is_running():
                        raise RuntimeError('Application closed during training')
                    action = algorithm.act(observation)
                    observation, rewards, dones, extras = env.step(action)
                    algorithm.process_env_step(observation, rewards, dones, extras)
                algorithm.compute_returns(observation)
            collection_seconds = time.perf_counter() - started
            updated = time.perf_counter()
            losses = algorithm.update()
            elapsed = time.perf_counter() - started
            transitions = env.num_envs * config['num_steps_per_env']
            entry = dict(iteration=iteration, transitions=transitions,
                         collect_seconds=collection_seconds, update_seconds=time.perf_counter() - updated,
                         transitions_per_second=transitions / elapsed,
                         sim_seconds_per_wall_second_per_env=config['num_steps_per_env'] / (60 * elapsed),
                         losses={name: float(value) for name, value in losses.items()})
            log.write(json.dumps(entry, allow_nan=False) + '\n')
            print(json.dumps(entry), flush=True)
            if iteration % config['save_interval'] == 0:
                save_checkpoint(algorithm, args.output / f'model_{iteration}.pt', iteration)
    final = args.output / f'model_{iteration}.pt'
    save_checkpoint(algorithm, final, iteration)
    return dict(iteration=iteration, checkpoint=final.name)


def evaluate(env, args, app):
    import torch
    from .policy import load_policy
    from cable_robot_sample.recording.trajectory import TrajectoryRecorder

    policy = load_policy(args.checkpoint, args.device)
    ids = args.initial_ids if args.initial_ids is not None else list(env.bank.evaluation_ids)
    if args.command == 'play' and args.initial_ids is None:
        ids = ids[:1]
    if not set(ids).issubset(env.bank.evaluation_ids):
        raise ValueError('Choose held-out IDs from the bank manifest')
    results, recorder = [], None
    try:
        with torch.inference_mode():
            for offset in range(0, len(ids), env.num_envs):
                batch = ids[offset:offset + env.num_envs]
                assigned = batch + [batch[-1]] * (env.num_envs - len(batch))
                env.reset(initial_ids=assigned, split='test')
                if args.record:
                    if recorder is None:
                        env.base._cable_output = args.output
                        env.base._force_recorder = None
                        env.base._cable_step = 0
                        recorder = TrajectoryRecorder(env.base)
                    else:
                        recorder.new_segment()
                    recorder.capture('initial')
                finished = set()
                for _ in range(env.max_episode_length):
                    if not app.is_running():
                        raise RuntimeError('Application closed before evaluation completed')
                    parked = sorted(finished | set(range(len(batch), env.num_envs)))
                    if parked:
                        env.reset(selected=parked, initial_ids=[assigned[world] for world in parked], split='test')
                    action = policy(env.get_observations()['policy'])
                    if parked:
                        action[parked] = 0
                    _, _, _, extras = env.step(action)
                    if recorder is not None:
                        env.base._cable_step += 1
                        recorder.capture('step', env.snapshot.targets)
                    for record in extras['episodes']:
                        world = record['world']
                        if world < len(batch) and world not in finished:
                            finished.add(world)
                            results.append(record)
                            with (args.output / 'evaluation.jsonl').open('a') as stream:
                                stream.write(json.dumps(record, allow_nan=False) + '\n')
                    if len(finished) == len(batch):
                        break
        if len(results) != len(ids):
            raise RuntimeError('Incomplete evaluation; no aggregate success rate emitted')
        successes = sum(record['reason'] == 'success' for record in results)
        return dict(episodes=len(results), successes=successes, success_rate=successes / len(results),
                    initial_ids=ids, deterministic=True)
    finally:
        if recorder is not None:
            recorder.close()


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    validate_arguments(args)
    for name in ('task_assets', 'robot', 'output', 'checkpoint'):
        path = getattr(args, name)
        if path is not None:
            setattr(args, name, path.resolve())
    for name, version in [('rsl-rl-lib', '5.4.1'), ('tensordict', '0.13.0')]:
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f'This task uses {name}=={version}')
    args.reference = args.task_assets / 'reference'
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'launch.json').write_text(json.dumps(vars(args), default=str, indent=2))
    os.environ.update(CABLE_ROBOT_USD=str(args.robot), CABLE_HEADLESS='1' if args.headless else '0')
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=args.headless, visualizer='none' if args.headless else 'kit', device=args.device)
    app, base = launcher.app, None
    report = dict(status='failed')
    started = time.perf_counter()
    try:
        import newton
        import torch
        from isaaclab.envs import ManagerBasedEnv
        from isaaclab_newton.physics import NewtonManager
        from .bank import HeldStateBank
        from .reference import HeldReference
        from .scene import build_config, configure_grip
        from .environment import InsertionEnv
        from cable_robot_sample.physics.vbd import NewtonAllVBDManager, straighten
        from cable_robot_sample.teleop.compatibility import install_capture_fix

        torch.manual_seed(args.seed)
        bank = HeldStateBank(args.task_assets / 'bank', args.task_assets / 'bank/admission.json')
        reference = HeldReference(args.reference, args.reference)
        install_capture_fix(NewtonAllVBDManager)
        cfg, hashes = build_config(args, reference)
        base = ManagerBasedEnv(cfg)
        base.reset()
        straighten()
        configure_grip(NewtonManager.get_model())
        NewtonManager._solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        env = InsertionEnv(base, reference, bank, args)
        (args.output / 'task-config.json').write_text(json.dumps(dict(env.cfg, asset_sha256=hashes), indent=2))
        if args.command == 'train':
            env.reset()
            result = train(env, args, app)
        else:
            result = evaluate(env, args, app)
        report.update(status='completed', **result)
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        report['wall_seconds_after_app_launch'] = time.perf_counter() - started
        (args.output / 'result.json').write_text(json.dumps(report, indent=2))
        try:
            if base is not None:
                base.close()
        finally:
            app.close(exit_code=0 if report['status'] == 'completed' else 1)
