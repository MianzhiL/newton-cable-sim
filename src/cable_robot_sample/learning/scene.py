import json
from pathlib import Path

import numpy as np

from .reference import file_digest


JOINT_NAMES = tuple(f'fr3_joint{index}' for index in range(1, 8)) + ('fr3_finger_joint1', 'fr3_finger_joint2')


def validate_scene_assets(directory, robot):
    directory = Path(directory)
    material = Path(__file__).resolve().parents[1] / 'rod_material.json'
    paths = dict(cable=directory / 'cable/supported_cable.usda',
                 rack=directory / 'scene/scene.usdc', robot=Path(robot), material=material)
    provenance = json.loads((directory / 'task.json').read_text())
    hashes = {}
    for name, path in paths.items():
        expected = provenance['sha256'][name]
        if file_digest(path) != expected:
            raise ValueError(f'Task asset checksum mismatch: {name}')
        hashes[name] = expected
    layout = json.loads((directory / 'cable/layout.json').read_text())
    manifest = json.loads((directory / 'scene/manifest.json').read_text())
    ports = [value for value in manifest.values() if isinstance(value, dict) and 'port_proxy_opening_m' in value]
    if len(ports) != 1 or ports[0]['port_proxy_opening_m'] != [.022, .012] or ports[0]['port_proxy_depth_m'] != .06:
        raise ValueError('This task requires the 22x12x60mm proxy corridor')
    if manifest['scene_sha256'] != hashes['rack']:
        raise ValueError('Rack manifest checksum mismatch')
    for name, size in [('fixed_position_m', 3), ('root_rotation_xyzw', 4)]:
        value = np.asarray(layout[name])
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError(f'Invalid scene layout: {name}')
    return paths, layout, hashes


def build_config(args, reference):
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg
    from isaaclab.envs import ManagerBasedEnvCfg
    import isaaclab.envs.mdp as mdp
    from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg
    from isaaclab.scene import InteractiveSceneCfg
    import isaaclab.sim as sim
    from isaaclab.utils.configclass import configclass
    from isaaclab_newton.physics import NewtonCfg, NewtonShapeCfg
    from cable_robot_sample.scene.robot import RobotCableSceneCfg, robot_state
    from cable_robot_sample.physics.config import CableVBDSolverCfg
    from cable_robot_sample.physics.rods import RodModelBuilder, reset_rod_scene
    from cable_robot_sample.physics.vbd import NewtonAllVBDManager

    paths, layout, hashes = validate_scene_assets(args.task_assets, args.robot)

    @configclass
    class Actions:
        robot_joints = mdp.JointPositionActionCfg(asset_name='robot', joint_names=list(JOINT_NAMES),
                                                  scale=1., use_default_offset=False, preserve_order=True)

    @configclass
    class Observations:
        @configclass
        class State(ObservationGroupCfg):
            robot = ObservationTermCfg(func=robot_state)
            enable_corruption = False
            concatenate_terms = True
        state = State()

    @configclass
    class Events:
        reset = EventTermCfg(func=reset_rod_scene, mode='reset')

    scene = InteractiveSceneCfg(num_envs=args.num_envs, env_spacing=2., replicate_physics=True)
    scene.cable = ArticulationCfg(prim_path='{ENV_REGEX_NS}/Cable',
        spawn=sim.UsdFileCfg(usd_path=str(paths['cable'])), actuators={},
        init_state=ArticulationCfg.InitialStateCfg(pos=tuple(layout['fixed_position_m']),
                                                 rot=tuple(layout['root_rotation_xyzw'])))
    scene.robot = RobotCableSceneCfg().robot.copy()
    scene.robot.init_state.pos = (0., -.55, .225)
    from scipy.spatial.transform import Rotation
    scene.robot.init_state.rot = tuple(Rotation.from_euler('z', 90, degrees=True).as_quat())
    for name in JOINT_NAMES:
        joint = next(index for index, label in enumerate(reference.joint_labels) if label.endswith('/' + name))
        coordinate = int(reference.structure['joint_q_start'][joint])
        scene.robot.init_state.joint_pos[name] = float(reference.rest_coordinates[coordinate])
    scene.robot.init_state.joint_pos.pop('fr3_finger_joint.*', None)
    scene.robot.actuators['hand'].stiffness = 8000.
    scene.robot.actuators['hand'].joint_effort_limit = 500.
    scene.pedestal = RobotCableSceneCfg().pedestal.copy()
    scene.pedestal.init_state.pos = (0., -.55, .1125)
    scene.datacenter = AssetBaseCfg(prim_path='{ENV_REGEX_NS}/Datacenter',
                                   spawn=sim.UsdFileCfg(usd_path=str(paths['rack'])))
    scene.light = AssetBaseCfg(prim_path='/World/Light', spawn=sim.DomeLightCfg(intensity=1500.))
    RodModelBuilder.material = json.loads(paths['material'].read_text())
    RodModelBuilder.rod_parameter_hook = None
    NewtonAllVBDManager.separate_contact_colors = True
    physics = NewtonCfg(
        solver_cfg=CableVBDSolverCfg(class_type=NewtonAllVBDManager, iterations=20),
        num_substeps=1, collision_decimation=0,
        default_shape_cfg=NewtonShapeCfg(gap=.00006, margin=0., ke=1e5, kd=10., mu=.6))
    cfg = ManagerBasedEnvCfg(scene=scene, actions=Actions(), observations=Observations(), events=Events(),
        decimation=16, sim=sim.SimulationCfg(dt=1 / 960, render_interval=16, gravity=(0., 0., -9.81),
                                            device=args.device, physics=physics), ui_window_class_type=None)
    cfg.seed = args.seed
    if args.headless:
        cfg.sim.visualizer_cfgs = []
    cfg.viewer.eye = (-1.3, -1.5, 1.15)
    cfg.viewer.lookat = (0., .03, .72)
    cfg.cable_assets = str(args.task_assets)
    cfg.recording_assets = {'Datacenter': 'scene/scene.usdc', 'Cable': 'cable/supported_cable.usda'}
    return cfg, hashes


def configure_grip(model):
    import newton

    flags = model.shape_flags.numpy()
    collision = (flags & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
    fingers = [index for index, label in enumerate(model.shape_label)
               if collision[index] and '/Robot/' in label and
               ('/fr3_leftfinger/' in label or '/fr3_rightfinger/' in label)]
    heads = [index for index, label in enumerate(model.shape_label)
             if collision[index] and '/Cable/sfp_module_link/' in label]
    friction = model.shape_material_mu.numpy()
    if not fingers or not heads or not np.isfinite(friction[heads]).all() or np.any(friction[heads] <= 0):
        raise ValueError('Missing or invalid grip materials')
    if not np.allclose(friction[heads], friction[heads[0]]):
        raise ValueError('Connector shape friction is not uniform')
    friction[fingers] = 30. ** 2 / float(friction[heads[0]])
    model.shape_material_mu.assign(friction)
    return dict(effective_grip_friction=30., finger_shape_count=len(fingers), connector_shape_count=len(heads))
