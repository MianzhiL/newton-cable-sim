import json
from pathlib import Path
import numpy as np


def build_config(args):
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg
    from isaaclab.controllers import DifferentialIKControllerCfg
    from isaaclab.envs import ManagerBasedEnvCfg
    import isaaclab.envs.mdp as mdp
    from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg
    from isaaclab.scene import InteractiveSceneCfg
    import isaaclab.sim as sim
    from isaaclab.utils.configclass import configclass
    from isaaclab_visualizers.kit import KitVisualizerCfg
    from cable_robot_sample.physics.config import CableVBDSolverCfg
    from cable_robot_sample.scene.robot import RobotCableSceneCfg, robot_state
    from cable_robot_sample.physics.rods import RodModelBuilder, reset_rod_scene
    from cable_robot_sample.physics.vbd import NewtonAllVBDManager
    from isaaclab_newton.physics import NewtonCfg, NewtonShapeCfg

    @configclass
    class Actions:
        arm = mdp.DifferentialInverseKinematicsActionCfg(asset_name='robot', joint_names=['fr3_joint[1-7]'], body_name='fr3_hand', controller=DifferentialIKControllerCfg(command_type='pose', use_relative_mode=False, ik_method='dls', ik_params={'lambda_val': args.ik_damping}))
        hand = mdp.BinaryJointPositionActionCfg(asset_name='robot', joint_names=['fr3_finger_joint.*'], open_command_expr={'fr3_finger_joint.*': 0.04}, close_command_expr={'fr3_finger_joint.*': 0.0})

    actions = Actions()
    if args.ik_update_rate == 'control':
        from cable_robot_sample.control.ik import ControlRateDifferentialIKAction
        actions.arm.class_type = ControlRateDifferentialIKAction

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

    scene = InteractiveSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=False)
    scene.robot = RobotCableSceneCfg().robot.copy()
    scene.robot.init_state.pos = (0.0, -0.55, 0.225)
    scene.robot.init_state.rot = (0.0, 0.0, 0.7071067811865475, 0.7071067811865476)
    scene.robot.init_state.joint_pos['fr3_joint1'] = 0.9
    scene.robot.init_state.joint_pos['fr3_finger_joint.*'] = 0.04
    scene.robot.actuators['hand'].stiffness = 8000.0
    scene.robot.actuators['hand'].joint_effort_limit = 500.0
    scene.pedestal = RobotCableSceneCfg().pedestal.copy()
    scene.pedestal.init_state.pos = (0.0, -0.55, 0.1125)
    scene.datacenter = AssetBaseCfg(prim_path='{ENV_REGEX_NS}/Datacenter', spawn=sim.UsdFileCfg(usd_path=str(args.assets / 'rack.usda')))
    manifest = json.loads((args.assets / 'manifest.json').read_text())
    for index, cable in enumerate(manifest['cables'], 1):
        anchor = cable['bodies']['sc_plug_link']
        setattr(scene, f'cable{index}', ArticulationCfg(prim_path='{ENV_REGEX_NS}/Cable' + str(index), spawn=sim.UsdFileCfg(usd_path=str(args.assets / f'Cable{index}.usdc'), articulation_props=sim.ArticulationRootPropertiesCfg(enabled_self_collisions=True)), init_state=ArticulationCfg.InitialStateCfg(pos=tuple(anchor['position']), rot=tuple(anchor['quaternion_xyzw'])), actuators={}))
    scene.ground = AssetBaseCfg(prim_path='/World/Ground', spawn=sim.GroundPlaneCfg(color=(0.16, 0.17, 0.19)))
    scene.light = AssetBaseCfg(prim_path='/World/Light', spawn=sim.DomeLightCfg(intensity=1600.0, color=(0.85, 0.9, 1.0)))
    NewtonAllVBDManager.separate_contact_colors = True
    material_path = args.assets / 'rod_material.json'
    if not material_path.is_file():
        material_path = Path(__file__).resolve().parents[1] / 'rod_material.json'
    RodModelBuilder.material = json.loads(material_path.read_text())
    lengths_by_name = {cable['name']: cable['segment_lengths_m'] for cable in manifest['cables']}

    def rod_parameters(builder, parent, child):
        parent_path = builder.body_label[parent].split('/')
        child_path = builder.body_label[child].split('/')
        lengths = lengths_by_name[parent_path[-2]]
        spacing = 0.5 * (lengths[int(parent_path[-1].split('_')[-1])] + lengths[int(child_path[-1].split('_')[-1])])
        ratio = 0.047 / spacing
        parameters = {}
        for mode in ('stretch', 'shear', 'bend', 'twist'):
            units = ('N_per_m', 'N_s_per_m') if mode in ('stretch', 'shear') else ('Nm_per_rad', 'Nm_s_per_rad')
            parameters[mode + '_stiffness'] = RodModelBuilder.material[mode + '_stiffness_' + units[0]] * ratio
            parameters[mode + '_damping'] = RodModelBuilder.material[mode + '_damping_' + units[1]] * ratio
        return parameters

    RodModelBuilder.rod_parameter_hook = rod_parameters
    physics = NewtonCfg(solver_cfg=CableVBDSolverCfg(class_type=NewtonAllVBDManager, iterations=20), num_substeps=1, collision_decimation=0, default_shape_cfg=NewtonShapeCfg(gap=0.00006, margin=0.0, ke=1e5, kd=10.0, mu=0.6))
    cfg = ManagerBasedEnvCfg(scene=scene, actions=actions, observations=Observations(), events=Events(), decimation=16, sim=sim.SimulationCfg(dt=1 / 960, render_interval=16, gravity=(0.0, 0.0, -9.81), physics=physics), ui_window_class_type=None)
    cfg.sim.visualizer_cfgs = [] if args.no_render else [KitVisualizerCfg(window_width=1440, window_height=1080)]
    cfg.viewer.eye = (1.45, -2.25, 1.48)
    cfg.viewer.lookat = (-0.12, 0.12, 0.95)
    return cfg


class CableAppearance:
    def __init__(self, model, manifest):
        import isaaclab.sim as sim
        from pxr import UsdGeom
        from cable_robot_sample.scene.materials import material

        self.stage = sim.get_current_stage()
        self.curves = []
        for cable in manifest['cables']:
            name = cable['name']
            ids = [next(index for index, label in enumerate(model.body_label) if label.endswith(f'/{name}/link_{segment}')) for segment in range(cable['rod_count'])]
            path = '/World/CableAppearance/' + name
            curve = UsdGeom.BasisCurves.Define(self.stage, path)
            curve.CreateTypeAttr('cubic')
            curve.CreateBasisAttr('catmullRom')
            curve.CreateWrapAttr('nonperiodic')
            curve.CreateCurveVertexCountsAttr([len(ids) + 4])
            curve.CreateWidthsAttr([2 * cable['jacket_radius_m']])
            curve.SetWidthsInterpolation('constant')
            from pxr import UsdShade
            UsdShade.MaterialBindingAPI.Apply(curve.GetPrim()).Bind(material(self.stage, path + '/Material', cable['jacket_color']))
            lengths = cable['segment_lengths_m']
            self.curves.append((curve, ids, 0.5 * lengths[0], 0.5 * lengths[-1]))

    def update(self, poses):
        from pxr import Vt
        from scipy.spatial.transform import Rotation

        for curve, ids, first_half_length, last_half_length in self.curves:
            selected = poses[ids]
            axes = Rotation.from_quat(selected[:, 3:]).apply(np.tile([1.0, 0.0, 0.0], (len(ids), 1)))
            start = selected[0, :3] - first_half_length * axes[0]
            end = selected[-1, :3] + last_half_length * axes[-1]
            points = np.vstack([start, start, selected[:, :3], end, end]).astype(np.float32)
            curve.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
            curve.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.vstack([points.min(axis=0) - 0.004, points.max(axis=0) + 0.004])))
