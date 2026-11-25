import os
import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.utils.configclass import configclass
from isaaclab_newton.sim.schemas.schemas_cfg import NewtonCollisionPropertiesCfg

ROBOT_BASE = (-0.45, -0.15, 0.225)


def robot_state(env):
    robot = env.scene["robot"]
    return torch.cat((robot.data.joint_pos.torch, robot.data.joint_vel.torch), dim=-1)


@configclass
class RobotCableSceneCfg:
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.environ["CABLE_ROBOT_USD"],
            collision_props=NewtonCollisionPropertiesCfg(contact_gap=0.001, contact_margin=0.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ROBOT_BASE,
            joint_pos={"fr3_joint1": 0.0, "fr3_joint2": -0.5, "fr3_joint3": 0.0, "fr3_joint4": -2.2, "fr3_joint5": 0.0, "fr3_joint6": 1.7, "fr3_joint7": 0.7854, "fr3_finger_joint.*": 0.025},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["fr3_joint[1-7]"],
                joint_effort_limit={"fr3_joint[1-4]": 87.0, "fr3_joint[5-7]": 12.0},
                joint_velocity_limit=2.0, stiffness=1500.0, damping=80.0, armature=0.001,
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=["fr3_finger_joint.*"], joint_effort_limit=20.0,
                joint_velocity_limit=0.1, stiffness=1500.0, damping=100.0, armature=0.00001,
            ),
        },
    )
    pedestal = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Pedestal",
        spawn=sim_utils.CuboidCfg(
            size=(0.22, 0.22, 0.225), collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.28, 0.32)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(ROBOT_BASE[0], ROBOT_BASE[1], 0.1125)),
    )
