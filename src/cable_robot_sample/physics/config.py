from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import VBDSolverCfg


@configclass
class CableVBDSolverCfg(VBDSolverCfg):
    rigid_compliant_alm: bool = True
    rigid_contact_history: bool = False
    rigid_body_contact_buffer_size: int = 512
    rigid_avbd_joint_alpha: float = 0.0
    rigid_avbd_gamma: float = 0.999
