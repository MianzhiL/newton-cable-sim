import newton
import warp as wp
from isaaclab_newton.physics import NewtonManager
from .rods import NewtonRodVBDManager


@wp.kernel
def compensate_robot_gravity(
    selected: wp.array(dtype=wp.int32), mass: wp.array(dtype=float),
    forces: wp.array(dtype=wp.spatial_vector), sign: float,
):
    body = wp.tid()
    if selected[body] != 0:
        forces[body] = forces[body] + wp.spatial_vector(0.0, 0.0, sign * mass[body] * 9.81, 0.0, 0.0, 0.0)


class NewtonAllVBDManager(NewtonRodVBDManager):
    separate_contact_colors = False
    original_color_groups = None

    @classmethod
    def _build_solver(cls, model, solver_cfg):
        cls.robot_mask = wp.array([int("/Robot/" in label) for label in model.body_label], dtype=wp.int32, device=model.device)
        cls.original_color_groups = [group.numpy().tolist() for group in model.body_color_groups]
        if cls.separate_contact_colors:
            groups = []
            for group in cls.original_color_groups:
                for robot in (False, True):
                    selected = [body for body in group if ("/Robot/" in model.body_label[body]) == robot]
                    if selected:
                        groups.append(wp.array(selected, dtype=wp.int32, device=model.device))
            model.body_color_groups = groups
        super()._build_solver(model, solver_cfg)
        cls.contact_previous_pose = wp.clone(NewtonManager._solver.body_q_prev)

    @classmethod
    def _step_solver(cls, state_0, state_1, control, contacts, substep_dt):
        model = NewtonManager.get_model()
        wp.launch(compensate_robot_gravity, dim=model.body_count,
                  inputs=[cls.robot_mask, model.body_mass, state_0.body_f, 1.0], device=model.device)
        wp.copy(cls.contact_previous_pose, NewtonManager._solver.body_q_prev)
        super()._step_solver(state_0, state_1, control, contacts, substep_dt)
        newton.eval_ik(model, state_1, state_1.joint_q, state_1.joint_qd)
        wp.launch(compensate_robot_gravity, dim=model.body_count,
                  inputs=[cls.robot_mask, model.body_mass, state_0.body_f, -1.0], device=model.device)


def straighten():
    solver = NewtonManager._solver
    solver.joint_rod_rest_kb_local.zero_()
    solver.joint_rod_rest_twist.zero_()
