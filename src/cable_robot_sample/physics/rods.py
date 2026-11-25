import math
import re

import newton
import torch
import warp as wp

from isaaclab_newton.physics.vbd_manager import NewtonVBDManager
from isaaclab_newton.physics import NewtonManager
import isaaclab.envs.mdp as env_mdp


class RodModelBuilder(newton.ModelBuilder):
    material = None
    finalize_hook = None
    rod_parameter_hook = None

    def finalize(self, *args, **kwargs):
        if type(self).finalize_hook is not None and not getattr(self, "_reference_prepared", False):
            type(self).finalize_hook(self)
            self._reference_prepared = True
        return super().finalize(*args, **kwargs)

    def add_joint_d6(self, parent, child, **kwargs):
        body_names = (self.body_label[parent], self.body_label[child]) if parent >= 0 else ("", "")
        if not all(re.search(r"/link_\d+$", name) for name in body_names):
            return super().add_joint_d6(parent, child, **kwargs)
        if self.material is None:
            raise RuntimeError("Rod material was not configured")
        alignment = wp.transform(wp.vec3(0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi / 2))
        parameters = {}
        for mode in ("stretch", "shear", "bend", "twist"):
            unit = "N_per_m" if mode in ("stretch", "shear") else "Nm_per_rad"
            damping_unit = "N_s_per_m" if mode in ("stretch", "shear") else "Nm_s_per_rad"
            parameters[f"{mode}_stiffness"] = self.material[f"{mode}_stiffness_{unit}"]
            parameters[f"{mode}_damping"] = self.material[f"{mode}_damping_{damping_unit}"]
        if any(not math.isfinite(value) or value < 0 for value in parameters.values()):
            raise ValueError("Rod stiffness and damping must be finite and nonnegative")
        if type(self).rod_parameter_hook is not None:
            parameters.update(type(self).rod_parameter_hook(self, parent, child))
        return self.add_joint_rod(
            parent, child,
            parent_xform=kwargs["parent_xform"] * alignment,
            child_xform=kwargs["child_xform"] * alignment,
            label=kwargs.get("label"),
            collision_filter_parent=kwargs.get("collision_filter_parent"),
            enabled=kwargs.get("enabled", True),
            **parameters,
        )


class NewtonRodVBDManager(NewtonVBDManager):
    @classmethod
    def create_builder(cls, up_axis=None, **kwargs):
        defaults = super().create_builder(up_axis=up_axis, **kwargs)
        builder = RodModelBuilder(up_axis=defaults.up_axis, **kwargs)
        builder.default_shape_cfg = defaults.default_shape_cfg
        builder.default_bvh_cfg = defaults.default_bvh_cfg
        cls._register_builder_attributes(builder)
        if RodModelBuilder.rod_parameter_hook is not None and "vbd:dahl_eps_max" not in builder.custom_attributes:
            from newton.solvers import SolverVBD

            SolverVBD.register_custom_attributes(builder)
        return builder


def reset_rod_scene(env, env_ids=None):
    env_mdp.reset_scene_to_default(env, env_ids)
    model = NewtonManager.get_model()
    selected = torch.arange(env.num_envs, device=env.device) if env_ids is None else torch.as_tensor(env_ids, device=env.device)
    body_worlds = wp.to_torch(model.body_world)
    mask = torch.isin(body_worlds, selected)
    for state in (NewtonManager.get_state_0(), NewtonManager.get_state_1()):
        if state is not None:
            wp.to_torch(state.body_q)[mask] = wp.to_torch(model.body_q)[mask]
            wp.to_torch(state.body_qd)[mask] = 0.0
            wp.to_torch(state.body_f)[mask] = 0.0
    NewtonManager.invalidate_body_state(wp.from_torch(selected.to(torch.int32).contiguous(), dtype=wp.int32))
