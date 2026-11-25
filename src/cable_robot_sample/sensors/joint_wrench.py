import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import warp as wp
from newton.solvers import SolverVBD
from newton._src.solvers.vbd import rigid_vbd_kernels

OUTPUT = None
JOINTS = []
KERNEL = None
ARGUMENTS = []


def build_kernel():
    global KERNEL, ARGUMENTS
    source_path = Path(rigid_vbd_kernels.__file__)
    lock = json.loads((Path(__file__).resolve().parents[1] / 'runtime-lock.json').read_text())
    expected = lock['newton_source_sha256']['_src/solvers/vbd/rigid_vbd_kernels.py']
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != expected:
        raise RuntimeError('VBD wrench observer source mismatch. Review the new kernel API and validate the observer before updating runtime-lock.json, or run with --no-forces.')
    source = source_path.read_text()
    tree = ast.parse(source)
    function = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == 'evaluate_joint_force_hessian')
    arguments = function.args.args
    ARGUMENTS = [argument.arg for argument in arguments[2:]]
    declarations = [f'{argument.arg}: {ast.unparse(argument.annotation)}' for argument in arguments[2:]]
    signature = ',\n    '.join(['selected: wp.array[wp.int32]'] + declarations + ['output: wp.array[wp.spatial_vector]'])
    values = ', '.join(['body_index', 'joint_index'] + ARGUMENTS)
    generated = ('import warp as wp\nfrom newton._src.solvers.vbd.rigid_vbd_kernels import evaluate_joint_force_hessian\n'
        f'@wp.kernel\ndef collect_joint_wrench(\n    {signature}\n):\n'
        '    joint_index = selected[wp.tid()]\n    body_index = joint_child[joint_index]\n'
        f'    force, torque, hll, hal, haa = evaluate_joint_force_hessian({values})\n'
        '    output[body_index] = wp.spatial_vector(force[0], force[1], force[2], torque[0], torque[1], torque[2])\n')
    path = OUTPUT / 'generated_wrench_kernel.py'
    path.write_text(generated)
    spec = importlib.util.spec_from_file_location('generated_wrench_kernel', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    KERNEL = module.collect_joint_wrench
    return dict(path=str(source_path), sha256=hashlib.sha256(source_path.read_bytes()).hexdigest())


class ProbeVBD(SolverVBD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.probe_selected = wp.array(JOINTS, dtype=wp.int32, device=self.device)
        self.probe_body_parent_f = wp.zeros(self.model.body_count, dtype=wp.spatial_vector, device=self.device)

    def step(self, state_in, state_out, control, contacts, dt):
        self.probe_control = control if control is not None else self.model.control(clone_variables=False)
        super().step(state_in, state_out, self.probe_control, contacts, dt)

    def _finalize_rigid_bodies(self, state_in, state_out, dt):
        override = dict(body_q=state_in.body_q, body_q_prev=self.body_q_prev, body_q_rest=self.model.body_q,
            joint_target_q=self.probe_control.joint_target_q, joint_target_qd=self.probe_control.joint_target_qd,
            stab_alpha=self.rigid_joint_alpha, joint_compliant_alm=self.rigid_compliant_alm, dt=dt)
        values = []
        for name in ARGUMENTS:
            if name in override:
                values.append(override[name])
            elif hasattr(self, name):
                values.append(getattr(self, name))
            else:
                values.append(getattr(self.model, name))
        self.probe_body_parent_f.zero_()
        wp.launch(KERNEL, dim=len(JOINTS), inputs=[self.probe_selected] + values,
            outputs=[self.probe_body_parent_f], device=self.device)
        wp.copy(state_out.body_parent_f, self.probe_body_parent_f)
        super()._finalize_rigid_bodies(state_in, state_out, dt)
