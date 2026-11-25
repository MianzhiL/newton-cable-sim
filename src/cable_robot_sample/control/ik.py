from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction


class ControlRateDifferentialIKAction(DifferentialInverseKinematicsAction):
    def process_actions(self, actions):
        super().process_actions(actions)
        self._pending_solve = True

    def apply_actions(self):
        if self._pending_solve:
            super().apply_actions()
            self._pending_solve = False
