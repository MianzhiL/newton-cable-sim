import torch
from rsl_rl.algorithms import PPO


class SeparatedOptimizers:
    def __init__(self, algorithm, critic_learning_rate, actor_lr_limit):
        self.algorithm = algorithm
        self.actor_lr_limit = actor_lr_limit
        self.actor = torch.optim.Adam(algorithm.actor.parameters(), lr=min(algorithm.learning_rate, actor_lr_limit))
        self.critic = torch.optim.Adam(algorithm.critic.parameters(), lr=critic_learning_rate)

    @property
    def param_groups(self):
        return self.actor.param_groups

    def zero_grad(self):
        self.actor.zero_grad(set_to_none=True)
        self.critic.zero_grad(set_to_none=True)

    def step(self):
        parameters = list(self.algorithm.actor.parameters()) + list(self.algorithm.critic.parameters())
        if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()) for parameter in parameters):
            raise RuntimeError('Nonfinite gradient; neither optimizer was stepped')
        self.algorithm.learning_rate = min(self.algorithm.learning_rate, self.actor_lr_limit)
        for group in self.actor.param_groups:
            group['lr'] = self.algorithm.learning_rate
        self.actor.step()
        self.critic.step()

    def state_dict(self):
        return dict(format='risk_v4_separated_adam', actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                    actor_lr_limit=self.actor_lr_limit)

    def load_state_dict(self, state):
        if state.get('format') != 'risk_v4_separated_adam' or state['actor_lr_limit'] != self.actor_lr_limit:
            raise ValueError('Legacy or incompatible optimizer state requires explicit migration')
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        self.algorithm.learning_rate = self.actor.param_groups[0]['lr']


class IndependentCriticPPO(PPO):
    def __init__(self, *args, critic_learning_rate=3e-4, actor_lr_limit=1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        if self.rnd or self.symmetry or self.actor.is_recurrent or self.critic.is_recurrent:
            raise ValueError('This adapter is validated for separate feed-forward actor/critic only')
        if not 0 < critic_learning_rate <= 1e-2 or not 0 < actor_lr_limit <= 1e-2:
            raise ValueError('Invalid independent learning rates')
        if {id(value) for value in self.actor.parameters()} & {id(value) for value in self.critic.parameters()}:
            raise ValueError('Actor and critic must not share parameters')
        self.optimizer = SeparatedOptimizers(self, critic_learning_rate, actor_lr_limit)

