import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        # Actor: Outputs mean of the distributions
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )

        # Actor (log) standard deviation parameter.
        # Reminder: log because it handles tiny probabilities better
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

        # Critic: Outputs value of the state
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def get_value(self, state):
        return self.critic(state)

    def get_action_and_value(self, state, action=None):
        action_mean = self.actor_mean(state)
        action_std = torch.exp(self.actor_log_std.expand_as(action_mean))

        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        # Sum log_probs over the action dimension (independent joint probabilities)
        log_prob = probs.log_prob(action).sum(axis=-1)
        entropy = probs.entropy().sum(axis=-1)

        return action, log_prob, entropy, self.get_value(state).squeeze(-1)
