import torch
import torch.nn as nn
from torch.distributions import Normal


class VectorActorField(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        # last dim is time
        # Supposedly Mish is better for flow matching
        self.VectorFieldActor = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1, 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
            nn.Linear(256, action_dim),
        )

        # Actor (log) standard deviation parameter.
        # Reminder: log because it handles tiny probabilities better
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, state, noisy_action, time):
        velocity = self.net(torch.cat([state, noisy_action, time], dim=-1))
        return velocity

    def get_value(self, state):
        return self.critic(state)

    def get_action_value(self, state, action=None):

        action_mean = self.actor_mean(state)
        # Reminder: batches can go through NN, so need to expand
        action_std = torch.exp(self.actor_log_std.expand_as(action_mean))

        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        # Sum log probabilities over each batch sample
        log_prob = probs.log_prob(action).sum(axis=-1)
        entropy = probs.entropy().sum(axis=-1)

        return action, log_prob, entropy, self.get_value(state).squeeze(-1)
