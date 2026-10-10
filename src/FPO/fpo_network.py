import torch
import torch.nn as nn


# VectorActorCritic creates the neural network, adds cfm_loss and sample action and value
class VectorActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        # 3 inputs: State, Noisy Action, Flow Time.
        # Supposedly Mish is better for flow matching
        self.VectorFieldActor = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1, 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
            nn.Linear(256, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, state, noisy_action, time):
        return self.VectorFieldActor(torch.cat([state, noisy_action, time], dim=-1))

    def cfm_loss(self, states, actions):
        # calculates the conditional flowm matching loss
        batch_size = states.shape[0]
        device = states.device

        # Sample random action with noise, random timestep between 0 and 1
        eps_i = torch.randn_like(actions)
        t = torch.rand((batch_size, 1), device=device)

        # Straight line interpolation
        x_t = (1 - t) * eps_i + t * actions

        v_target = actions - eps_i
        v_prediction = self.forward(states, x_t, t)

        return torch.sum((v_prediction - v_target) ** 2, dim=-1)

    def sample_action(self, state, num_steps=10):
        batch_size = state.shape[0]
        device = state.device

        # using Euler's method in ODEs to go noise into action
        x_t = torch.randn((batch_size, self.action_dim), device=device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((batch_size, 1), i * dt, device=device)
            velocity = self.forward(state, x_t, t)
            x_t = x_t + velocity * dt

        return x_t

    def get_value(self, state):
        return self.critic(state)

    def get_action_value(self, state):
        return self.sample_action(state), self.get_value(state).squeeze(-1)
