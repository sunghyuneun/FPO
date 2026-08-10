import os
import torch as T
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from itertools import count
from torch.distributions.categorical import Normal
from collections import namedtuple

import gymnasium

env = gymnasium.make('CartPole-v1')

GAMMA = 0.99 # Discount Factor

class Policy(nn.Module):

    def __init__(self, state_dim = 4, action_dim = 6):
        # Inherits from nn.Module, initializes Actor and Critic
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh,
            nn.Linear(64,action_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,1)
        )

    def get_value(self, state):
        return self.critic(state)

    def get_action_and_value(self, state, action=None):
        logits = self.actor(state)
        probs = Normal(logits=logits)
        if action == None:
            action = probs.sample()
        
        pass


class Buffer():
    def __init__(self):
        # List of dict of a torch tensor
        self.buffer = []

    def add(self, state, action, reward, next_state, done):
        self.buffer.append({"state": state,
                                    "action": action,
                                    "reward": reward,
                                    "next_state": next_state,
                                    "done": done})


    def clear(self):
        self.buffer.clear()




model = Policy()
batches = Buffer()
# Optimizer nudges the weights based on gradient of backpropagation...
optimizer = optim.Adam(model.parameters(), lr=3e-2)


# Memory Buffer

# I need to hold those memories...?
# Collects state, action, log-prob of actions, rewards, "done" flags?


# Write logic that tells actor to pick an action based on state, steps, saves to buffer
def select_action(state):
    # Put the state in and find the stuff
    batches.states.append(state)
    state = T.from_numpy(state).float().unsqueeze(0)
    actor_logits, state_value = model.forward(state)

    m = Categorical(logits = actor_logits)

    # Turns it into a discrete distribution, and samples from one of them?
    action = m.sample()

    batches.saved_actions.append(action)
    batches.log_probs.append(m.log_prob(action))

    # Returns the action
    return action.item()

# Once I get a bunch of actions, eventually it will end.
# Then, i can calculate the reward, predicted value
def end_episode():
    pass

# Advantage Estimation
# Generalized Advantage Estimation


# PPO Update Logic
# Calculate ratio, apply clip function, entropy bonus?

def main():
    for i_episode in count(1):
        state, _ = env.reset()
        ep_reward = 0

        for t in range(1, 2048):
            action = select_action(state)

            state, reward, terminated, truncated, _ = env.step(action)

            batches.rewards.append(reward)
            ep_reward += reward
            model.done.append(terminated or truncated)
            if terminated or truncated:
                
                break
        
        # Now that episode is over, do backpropagation
        
if __name__ == '__main__':
    main()